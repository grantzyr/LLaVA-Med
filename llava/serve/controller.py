"""
A controller manages distributed workers.
It sends worker addresses to clients.
"""
import argparse
import asyncio
import dataclasses
from enum import Enum, auto
import json
import logging
import time
from typing import List, Union
import threading
import uuid

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np
import requests
import uvicorn

from llava.constants import CONTROLLER_HEART_BEAT_EXPIRATION
from llava.utils import build_logger, server_error_msg

logger = build_logger("controller", "controller.log")

# OpenAI API compatible models (reuse from worker)
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 256
    stream: bool = False
    stop: Union[str, List[str], None] = None

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: dict
    finish_reason: str = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

class DispatchMethod(Enum):
    LOTTERY = auto()
    SHORTEST_QUEUE = auto()

    @classmethod
    def from_str(cls, name):
        if name == "lottery":
            return cls.LOTTERY
        elif name == "shortest_queue":
            return cls.SHORTEST_QUEUE
        else:
            raise ValueError(f"Invalid dispatch method")


@dataclasses.dataclass
class WorkerInfo:
    model_names: List[str]
    speed: int
    queue_length: int
    check_heart_beat: bool
    last_heart_beat: str


def heart_beat_controller(controller):
    while True:
        time.sleep(CONTROLLER_HEART_BEAT_EXPIRATION)
        controller.remove_stable_workers_by_expiration()


class Controller:
    def __init__(self, dispatch_method: str):
        # Dict[str -> WorkerInfo]
        self.worker_info = {}
        self.dispatch_method = DispatchMethod.from_str(dispatch_method)

        self.heart_beat_thread = threading.Thread(
            target=heart_beat_controller, args=(self,))
        self.heart_beat_thread.start()

        logger.info("Init controller")

    def register_worker(self, worker_name: str, check_heart_beat: bool,
                        worker_status: dict):
        if worker_name not in self.worker_info:
            logger.info(f"Register a new worker: {worker_name}")
        else:
            logger.info(f"Register an existing worker: {worker_name}")

        if not worker_status:
            worker_status = self.get_worker_status(worker_name)
        if not worker_status:
            return False

        self.worker_info[worker_name] = WorkerInfo(
            worker_status["model_names"], worker_status["speed"], worker_status["queue_length"],
            check_heart_beat, time.time())

        logger.info(f"Register done: {worker_name}, {worker_status}")
        return True

    def get_worker_status(self, worker_name: str):
        try:
            r = requests.post(worker_name + "/worker_get_status", timeout=5)
        except requests.exceptions.RequestException as e:
            logger.error(f"Get status fails: {worker_name}, {e}")
            return None

        if r.status_code != 200:
            logger.error(f"Get status fails: {worker_name}, {r}")
            return None

        return r.json()

    def remove_worker(self, worker_name: str):
        del self.worker_info[worker_name]

    def refresh_all_workers(self):
        old_info = dict(self.worker_info)
        self.worker_info = {}

        for w_name, w_info in old_info.items():
            if not self.register_worker(w_name, w_info.check_heart_beat, None):
                logger.info(f"Remove stale worker: {w_name}")

    def list_models(self):
        model_names = set()

        for w_name, w_info in self.worker_info.items():
            model_names.update(w_info.model_names)

        return list(model_names)

    def get_worker_address(self, model_name: str):
        if self.dispatch_method == DispatchMethod.LOTTERY:
            worker_names = []
            worker_speeds = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_speeds.append(w_info.speed)
            worker_speeds = np.array(worker_speeds, dtype=np.float32)
            norm = np.sum(worker_speeds)
            if norm < 1e-4:
                return ""
            worker_speeds = worker_speeds / norm
            if True:  # Directly return address
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]
                return worker_name

            # Check status before returning
            while True:
                pt = np.random.choice(np.arange(len(worker_names)),
                    p=worker_speeds)
                worker_name = worker_names[pt]

                if self.get_worker_status(worker_name):
                    break
                else:
                    self.remove_worker(worker_name)
                    worker_speeds[pt] = 0
                    norm = np.sum(worker_speeds)
                    if norm < 1e-4:
                        return ""
                    worker_speeds = worker_speeds / norm
                    continue
            return worker_name
        elif self.dispatch_method == DispatchMethod.SHORTEST_QUEUE:
            worker_names = []
            worker_qlen = []
            for w_name, w_info in self.worker_info.items():
                if model_name in w_info.model_names:
                    worker_names.append(w_name)
                    worker_qlen.append(w_info.queue_length / w_info.speed)
            if len(worker_names) == 0:
                return ""
            min_index = np.argmin(worker_qlen)
            w_name = worker_names[min_index]
            self.worker_info[w_name].queue_length += 1
            logger.info(f"names: {worker_names}, queue_lens: {worker_qlen}, ret: {w_name}")
            return w_name
        else:
            raise ValueError(f"Invalid dispatch method: {self.dispatch_method}")

    def receive_heart_beat(self, worker_name: str, queue_length: int):
        if worker_name not in self.worker_info:
            logger.info(f"Receive unknown heart beat. {worker_name}")
            return False

        self.worker_info[worker_name].queue_length = queue_length
        self.worker_info[worker_name].last_heart_beat = time.time()
        logger.info(f"Receive heart beat. {worker_name}")
        return True

    def remove_stable_workers_by_expiration(self):
        expire = time.time() - CONTROLLER_HEART_BEAT_EXPIRATION
        to_delete = []
        for worker_name, w_info in self.worker_info.items():
            if w_info.check_heart_beat and w_info.last_heart_beat < expire:
                to_delete.append(worker_name)

        for worker_name in to_delete:
            self.remove_worker(worker_name)

    def worker_api_generate_stream(self, params):
        worker_addr = self.get_worker_address(params["model"])
        if not worker_addr:
            logger.info(f"no worker: {params['model']}")
            ret = {
                "text": server_error_msg,
                "error_code": 2,
            }
            yield json.dumps(ret).encode() + b"\0"

        try:
            response = requests.post(worker_addr + "/worker_generate_stream",
                json=params, stream=True, timeout=5)
            for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                if chunk:
                    yield chunk + b"\0"
        except requests.exceptions.RequestException as e:
            logger.info(f"worker timeout: {worker_addr}")
            ret = {
                "text": server_error_msg,
                "error_code": 3,
            }
            yield json.dumps(ret).encode() + b"\0"

    def openai_chat_completions(self, request: ChatCompletionRequest):
        """Forward OpenAI chat completions to worker"""
        if not request.model:
            worker_addr = self.get_worker_address("llava-med-v1.5-mistral-7b")
        else:
            worker_addr = self.get_worker_address(request.model)
        if not worker_addr:
            logger.info(f"no worker for model: {request.model}")
            # Return error response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="No available worker for this model."),
                    finish_reason="error"
                )]
            )

        try:
            # Forward request to worker
            response = requests.post(
                worker_addr + "/v1/chat/completions",
                # json=request.model_dump(),
                json=request.dict(),
                stream=request.stream,
                timeout=60 if not request.stream else None
            )
            # Forward non-streaming response
            logger.info(f"response: {response}")
            logger.info(f"response json: {response.json()}")
            if response.status_code == 200:
                return ChatCompletionResponse(**response.json())
            else:
                # Error handling for non-streaming
                logger.error(f"Worker error: {response.status_code}, {response.text}")
                return ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex}",
                    created=int(time.time()),
                    model=request.model,
                    choices=[ChatCompletionChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content="Worker processing error."),
                        finish_reason="error"
                    )]
                )
                    
        except requests.exceptions.RequestException as e:
            logger.info(f"worker timeout: {worker_addr}, error: {e}")
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="Worker connection error."),
                    finish_reason="error"
                )]
            )
            
    # Let the controller act as a worker to achieve hierarchical
    # management. This can be used to connect isolated sub networks.
    def worker_api_get_status(self):
        model_names = set()
        speed = 0
        queue_length = 0

        for w_name in self.worker_info:
            worker_status = self.get_worker_status(w_name)
            if worker_status is not None:
                model_names.update(worker_status["model_names"])
                speed += worker_status["speed"]
                queue_length += worker_status["queue_length"]

        return {
            "model_names": list(model_names),
            "speed": speed,
            "queue_length": queue_length,
        }


app = FastAPI()


@app.post("/register_worker")
async def register_worker(request: Request):
    data = await request.json()
    controller.register_worker(
        data["worker_name"], data["check_heart_beat"],
        data.get("worker_status", None))


@app.post("/refresh_all_workers")
async def refresh_all_workers():
    models = controller.refresh_all_workers()


@app.post("/list_models")
async def list_models():
    models = controller.list_models()
    return {"models": models}


@app.post("/get_worker_address")
async def get_worker_address(request: Request):
    data = await request.json()
    addr = controller.get_worker_address(data["model"])
    return {"address": addr}


@app.post("/receive_heart_beat")
async def receive_heart_beat(request: Request):
    data = await request.json()
    exist = controller.receive_heart_beat(
        data["worker_name"], data["queue_length"])
    return {"exist": exist}


@app.post("/worker_generate_stream")
async def worker_api_generate_stream(request: Request):
    params = await request.json()
    generator = controller.worker_api_generate_stream(params)
    return StreamingResponse(generator)

def call_chat_completion_sync(request: ChatCompletionRequest) -> ChatCompletionResponse:
    return controller.openai_chat_completions(request)
    
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint with load balancing"""
    logger.info(f"Received chat completion request for model: {request.model}, stream: {request.stream}")
    
    # Non-streaming response
    # response = controller.openai_chat_completions(request)
    # logger.info(f"final response: {response}")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, call_chat_completion_sync, request)
    logger.info(f"final response: {response}")

    return response
        
@app.post("/worker_get_status")
async def worker_api_get_status(request: Request):
    return controller.worker_api_get_status()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21001)
    parser.add_argument("--dispatch-method", type=str, choices=[
        "lottery", "shortest_queue"], default="shortest_queue")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller = Controller(args.dispatch_method)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
