"""
A model worker executes the model.
"""
import argparse
import asyncio
import json
import time
import threading
import uuid
import base64
from typing import List, Dict, Any, Optional, Union

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import torch
import uvicorn
from functools import partial

from llava.constants import WORKER_HEART_BEAT_INTERVAL
from llava.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, load_image_from_base64, tokenizer_image_token, KeywordsStoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from transformers import TextIteratorStreamer
from threading import Thread



GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None

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
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]

def convert_openai_to_llava_format(messages: List[ChatMessage]) -> tuple:
    """Convert OpenAI format messages to LLaVA format"""
    prompt_parts = []
    images = []
    
    for message in messages:
        if message.role == "user":
            if isinstance(message.content, str):
                prompt_parts.append(message.content)
            elif isinstance(message.content, list):
                for content_item in message.content:
                    if content_item["type"] == "text":
                        prompt_parts.append(content_item["text"])
                    elif content_item["type"] == "image_url":
                        # Add image token to prompt
                        prompt_parts.append(DEFAULT_IMAGE_TOKEN)
                        # Handle image URL - convert to base64 if needed
                        image_url = content_item["image_url"]["url"]

                        if image_url.startswith("http"):
                            logger.info("getting http image...")
                            # For HTTP URLs, you might want to fetch and convert to base64
                            # For now, we'll assume it's a local file path
                            response = requests.get(image_url)
                            image_data = response.content
                            encoded_image = base64.b64encode(image_data).decode('utf-8')
                            images.append(encoded_image)
                            logger.info("image loaded")
                        else:
                            logger.info("getting base64 image...")
                            # Extract base64 data from data URL
                            base64_data = image_url
                            images.append(base64_data)
                            logger.info("image loaded")

        elif message.role == "assistant":
            prompt_parts.append(message.content)
    logger.info("start ot concat prompt")
    prompt = " ".join(prompt_parts)
    prompt = f"[INST] {prompt} [/INST]"
    return prompt, images

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()
  
class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_base, model_name,
                 load_8bit, load_4bit, device):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        self.device = device
        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, model_base, self.model_name, load_8bit, load_4bit, device=self.device)
        self.is_multimodal = 'llava' in self.model_name.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        # logger.info(f"all params: {params}")
        for k,v in params.items():
            if k != "images":
                logger.info(f"param k: {v}")
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        num_image_tokens = 0
        if images is not None and len(images) > 0 and self.is_multimodal:
            if len(images) > 0:
                if len(images) != prompt.count(DEFAULT_IMAGE_TOKEN):
                    raise ValueError("Number of images does not match number of <image> tokens in prompt")

                images = [load_image_from_base64(image) for image in images]
                images = process_images(images, image_processor, model.config)

                if type(images) is list:
                    images = [image.to(self.model.device, dtype=torch.float16) for image in images]
                else:
                    images = images.to(self.model.device, dtype=torch.float16)

                replace_token = DEFAULT_IMAGE_TOKEN
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

                num_image_tokens = prompt.count(replace_token) * model.get_vision_tower().num_patches
            else:
                images = None
            image_args = {"images": images}
        else:
            images = None
            image_args = {}
            
        logger.info("image process done")
        
        temperature = float(params.get("temperature", 1.0))
        top_p = float(params.get("top_p", 1.0))
        max_context_length = getattr(model.config, 'max_position_embeddings', 2048)
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        if stop_str is None:
            stop_str = "</s>"
        do_sample = True if temperature > 0.001 else False
        
        logger.info("basic setting done")
        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.device)
        
        logger.info("input ids done")
        
        keywords = [stop_str]

        logger.info(f"keywords done: {keywords}")
        
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        logger.info("stopping_criteria done")
        
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=15)
        
        logger.info("streamer setting done")
        
        max_new_tokens = min(max_new_tokens, max_context_length - input_ids.shape[-1] - num_image_tokens)
        
        if max_new_tokens < 1:
            yield json.dumps({"text": ori_prompt + "Exceeds max token length. Please start a new conversation, thanks.", "error_code": 0}).encode() + b"\0"
            return
        logger.info("max new tokens done")
        thread = Thread(target=model.generate, kwargs=dict(
            inputs=input_ids,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
            **image_args
        ))
        logger.info("thread setting done")
        thread.start()

        generated_text = ori_prompt
        for new_text in streamer:
            generated_text += new_text
            if generated_text.endswith(stop_str):
                generated_text = generated_text[:-len(stop_str)]
            yield json.dumps({"text": generated_text, "error_code": 0}).encode() + b"\0"

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except Exception as e:
            print("Caught Unknown Error", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"

    def generate_openai_stream(self, request: ChatCompletionRequest):
        """Generate OpenAI compatible streaming response"""
        try:
            # Convert OpenAI format to LLaVA format
            prompt, images = convert_openai_to_llava_format(request.messages)
            
            params = {
                "prompt": prompt,
                "images": images,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens,
                "stop": request.stop if isinstance(request.stop, str) else (request.stop[0] if request.stop else None)
            }
            
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            created = int(time.time())
            
            # First chunk with role
            first_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={"role": "assistant"},
                    finish_reason=None
                )]
            )
            # yield f"data: {first_chunk.model_dump_json()}\n\n"
            yield f"data: {first_chunk.json()}\n\n"
            
            # Generate content
            accumulated_content = ""
            for chunk in self.generate_stream(params):
                try:
                    data = json.loads(chunk.decode().rstrip('\0'))
                    if data.get("error_code", 0) == 0:
                        full_text = data["text"].split("[/INST]")[-1]
                        # Extract only the new content (response part)
                        if full_text.startswith(prompt):
                            new_content = full_text[len(prompt):].strip()
                        else:
                            new_content = full_text.strip()
                        
                        # Send incremental content
                        if new_content != accumulated_content:
                            delta_content = new_content[len(accumulated_content):]
                            if delta_content:
                                stream_chunk = ChatCompletionStreamResponse(
                                    id=completion_id,
                                    created=created,
                                    model=request.model,
                                    choices=[ChatCompletionStreamChoice(
                                        index=0,
                                        delta={"content": delta_content},
                                        finish_reason=None
                                    )]
                                )
                                # yield f"data: {stream_chunk.model_dump_json()}\n\n"
                                yield f"data: {stream_chunk.json()}\n\n"
                                accumulated_content = new_content
                    else:
                        # Error case
                        error_chunk = ChatCompletionStreamResponse(
                            id=completion_id,
                            created=created,
                            model=request.model,
                            choices=[ChatCompletionStreamChoice(
                                index=0,
                                delta={},
                                finish_reason="error"
                            )]
                        )
                        # yield f"data: {error_chunk.model_dump_json()}\n\n"
                        yield f"data: {error_chunk.json()}\n\n"
                        break
                except Exception as e:
                    logger.error(f"Error processing stream chunk: {e}")
                    continue
            
            # Final chunk
            final_chunk = ChatCompletionStreamResponse(
                id=completion_id,
                created=created,
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="stop"
                )]
            )
            # yield f"data: {final_chunk.model_dump_json()}\n\n"
            yield f"data: {final_chunk.json()}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Error in generate_openai_stream: {e}")
            error_chunk = ChatCompletionStreamResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionStreamChoice(
                    index=0,
                    delta={},
                    finish_reason="error"
                )]
            )
            # yield f"data: {error_chunk.model_dump_json()}\n\n"
            yield f"data: {error_chunk.json()}\n\n"

    def generate_openai_completion(self, request: ChatCompletionRequest):
        """Generate OpenAI compatible non-streaming response"""
        logger.info("generating non-stream response...")
        try:
            # Convert OpenAI format to LLaVA format
            prompt, images = convert_openai_to_llava_format(request.messages)
            logger.info(f"prompt is: {prompt}")
            logger.info(f"num of images are: {len(images)}")
            params = {
                "prompt": prompt,
                "images": images,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "max_new_tokens": request.max_tokens,
                "stop": request.stop if isinstance(request.stop, str) else (request.stop[0] if request.stop else None)
            }
            
            # Collect all streaming output
            full_response = ""
            logger.info("Start to generate_stream")
            for chunk in self.generate_stream(params):
                try:
                    logger.info(f"loaded chunk: {chunk}")
                    data = json.loads(chunk.decode("utf-8").rstrip('\0'))
                    
                    logger.info(f"loaded data: {data}")
                    if data.get("error_code", 0) == 0:
                        full_text = data["text"].split("[/INST]")[-1]
                        
                        logger.info(f"full text: {full_text}")
                        
                        # Extract only the new content (response part)
                        if full_text.startswith(prompt):
                            response_content = full_text[len(prompt):].strip()
                        else:
                            response_content = full_text.strip()
                        full_response = response_content
                    else:
                        raise Exception("Generation error")
                except Exception as e:
                    logger.error(f"Error processing completion chunk: {e}")
                    continue
            
            # Create OpenAI compatible response
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
            response = ChatCompletionResponse(
                id=completion_id,
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=full_response),
                    finish_reason="stop"
                )]
            )
            logger.info(f"before return: {response.choices[0].message.content}")
            
            return response
            
        except Exception as e:
            logger.error(f"Error in generate_openai_completion: {e}")
            # Return error response
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content="I apologize, but I encountered an error while processing your request."),
                    finish_reason="error"
                )]
            )

app = FastAPI()

def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI compatible chat completions endpoint"""
    logger.info("receive chat completion reuquest")
    global model_semaphore, global_counter
    global_counter += 1

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    
    try:
        if request.stream:
            # Streaming response
            generator = worker.generate_openai_stream(request)
            background_tasks = BackgroundTasks()
            background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
            return StreamingResponse(generator, media_type="text/plain", background=background_tasks)
        else:
            logger.info("start to generate non-streaming response")
            # Non-streaming response
            response = worker.generate_openai_completion(request)
            background_tasks = BackgroundTasks()
            background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
            # Execute background tasks manually for non-streaming
            release_model_semaphore(fn=worker.send_heart_beat)
            return response
    except Exception as e:
        logger.error(f"Error in chat_completions: {e}")
        release_model_semaphore(fn=worker.send_heart_beat)
        raise


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=1)
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_base,
                         args.model_name,
                         args.load_8bit,
                         args.load_4bit,
                         args.device)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
