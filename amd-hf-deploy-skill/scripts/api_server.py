"""
AMD GPU (ROCm) + Transformers + FastAPI 模型 API 服务
提供 OpenAI 兼容的 /v1/chat/completions 接口，支持流式和非流式输出。
支持 <think> 思维链模型，自动分离 reasoning_content 和 content。

使用方法：
  1. 修改下方 MODEL_PATH 和 MODEL_NAME
  2. 如果是纯文本模型，将 AutoModelForImageTextToText 改为 AutoModelForCausalLM
  3. python3 api_server.py
"""

import torch
import time
import uuid
import json
import re
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from transformers import AutoModelForImageTextToText, AutoTokenizer, AutoProcessor, TextIteratorStreamer
from threading import Thread

# ╔══════════════════════════════════════════╗
# ║           只需修改这两个变量              ║
# ╚══════════════════════════════════════════╝
MODEL_PATH = "/path/to/your/model"       # 模型本地路径
MODEL_NAME = "your-model-name"            # API 中显示的模型名
PORT = 8001                               # 服务端口


# ──────────────────────────────────────────
# 初始化
# ──────────────────────────────────────────
app = FastAPI()

print("Loading tokenizer...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer = (
    processor.tokenizer
    if hasattr(processor, "tokenizer")
    else AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully!")


# ──────────────────────────────────────────
# 思维链分离
# ──────────────────────────────────────────
def split_thinking(text: str):
    """分离模型输出中的思维链和正式回答。

    处理三种情况：
    1. <think>reasoning</think>content  — 两个标签都在
    2. reasoning</think>content         — <think> 被 skip_special_tokens 去掉
    3. 无标签                            — 全部是 content
    """
    # Case 1: 两个标签都在
    if "<think>" in text and "</think>" in text:
        pattern = r"^(?:\s*<think>(.*?)</think>\s*)?(.*)$"
        m = re.match(pattern, text, re.DOTALL)
        if m:
            return (m.group(1) or "").strip(), (m.group(2) or "").strip()

    # Case 2: 只有 </think>（<think> 被 tokenizer 跳过了）
    if "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")

    # Case 3: 没有标签
    return "", text.strip()


# ──────────────────────────────────────────
# API 端点
# ──────────────────────────────────────────
@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 4096)
    top_p = body.get("top_p", 0.9)
    stream = body.get("stream", False)

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature if temperature > 0 else 1.0,
        "top_p": top_p,
        "do_sample": temperature > 0,
    }

    # ── 流式输出 ──
    if stream:
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs["streamer"] = streamer
        thread = Thread(target=model.generate, kwargs={**inputs, **gen_kwargs})
        thread.start()

        async def generate():
            chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            buffer = ""
            in_think = True
            think_closed = False

            for text_chunk in streamer:
                if not text_chunk:
                    continue
                buffer += text_chunk

                # 还在思维链阶段，等 </think> 出现
                if in_think and not think_closed:
                    if "</think>" in buffer:
                        parts = buffer.split("</think>", 1)
                        reasoning_part = parts[0].replace("<think>", "")
                        remaining = parts[1].lstrip("\n") if len(parts) > 1 else ""
                        think_closed = True
                        in_think = False

                        if reasoning_part.strip():
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": MODEL_NAME,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "reasoning_content": reasoning_part.strip()
                                        },
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"

                        if remaining:
                            chunk = {
                                "id": chat_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": MODEL_NAME,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": remaining},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        buffer = ""
                        continue
                    else:
                        continue

                # 思维链结束后，逐块流式输出 content
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": MODEL_NAME,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

            # 如果始终没看到 </think>，整个 buffer 做一次分离
            if in_think and buffer:
                reasoning, content = split_thinking(buffer)
                if reasoning:
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"reasoning_content": reasoning},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                if content:
                    chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": MODEL_NAME,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"

            # 结束标记
            final = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    # ── 非流式输出 ──
    else:
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        reasoning_content, content = split_thinking(response_text)

        message = {"role": "assistant", "content": content}
        if reasoning_content:
            message["reasoning_content"] = reasoning_content

        return JSONResponse(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": MODEL_NAME,
                "choices": [
                    {"index": 0, "message": message, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": len(new_tokens),
                    "total_tokens": inputs["input_ids"].shape[1] + len(new_tokens),
                },
            }
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


# ──────────────────────────────────────────
# 启动
# ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
