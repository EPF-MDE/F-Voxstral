"""Mock of the vLLM OpenAI-compatible API, so the full Docker stack can be
tested on a machine without GPU. Same endpoints, fake (deterministic) outputs."""
import asyncio
import io
import os
import time
import wave

from fastapi import FastAPI, File, Form, UploadFile

MODEL = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-3B-2507")
DELAY = float(os.getenv("MOCK_DELAY_S", "0.3"))  # simulated inference time
app = FastAPI(title="vLLM mock")


@app.get("/health")
async def health():
    return {}


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock"}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(file: UploadFile = File(...), model: str = Form(...),
                         language: str | None = Form(None)):
    data = await file.read()
    with wave.open(io.BytesIO(data)) as w:
        dur = w.getnframes() / w.getframerate()
    await asyncio.sleep(DELAY)
    return {"text": f"[mock ASR] {dur:.1f}s of audio received (language={language or 'auto'})"}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    await asyncio.sleep(DELAY)
    prompt = body["messages"][-1]["content"]
    text = prompt.rsplit("Text:", 1)[-1].strip()
    return {
        "id": "mock", "object": "chat.completion", "created": int(time.time()), "model": MODEL,
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": f"[mock translation] {text}"}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
