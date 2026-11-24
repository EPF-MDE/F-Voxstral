"""Real-time voice -> translated text, backed by Voxtral served with vLLM.

Browser streams 16 kHz mono Int16 PCM over a WebSocket. The server cuts the
stream into utterances (energy-based VAD), transcribes each utterance with
Voxtral (/v1/audio/transcriptions) and translates the transcript with the
same model (/v1/chat/completions).
"""
import asyncio
import io
import json
import logging
import os
import time
import wave
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("voxtral-app")

BASE_URL = os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL = os.getenv("MODEL_NAME", "mistralai/Voxtral-Mini-3B-2507")
SR = 16000
VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "400"))
SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "700"))
MIN_SPEECH_MS = int(os.getenv("VAD_MIN_SPEECH_MS", "500"))
MAX_SEGMENT_MS = int(os.getenv("VAD_MAX_SEGMENT_MS", "12000"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE_PER_SESSION", "4"))
# Global cap on in-flight requests to the (shared) GPU, all sessions combined.
GPU_SEM = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")))

LANGS = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "hi": "Hindi",
}

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
app = FastAPI(title="Voxtral live translation")
STATIC = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


# ---------------------------------------------------------------- audio utils
def pcm_to_wav(pcm: bytes, sr: int = SR) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm)
    return buf.getvalue()


def resample(pcm: bytes, sr_in: int, sr_out: int = SR) -> bytes:
    if sr_in == sr_out:
        return pcm
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(round(len(x) * sr_out / sr_in))
    if n_out == 0:
        return b""
    y = np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x)
    return y.astype(np.int16).tobytes()


class Segmenter:
    """Energy-based VAD: emits one PCM chunk per utterance."""

    FRAME = 320  # 20 ms at 16 kHz
    FRAME_MS = 20

    def __init__(self):
        self.pending = b""
        self.preroll = deque(maxlen=15)  # 300 ms kept before speech onset
        self.buf = bytearray()
        self.in_speech = False
        self.silence = 0

    def _dur_ms(self) -> float:
        return len(self.buf) / 2 / SR * 1000

    def feed(self, data: bytes) -> list[bytes]:
        self.pending += data
        out = []
        step = self.FRAME * 2
        while len(self.pending) >= step:
            frame, self.pending = self.pending[:step], self.pending[step:]
            s = np.frombuffer(frame, dtype=np.int16).astype(np.float32)
            voiced = float(np.sqrt(np.mean(s * s))) > VAD_RMS_THRESHOLD
            if not self.in_speech:
                self.preroll.append(frame)
                if voiced:
                    self.in_speech = True
                    self.buf = bytearray(b"".join(self.preroll))
                    self.preroll.clear()
                    self.silence = 0
                continue
            self.buf += frame
            self.silence = 0 if voiced else self.silence + self.FRAME_MS
            dur = self._dur_ms()
            if self.silence >= SILENCE_MS or dur >= MAX_SEGMENT_MS:
                if dur - self.silence >= MIN_SPEECH_MS:
                    out.append(bytes(self.buf))
                self.buf = bytearray()
                self.in_speech = False
                self.silence = 0
        return out

    def flush(self) -> list[bytes]:
        out = []
        if self.in_speech and self._dur_ms() - self.silence >= MIN_SPEECH_MS:
            out.append(bytes(self.buf))
        self.buf, self.in_speech, self.silence = bytearray(), False, 0
        return out


# ------------------------------------------------------------- model calls
async def transcribe(wav: bytes, source: str) -> str:
    kwargs = dict(model=MODEL, file=("segment.wav", wav, "audio/wav"), temperature=0.0)
    if source in LANGS:
        kwargs["language"] = source
    async with GPU_SEM:
        r = await client.audio.transcriptions.create(**kwargs)
    return (r.text or "").strip()


async def translate(text: str, target: str, context: list[str]) -> str:
    prompt = f"Translate the following text into {LANGS.get(target, target)}. "
    prompt += "Reply with the translation only, without any explanation or quotes.\n\n"
    if context:
        prompt += "Previous sentences, for context only (do not translate them): "
        prompt += " ".join(context) + "\n\n"
    prompt += f"Text: {text}"
    async with GPU_SEM:
        r = await client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
    return (r.choices[0].message.content or "").strip()


# ------------------------------------------------------------------ routes
@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/healthz")
async def healthz():
    try:
        models = await client.models.list()
        return {"status": "ok", "backend": BASE_URL, "models": [m.id for m in models.data]}
    except Exception as e:  # backend not ready yet
        return JSONResponse({"status": "backend_unavailable", "error": str(e)}, status_code=503)


async def worker(ws: WebSocket, queue: asyncio.Queue, cfg: dict):
    seg_id = 0
    context: deque[str] = deque(maxlen=2)
    while True:
        pcm = await queue.get()
        seg_id += 1
        t0 = time.perf_counter()
        try:
            text = await transcribe(pcm_to_wav(pcm), cfg["source"])
            if not text:
                continue
            t1 = time.perf_counter()
            await ws.send_json({"type": "segment", "id": seg_id, "transcript": text,
                                "audio_s": round(len(pcm) / 2 / SR, 2),
                                "asr_ms": int((t1 - t0) * 1000)})
            tr = await translate(text, cfg["target"], list(context))
            context.append(text)
            await ws.send_json({"type": "translation", "id": seg_id, "translation": tr,
                                "latency_ms": int((time.perf_counter() - t0) * 1000)})
        except Exception as e:
            log.exception("segment %s failed", seg_id)
            try:
                await ws.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
            except Exception:
                return


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    cfg = {"source": "auto", "target": "fr", "sample_rate": SR}
    seg = Segmenter()
    queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
    task = asyncio.create_task(worker(ws, queue, cfg))

    async def enqueue(chunks):
        for c in chunks:
            if queue.full():  # GPU too slow: drop oldest to stay real-time
                queue.get_nowait()
                await ws.send_json({"type": "warning", "message": "GPU busy, segment dropped"})
            queue.put_nowait(c)

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await enqueue(seg.feed(resample(msg["bytes"], cfg["sample_rate"])))
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "config":
                    for k in ("source", "target", "sample_rate"):
                        if k in data:
                            cfg[k] = data[k]
                    cfg["sample_rate"] = int(cfg["sample_rate"])
                    await ws.send_json({"type": "ready", "config": cfg})
                elif data.get("type") == "flush":
                    await enqueue(seg.flush())
    except WebSocketDisconnect:
        pass
    finally:
        # let pending segments finish briefly, then stop
        for _ in range(600):
            if queue.empty():
                break
            await asyncio.sleep(0.1)
        task.cancel()
