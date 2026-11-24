"""End-to-end smoke test: streams a WAV file (or a synthetic signal) to the
app WebSocket at real-time speed and checks that translations come back.

    python smoke_test.py --url ws://localhost:8080/ws --wav sample.wav --target fr
Exit code 0 = at least one translation received.
"""
import argparse
import asyncio
import json
import sys
import time
import wave

import numpy as np
import websockets

SR = 16000


def load_wav(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        sys.exit("WAV must be 16-bit PCM (convert: ffmpeg -i in.mp3 -ac 1 -ar 16000 out.wav)")
    x = np.frombuffer(raw, dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return x.tobytes(), sr


def synthetic() -> tuple[bytes, int]:
    """3 bursts of 'voice-like' noise separated by silence (only useful with the mock)."""
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(3):
        t = np.arange(int(1.5 * SR)) / SR
        burst = 4000 * np.sin(2 * np.pi * 220 * t) + 1500 * rng.standard_normal(t.size)
        parts += [burst, np.zeros(SR)]
    return np.concatenate(parts).astype(np.int16).tobytes(), SR


async def run(url, pcm, sr, source, target, speed, idle_timeout):
    results = {}
    async with websockets.connect(url, max_size=None) as ws:
        await ws.send(json.dumps({"type": "config", "source": source,
                                  "target": target, "sample_rate": sr}))
        chunk = int(sr * 0.1) * 2  # 100 ms of int16

        async def sender():
            for i in range(0, len(pcm), chunk):
                await ws.send(pcm[i:i + chunk])
                await asyncio.sleep(0.1 / speed)
            await ws.send(b"\x00\x00" * sr)  # 1 s trailing silence
            await ws.send(json.dumps({"type": "flush"}))

        send_task = asyncio.create_task(sender())
        t0 = time.time()
        while True:
            try:
                msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=idle_timeout))
            except asyncio.TimeoutError:
                if send_task.done():
                    break
                continue
            print(f"[{time.time() - t0:6.1f}s] {msg}")
            if msg["type"] == "translation":
                results[msg["id"]] = msg
            if msg["type"] == "error":
                print("ERROR from server:", msg["message"], file=sys.stderr)
        await send_task
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="ws://localhost:8080/ws")
    p.add_argument("--wav", help="16-bit PCM WAV; omitted = synthetic signal")
    p.add_argument("--source", default="auto")
    p.add_argument("--target", default="fr")
    p.add_argument("--speed", type=float, default=1.0, help="1.0 = real time")
    p.add_argument("--idle-timeout", type=float, default=30)
    a = p.parse_args()
    pcm, sr = load_wav(a.wav) if a.wav else synthetic()
    res = asyncio.run(run(a.url, pcm, sr, a.source, a.target, a.speed, a.idle_timeout))
    if not res:
        print("FAIL: no translation received", file=sys.stderr)
        sys.exit(1)
    lat = [r["latency_ms"] for r in res.values()]
    print(f"OK: {len(res)} segment(s) translated, latency avg {sum(lat)/len(lat):.0f} ms, max {max(lat)} ms")


if __name__ == "__main__":
    main()
