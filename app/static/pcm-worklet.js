// Converts microphone Float32 samples to Int16 PCM and posts ~100 ms chunks.
class PcmWriter extends AudioWorkletProcessor {
  constructor() {
    super();
    this.size = Math.round(sampleRate / 10);
    this.buf = new Int16Array(this.size);
    this.n = 0;
    this.peak = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      const s = Math.max(-1, Math.min(1, ch[i]));
      this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      const a = Math.abs(s);
      if (a > this.peak) this.peak = a;
      if (this.n === this.size) {
        this.port.postMessage({ pcm: this.buf.buffer, level: this.peak }, [this.buf.buffer]);
        this.buf = new Int16Array(this.size);
        this.n = 0;
        this.peak = 0;
      }
    }
    return true;
  }
}
registerProcessor("pcm-writer", PcmWriter);
