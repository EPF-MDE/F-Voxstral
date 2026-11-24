# Official vLLM OpenAI server + audio extras required by Voxtral
# (transcription endpoint needs librosa/soundfile, Voxtral needs mistral_common[audio]).
ARG VLLM_VERSION=v0.10.1.1
FROM vllm/vllm-openai:${VLLM_VERSION}
RUN pip install --no-cache-dir "mistral_common[audio]>=1.8.1" librosa soundfile
