#!/usr/bin/env bash
# Build, start, wait for health, and smoke-test the stack.
#   ./scripts/deploy.sh mock            # local test, no GPU (Windows Git Bash / Linux / macOS)
#   ./scripts/deploy.sh gpu             # GPU VM with vLLM + Voxtral
#   ./scripts/deploy.sh gpu sample.wav  # + smoke test with a real speech file
#   ./scripts/deploy.sh down [mock|gpu] # stop the stack
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-mock}"
WAV="${2:-}"
log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }

if [[ "$MODE" == "down" ]]; then
  docker compose -f docker-compose.yml -f "compose.${2:-mock}.yml" down
  exit 0
fi
[[ "$MODE" == "mock" || "$MODE" == "gpu" ]] || { echo "usage: $0 mock|gpu [file.wav] | down [mock|gpu]"; exit 2; }
COMPOSE=(docker compose -f docker-compose.yml -f "compose.${MODE}.yml")

if [[ ! -f .env ]]; then
  cp .env.example .env
  if [[ "$MODE" == "gpu" ]]; then
    key=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
    sed -i "s/^VLLM_API_KEY=.*/VLLM_API_KEY=${key}/" .env
  fi
  log "Created .env from .env.example"
fi
set -a; source .env; set +a

if [[ "$MODE" == "gpu" ]]; then
  log "GPU status (shared VM: check free memory before starting)"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
  total=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.total --format=csv,noheader,nounits)
  free=$(nvidia-smi -i "${GPU_ID:-0}" --query-gpu=memory.free --format=csv,noheader,nounits)
  need=$(awk -v t="$total" -v u="${GPU_MEM_UTIL:-0.45}" 'BEGIN{printf "%d", t*u}')
  log "GPU ${GPU_ID:-0}: vLLM will reserve ~${need} MiB, ${free} MiB free"
  if (( free < need )); then
    echo "Not enough free GPU memory. Lower GPU_MEM_UTIL, pick another GPU_ID, or wait." >&2
    exit 1
  fi
fi

log "Building and starting (${MODE})..."
"${COMPOSE[@]}" up -d --build

URL="http://localhost:${APP_PORT:-8080}/healthz"
log "Waiting for ${URL} (first GPU start downloads the model, can take 10+ min)"
for i in $(seq 1 180); do
  if curl -fsS "$URL" >/dev/null 2>&1; then break; fi
  if (( i % 6 == 0 )); then "${COMPOSE[@]}" logs --tail 3 backend || true; fi
  sleep 5
done
curl -fsS "$URL" || { echo; echo "Stack not healthy, see: ${COMPOSE[*]} logs" >&2; exit 1; }
echo

log "Smoke test (WebSocket end-to-end)"
ARGS=(--url ws://localhost:8080/ws --target fr)
if [[ -n "$WAV" ]]; then
  docker cp "$WAV" "$("${COMPOSE[@]}" ps -q app)":/tmp/sample.wav
  ARGS+=(--wav /tmp/sample.wav)
fi
"${COMPOSE[@]}" exec -T app python tools/smoke_test.py "${ARGS[@]}"

log "OK -> open http://localhost:${APP_PORT:-8080}"
