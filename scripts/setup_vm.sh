#!/usr/bin/env bash
# One-time setup of an Ubuntu 22.04/24.04 GPU VM: Docker + NVIDIA Container Toolkit.
# Idempotent: safe to re-run. Requires sudo and an already-installed NVIDIA driver.
set -euo pipefail

log() { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }

log "1/4 Checking NVIDIA driver"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: install the NVIDIA driver first (e.g. sudo ubuntu-drivers install)" >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv

log "2/4 Installing Docker (if missing)"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
fi
sudo usermod -aG docker "$USER" || true

log "3/4 Installing NVIDIA Container Toolkit (if missing)"
if ! command -v nvidia-ctk >/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y nvidia-container-toolkit
fi
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

log "4/4 Testing GPU access from a container"
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L

log "Done. Log out/in (or run 'newgrp docker') so docker works without sudo."
