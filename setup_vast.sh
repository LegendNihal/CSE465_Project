#!/usr/bin/env bash
# Setup for the Vast.ai box (single RTX 4090, 24 GB, >=60 GB disk).
#
# Recommended Vast template: image  vllm/vllm-openai:v0.10.1
#                            disk   60 GB
#                            ports  expose 8000 if you want to reach it remotely
# If you use that image, vLLM is already installed and you can skip section 1.
#
#   bash setup_vast.sh
set -euo pipefail

WORK=/workspace
MODEL_DIR=$WORK/models/VibeThinker-3B
mkdir -p "$WORK" && cd "$WORK"

echo "=== 1. system packages ==="
apt-get update -qq
# g++ and pypy3 are for step 2 (compiling and running candidate solutions on
# the CPU while the GPU keeps generating). git-lfs is for model download.
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    build-essential g++ pypy3 git git-lfs curl tmux htop
git lfs install --skip-repo

echo "=== 2. python deps ==="
pip install -q --upgrade pip
# Skip vllm here if your image already has it.
python -c "import vllm" 2>/dev/null || pip install -q "vllm==0.10.1"
pip install -q "huggingface_hub[cli]" httpx pyyaml

echo "=== 3. model ==="
if [ ! -d "$MODEL_DIR" ]; then
    hf download WeiboAI/VibeThinker-3B --local-dir "$MODEL_DIR"
fi
du -sh "$MODEL_DIR"

echo "=== 4. sanity ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
g++ --version | head -1
python -c "import vllm; print('vllm', vllm.__version__)"

cat <<'EOF'

Done. Next:
  tmux new -s vllm      # keep the server alive across ssh drops
  bash serve_vllm.sh    # then Ctrl-b d to detach

EOF
