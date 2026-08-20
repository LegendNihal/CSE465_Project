#!/usr/bin/env bash
# Serve VibeThinker-3B on a single RTX 4090.
#
# VRAM budget at --gpu-memory-utilization 0.92 on a 24 GB card:
#   weights (3B bf16)      ~6.2 GB
#   framework + activation ~1.5 GB
#   left for KV cache      ~14  GB
#
# Qwen2.5-3B geometry: 36 layers, 2 KV heads, head_dim 128, bf16
#   -> 2 * 36 * 2 * 128 * 2 bytes = 36 KB per token
#   -> ~380k tokens of KV, i.e. ~9 sequences at the full 40k context, and many
#      more in practice because most traces finish well short of that.
# --max-num-seqs 32 lets vLLM keep the batch full; it preempts if it overcommits.
set -euo pipefail

MODEL=${MODEL:-/workspace/models/VibeThinker-3B}
PORT=${PORT:-8000}
MAXLEN=${MAXLEN:-40960}

exec vllm serve "$MODEL" \
    --served-model-name vibethinker-3b \
    --dtype bfloat16 \
    --max-model-len "$MAXLEN" \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 32 \
    --swap-space 4 \
    --disable-log-requests \
    --host 0.0.0.0 \
    --port "$PORT"
