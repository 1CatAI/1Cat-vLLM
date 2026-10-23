#!/usr/bin/env bash
# Resolve model paths under $MODEL_DIR unless given explicitly, then run the launcher.
set -euo pipefail
: "${VLLM_API_KEY:?Set VLLM_API_KEY (docker run -e VLLM_API_KEY=...)}"
export MODEL_FILE=${MODEL_FILE:-$(ls "$MODEL_DIR"/gguf/*-00001-of-*.gguf 2>/dev/null | head -1)}
export MTP_DIR=${MTP_DIR:-$MODEL_DIR/mtp}
if [[ -z "${VISION_DIR+x}" && -f "$MODEL_DIR/vision/visual.safetensors" ]]; then
  export VISION_DIR=$MODEL_DIR/vision
fi
export EXPERT_PROFILE=${EXPERT_PROFILE:-/app/expert_profile.pt}
export PYTHON_BIN=/opt/venv/bin/python GPU_IDS=${GPU_IDS:-0,1,2,3}
[[ -f "$MODEL_FILE" ]] || { echo "No GGUF found: mount the model at $MODEL_DIR/gguf"; exit 1; }
[[ -f "$MTP_DIR/model.safetensors" ]] || { echo "No MTP weights at $MTP_DIR"; exit 1; }
echo "model=$MODEL_FILE mtp=$MTP_DIR vision=${VISION_DIR:-off} port=$PORT"
exec bash /app/serve_flashnext_gguf_256k.sh
