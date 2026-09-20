#!/usr/bin/env bash
# Incremental rebuild: warm build/ cache in this tree, so only the patched
# topk translation units recompile + relink. (No FA step.)
set -euo pipefail

TREE=/home/nvidia/Dev/references/1Cat-vLLM
VENV=/home/nvidia/miniconda3/envs/1cat-vllm-sm70
OUT=$TREE/dist-cu128-sm70-qwen4exp

export PATH=$VENV/bin:/usr/local/cuda-12.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.8
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST="7.0"
export MAX_JOBS=12
export NVCC_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=12

cd "$TREE"
echo "### VLLM-INC $(date) tree=$(git -C "$TREE" rev-parse --short HEAD) +topk256k-patch"
python -m build --wheel --no-isolation --outdir "$OUT"
echo "### VLLM WHEEL: $(ls -la $OUT/1cat_vllm-*.whl 2>/dev/null | tail -1)"
echo "### cleaning stale site-packages/vllm + dist-info"
rm -rf "$VENV/lib/python3.12/site-packages/vllm" "$VENV/lib/python3.12/site-packages/"vllm-*.dist-info
# newest wheel only: this dir keeps historical wheels, and passing two
# wheels for the same project makes pip's resolver give up.
WHL="$(ls -t "$OUT"/1cat_vllm-*.whl | head -1)"
echo "### installing $WHL"
pip install --force-reinstall --no-deps "$WHL"
echo "### INSTALLED into $VENV"
echo "### BUILD DONE $(date)"