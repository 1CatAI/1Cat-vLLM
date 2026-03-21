#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 0 && "${1}" != -* ]]; then
    exec "$@"
fi

default_compilation='{"cudagraph_mode":"full_and_piecewise","cudagraph_capture_sizes":[1]}'
default_mm_limits='{"image":0,"video":0}'

args=(python -m vllm.entrypoints.openai.api_server)

append_arg() {
    local flag="$1"
    local value="${2:-}"
    if [[ -n "${value}" ]]; then
        args+=("${flag}" "${value}")
    fi
}

append_arg --host "${VLLM_HOST:-0.0.0.0}"
append_arg --port "${VLLM_PORT:-8000}"
append_arg --model "${VLLM_MODEL:-}"
append_arg --served-model-name "${VLLM_SERVED_MODEL_NAME:-}"
append_arg --quantization "${VLLM_QUANTIZATION:-awq}"
append_arg --dtype "${VLLM_DTYPE:-float16}"
append_arg --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-}"
append_arg --max-model-len "${VLLM_MAX_MODEL_LEN:-}"
append_arg --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-}"
append_arg --max-num-seqs "${VLLM_MAX_NUM_SEQS:-}"
append_arg --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
append_arg --download-dir "${VLLM_DOWNLOAD_DIR:-}"

args+=(
    --attention-backend "${VLLM_ATTENTION_BACKEND:-TRITON_ATTN}"
    --compilation-config "${VLLM_COMPILATION_CONFIG:-${default_compilation}}"
    --limit-mm-per-prompt "${VLLM_LIMIT_MM_PER_PROMPT:-${default_mm_limits}}"
)

if [[ "${VLLM_SKIP_MM_PROFILING:-1}" != "0" ]]; then
    args+=(--skip-mm-profiling)
fi

args+=("$@")

exec "${args[@]}"
