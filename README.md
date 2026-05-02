# vllm-v100

vLLM fork for Tesla V100 (SM70) extending [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)'s AWQ support with compressed-tensors, MoE, and improved kernel accuracy.

## What this fork adds

1CatAI's fork provides AWQ 4-bit inference on V100 via hand-tuned TurboMind SM70 CUDA kernels. This fork extends that foundation with:

- **Compressed-tensors W4A16 on V100** -- lowers `min_capability` from 75 to 70 (from [vLLM PR #32597](https://github.com/vllm-project/vllm/pull/32597))
- **TurboMindLinearKernel** -- uses 1Cat's `awq_gemm_sm70` for dense linear layers instead of the Triton GPTQ kernel, which has ~2% mean relative error per matmul on V100 (compounds to garbage across deep networks). TurboMind achieves <0.1% error.
- **MoE compressed-tensors fix** -- `CompressedTensorsSM70WNA16MoEMethod` was missing ~20 layer attributes needed by the AWQ apply path. Fixed by delegating to `AWQSM70MoEMethod` after CT-to-AWQ weight conversion.
- **`_DEFAULT_MAX_TOKENS` naming fix** -- alias for renamed constant that broke the CT MoE import chain

## Verified models

| Model | Params | Quant | Architecture | TP | Status |
|-------|--------|-------|-------------|---:|--------|
| [cyankiwi/MiniMax-M2.7-AWQ-4bit](https://huggingface.co/cyankiwi/MiniMax-M2.7-AWQ-4bit) | 240B (11B active) | compressed-tensors W4A16 | MoE (256 experts) | 8 | Working |
| [tclf90/Qwen3.5-27B-AWQ](https://huggingface.co/tclf90/Qwen3.5-27B-AWQ) | 27B | AWQ | Dense | 2-4 | Working (1Cat-validated) |
| [tclf90/Qwen3.5-35B-A3B-AWQ](https://huggingface.co/tclf90/Qwen3.5-35B-A3B-AWQ) | 35B (3B active) | AWQ | MoE | 2-4 | Working (1Cat-validated) |
| [tclf90/Qwen3.5-122B-A10B-AWQ](https://huggingface.co/tclf90/Qwen3.5-122B-A10B-AWQ) | 122B (10B active) | AWQ | MoE | 4+ | Working (1Cat-validated) |
| [cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) | 27B | compressed-tensors W4A16 (asymmetric) | Hybrid Gated DeltaNet | 4 | Working (greedy + tool-calling smoke) |

## Hardware tested

- 8x Tesla V100 SXM2 32GB (TP=8, no expert parallel)

## Known issues

- **Expert parallel corrupts MoE output** for MiniMax M2.7 on this fork. Use tensor parallelism without `--enable-expert-parallel`. Root cause is likely in the EP code path for 256-expert models.
- **V100 Triton JIT compilation takes 30-90 minutes** on first request. Set `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` to avoid pod kills.
- **Do NOT use `--quantization gptq_marlin`** or `CUDA_LAUNCH_BLOCKING=1` on V100.

## Docker build

Build the image from the included Dockerfile:

```bash
docker build -f docker/Dockerfile.sm70-wheel -t vllm-v100:latest .
```

The Docker images install three artifacts: PyTorch (cu128), 1Cat's vLLM
wheel (v0.0.2 with our overlaid Python patches), and 1Cat's
`flash_attn_v100` wheel (v0.0.3, cp312/cu128). The FA-V100 wheel unlocks
`--attention-backend FLASH_ATTN_V100` (the SM70 FlashAttention-2 path).
Without it the registered backend silently falls back to Triton.

### Building flash_attn_v100 from source

If you need a different Python or CUDA combo than the published wheel
(`cp312-cp312-linux_x86_64`, cu128), build the extension from the
vendored source under `flash-attention-v100/`:

```bash
# Requires nvcc on PATH and the same torch already in the venv.
PATH=/usr/local/cuda-12.8/bin:$PATH \
  TORCH_CUDA_ARCH_LIST="7.0" \
  pip install -e flash-attention-v100/ --no-build-isolation
```

`--no-build-isolation` is important: it ensures the build picks up the
torch you already have installed instead of pulling a different version.

### Quick run (MiniMax M2.7 on 8x V100 32GB)

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/cyankiwi/MiniMax-M2.7-AWQ-4bit \
  -e VLLM_SERVED_MODEL_NAME=MiniMax-M2.7 \
  -e VLLM_QUANTIZATION=compressed-tensors \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=8 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=32768 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest
```

### Quick run (Qwen3.6-27B-AWQ-INT4 on 4x V100 32GB)

Hybrid Gated DeltaNet, asymmetric compressed-tensors W4A16. Requires the
new `TurboMindAsymLinearKernel` for dense Linear (already in this fork).
`--enforce-eager` is required: V100 hits an upstream `causal_conv1d`
CUDA-graph capture assertion (vllm-project/vllm#35945) on small batches.
Tool-calling is enabled with the `qwen3_coder` parser.

```bash
docker run --rm --gpus '"device=0,1,2,3"' --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/cyankiwi/Qwen3.6-27B-AWQ-INT4 \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.6-27B-AWQ-INT4 \
  -e VLLM_QUANTIZATION=compressed-tensors \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=4 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  -e VLLM_MAX_MODEL_LEN=32768 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  -e VLLM_COMPILATION_CONFIG='{"cudagraph_mode":"NONE"}' \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Quick run (Qwen3.5-27B-AWQ on 2x V100)

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Qwen3.5-27B-AWQ \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.5-27B-AWQ \
  -e VLLM_QUANTIZATION=awq \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=262144 \
  -e VLLM_MAX_NUM_SEQS=4 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --attention-backend TRITON_ATTN \
  --skip-mm-profiling \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

### Test the API

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-M2.7",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }'
```

## Technical details

### Kernel selection (dense linear layers)

For compressed-tensors W4A16 on V100, the kernel selection order is:

1. **TurboMindLinearKernel** (preferred) -- converts CT pack-quantized weights to AWQ format, uses `awq_gemm_sm70` (<0.1% error)
2. **TritonLinearKernel** (fallback) -- Triton GPTQ kernel from PR #32597 (~2% error, unsuitable for deep networks)
3. **ExllamaLinearKernel** (existing) -- standard Exllama path

### CT-to-AWQ weight conversion

The TurboMindLinearKernel handles weight format conversion at load time:
1. `permute_param_layout_` to get CT `[K/8, N]` with sequential packing
2. Unpack CT nibbles to `[K, N]`
3. Repack as AWQ `[K, N/8]` with interleaved order
4. Generate symmetric qzeros (`0x88888888`)
5. `awq_sm70_prepare` for TurboMind format

### MoE path

For MoE models using compressed-tensors quantization, `CompressedTensorsSM70WNA16MoEMethod` converts weights from CT to AWQ format, then delegates to `AWQSM70MoEMethod` for TurboMind setup (alignment, strided ptrs, buffer allocation).

## Validated stack

- GPU: Tesla V100 SXM2 32GB
- CUDA: 12.8
- Python: 3.12
- PyTorch: 2.9.1+cu128
- Driver: 570.x

## Acknowledgements

- [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) -- TurboMind SM70 AWQ CUDA kernels and base V100 support
- [vLLM](https://github.com/vllm-project/vllm) -- upstream inference engine
- [lmdeploy / TurboMind](https://github.com/InternLM/lmdeploy) -- original SM70 WMMA kernels

## License

Apache 2.0 -- same as upstream vLLM. See [LICENSE](LICENSE).
