# vllm-v100

vLLM fork for Tesla V100 (SM70) extending [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)'s AWQ support with compressed-tensors, MoE, and improved kernel accuracy.

## What this fork adds

1CatAI's fork provides AWQ 4-bit inference on V100 via hand-tuned TurboMind SM70 CUDA kernels. This fork extends that foundation with:

- **Compressed-tensors W4A16 on V100** -- lowers `min_capability` from 75 to 70 (from [vLLM PR #32597](https://github.com/vllm-project/vllm/pull/32597))
- **TurboMindLinearKernel** -- uses 1Cat's `awq_gemm_sm70` for dense linear layers instead of the Triton GPTQ kernel, which has ~2% mean relative error per matmul on V100 (compounds to garbage across deep networks). TurboMind achieves <0.1% error.
- **MoE compressed-tensors fix** -- `CompressedTensorsSM70WNA16MoEMethod` was missing ~20 layer attributes needed by the AWQ apply path. Fixed by delegating to `AWQSM70MoEMethod` after CT-to-AWQ weight conversion.
- **`_DEFAULT_MAX_TOKENS` naming fix** -- alias for renamed constant that broke the CT MoE import chain
- **DeepSeek-V4-Flash on V100** -- runnable model class for Intel's W4A16 AutoRound quant of V4-Flash (290B / ~37B active, 256 experts, MLA + sparse attention + Hyper-Connections). Includes a V100 fp16 sparse-attention kernel port, a `_hc_post` clamp that prevents fp16 residual overflow at pos 0, an Obstacle-1 CPU-mirror `start_pos` in attention metadata that drops the per-forward host sync, and a paged main-window KV cache (single-request scope; multi-request via paged compressor/indexer caches is the natural Stage-2 follow-up).
- **Mistral-Small-4 119B GGUF on V100** -- runnable model class for Bartowski's Q4_K_M GGUF of Mistral-Small-4-119B-2603 (MoE, MLA, fused `ffn_gate_up_exps` + split `attn_k_b`/`attn_v_b` tensor layouts). Ships with three latent fixes that affect any GGUF or MLA user on V100: a 4-site fp16 overflow clamp in the GGUF csrc kernels (kept kernels' internal fp32 accumulator, clamped to ±65504 at the implicit fp32→fp16 write-back), an MMQ kernel alignment dispatch in `gguf.py` (small dense models like Qwen2.5-0.5B with hidden=896 now correctly fall back to dequantize instead of reading past the qweight buffer), and a manual fp32 LSE-returning fallback in `mla_attention.py` so MLA models with prefix caching / chunked prefill no longer crash `merge_attn_states` on V100.
- **MiMo-V2.5 310B GGUF on V100** -- runnable model class for Bartowski's Q3_K_M GGUF of XiaomiMiMo/MiMo-V2.5 (310B / 15B active, hybrid SWA + full attention with asymmetric head dims Q/K=192 / V=128, fused `attn_qkv` + 3D `ffn_*_exps` tensor layouts, MTP blocks that we skip). Ships with two additional fixes that affect any V100 user, not just MiMo: an HDIM=192 template instantiation in the `flash_attn_v100` kernels (was only 64/80/96/112/128/256 before -- any model with head_dim=192 hit the default-case TORCH_CHECK), and an MMQ alignment guard mirrored from the dense path into `_fused_moe_gguf` (MoE models with K-quant experts whose per-rank `w2-input` isn't aligned silently IMA-crashed once batch crossed the MMVQ→MMQ threshold). Also pins `triton==3.5.1` in `requirements/cuda.txt` to match torch 2.9.1+cu128's wheel metadata, since triton 3.6.0's MLA decode codegen is ~3× slower on V100 sm_70 at long context (verified on Mistral4 T2 stress: 23.5 → 9.4 tok/s with 3.6.0, restored with the pin).
- **Qwen3.6-35B-A3B GGUF on V100** -- runnable model class for Bartowski's Q8_0 GGUF of Qwen3.6-35B-A3B (35B / 3B active, 256-expert MoE, hybrid Gated-DeltaNet + full-attention every 4th layer, interleaved M-RoPE; text backbone `Qwen3_5MoeForCausalLM`). transformers/vLLM have no GGUF support for arch `qwen35moe`, so the loader binds the text backbone via `--hf-config-path` + `--hf-overrides` (strip `vision_config`). Three GGUF-interpretation fixes were needed, each affecting any GGUF user of this arch: (a) **Gemma-style RMSNorm double-`+1`** -- Qwen3.5/3.6 use `y=(1+w)*x` and llama.cpp bakes the `+1` into the GGUF norm weights, so vLLM re-adding it doubled every norm (the loader subtracts 1 at load, excluding the gated `linear_attn.norm`); (b) **Gated-DeltaNet `A_log` double-exponentiation** -- GGUF `ssm_a` already stores the decay `A=-exp(A_log)`, so the loader stores `log(-ssm_a)` to keep `-exp(A_log)==ssm_a` instead of re-applying `-exp()` (which collapsed the recurrence in all 30 GDN layers); (c) **GDN value-head TILE vs repeat_interleave order** -- llama.cpp pairs value-head `i` with key-head `i % num_k`, vLLM's FLA kernel uses `i // r`, so the loader permutes the value heads pre-shard (head boundaries align to Q8_0 blocks, so the packed-byte permute stays byte-clean and TP-safe). A follow-on loads the model's native MTP (`nextn`) head from GGUF block 40 as a speculative-decode draft (~60% acceptance) -- it's loadable but **not recommended on V100**: spec-decode is net-negative single-stream here because the fast `flash_attn_v100` backend can't keep CUDA graphs under spec-decode (forced to PIECEWISE -> ~46 tok/s) and the `triton_attn` backend that can keep them still loses to no-spec (~77 vs ~100 tok/s) once the draft + 2-token verify overhead is counted. The same draft-loader path would pay off on Ampere/Hopper.

## Verified models

| Model | Params | Quant | Architecture | TP | Status |
|-------|--------|-------|-------------|---:|--------|
| [cyankiwi/MiniMax-M2.7-AWQ-4bit](https://huggingface.co/cyankiwi/MiniMax-M2.7-AWQ-4bit) | 240B (11B active) | compressed-tensors W4A16 | MoE (256 experts) | 8 | Working |
| [cyankiwi/Qwen3.6-27B-AWQ-INT4](https://huggingface.co/cyankiwi/Qwen3.6-27B-AWQ-INT4) | 27B | compressed-tensors W4A16 (asymmetric) | Hybrid Gated DeltaNet | 4 | Working (greedy + tool-calling smoke) |
| [cyankiwi/granite-4.1-8b-AWQ-INT4](https://huggingface.co/cyankiwi/granite-4.1-8b-AWQ-INT4) | 8B | compressed-tensors W4A16 group_size=32 (asymmetric) | Dense (GraniteForCausalLM) | 2 | Working (cudagraph; ~127 tok/s single-stream, ~587 tok/s aggregate batch=8) |
| [Intel/DeepSeek-V4-Flash-W4A16-AutoRound](https://huggingface.co/Intel/DeepSeek-V4-Flash-W4A16-AutoRound) | 290B (37B active) | auto-round W4A16 | MoE (256 experts) + MLA + sparse-attn + Hyper-Connections | 8 | Working (single-request, ~5.66 tok/s decode-only) |
| [bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF](https://huggingface.co/bartowski/mistralai_Mistral-Small-4-119B-2603-GGUF) (Q4_K_M) | 119B | GGUF Q4_K_M | MoE + MLA (`Mistral4ForCausalLM`) | 8 | Working (cudagraph; ~82 tok/s short prompt, ~24 tok/s @ 6k-tok prompt, ~26 tok/s prefix-cache replay) |
| [bartowski/MiMo-V2.5-GGUF](https://huggingface.co/bartowski/MiMo-V2.5-GGUF) (Q3_K_M) | 310B (15B active) | GGUF Q3_K_M | MoE + hybrid SWA + asymmetric head_dim (`MiMoV2FlashForCausalLM`) | 8 | Working (cudagraph; ~42 tok/s single-stream, ~64 tok/s aggregate batch=8) |
| [bartowski/Qwen_Qwen3.6-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF) (Q8_0) | 35B (3B active) | GGUF Q8_0 | MoE (256 experts) + hybrid Gated-DeltaNet (`Qwen3_5MoeForCausalLM`) | 2 | Working (cudagraph; ~100 tok/s single-stream, ~1900 tok/s aggregate 4×TP2) |

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

### Quick run (granite-4.1-8b-AWQ-INT4 on 2x V100 32GB)

Pure dense `GraniteForCausalLM`, asymmetric compressed-tensors W4A16 with
`group_size=32`. Uses the existing `TurboMindAsymLinearKernel` (already in
this fork; no new code path needed). The
`compile_ranges_split_points:[]` setting disables the chunked-prefill
split that otherwise triggers a silent `FLASH_ATTN_V100` fallback path
producing all-token-id-0 ("!") garbage. Cudagraph capture engages
cleanly -- do **not** add `--enforce-eager` (eager mode is ~3x slower
on this model). Local bench (TP=2, dual V100 32GB SXM2, 32-prompt ->
128-gen): 126.6 tok/s decode at batch=1; 586.8 tok/s aggregate / 73.3
per-seq at batch=8.

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/cyankiwi/granite-4.1-8b-AWQ-INT4 \
  -e VLLM_SERVED_MODEL_NAME=granite-4.1-8b-AWQ-INT4 \
  -e VLLM_QUANTIZATION=compressed-tensors \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  -e VLLM_MAX_MODEL_LEN=8192 \
  -e VLLM_MAX_NUM_SEQS=16 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=4096 \
  -e VLLM_COMPILATION_CONFIG='{"compile_ranges_split_points":[]}' \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --attention-backend FLASH_ATTN_V100
```

### Quick run (DeepSeek-V4-Flash on 8x V100 32GB)

Single-request only (`bsz==1`); compressor and indexer KV are kept on
module buffers rather than the paged cache for now. `--enforce-eager`
is required (cudagraph engagement is blocked by three uncaptureable
paths in the model -- TileLang JIT, TileLang deprecation warn, and a
Hash-MoE Python-state contract; the realistic post-cudagraph speedup
ceiling is also bounded by TP all-reduce dominating ~38% of decode-time
GPU work, so eager is the practical ship target on V100 SXM2).
`--max-num-seqs=4` is the sampler warmup OOM headroom; `block_size=64`
matches the V100 sparse-attn kernel's `BLOCK_N`. Decode-only throughput
in this configuration is ~5.66 tok/s warm (median ~5.27 across 4
fresh-process runs at TP=8, 4096-token context).

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Intel/DeepSeek-V4-Flash-W4A16-AutoRound \
  -e VLLM_SERVED_MODEL_NAME=V4-Flash-W4A16 \
  -e VLLM_QUANTIZATION=auto-round \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=8 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.85 \
  -e VLLM_MAX_MODEL_LEN=4096 \
  -e VLLM_MAX_NUM_SEQS=4 \
  -e VLLM_BLOCK_SIZE=64 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --enforce-eager \
  --no-enable-prefix-caching
```

### Quick run (Mistral-Small-4 119B Q4_K_M GGUF on 8x V100 32GB)

Bartowski's GGUF for `mistralai/Mistral-Small-4-119B-2603`. `VLLM_MODEL`
points at the first GGUF shard; `--tokenizer
mistralai/Mistral-Small-4-119B-2603` (passed through `$@`) overrides the
GGUF-embedded tokenizer with the HF one, which carries the official chat
template that emits the
`[MODEL_SETTINGS]{"reasoning_effort":"none"}[/MODEL_SETTINGS][INST]...[/INST]`
envelope the model was trained on. Cudagraph capture engages -- do
**not** add `--enforce-eager`. Prefix caching is supported (the LSE-SDPA
fallback in `mla_attention.py` keeps `merge_attn_states` happy on V100).
Local bench (TP=8, max_model_len=16384): ~82 tok/s decode on a short
prompt, ~24 tok/s decode at 6k-token prompt + 512-token generation
(chunked prefill on, `max_num_batched_tokens=2048`), ~26 tok/s decode on
prefix-cache replay.

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/bartowski/mistralai_Mistral-Small-4-119B-2603-Q4_K_M/mistralai_Mistral-Small-4-119B-2603-Q4_K_M-00001-of-00002.gguf \
  -e VLLM_SERVED_MODEL_NAME=Mistral-Small-4-119B-Q4_K_M \
  -e VLLM_QUANTIZATION=gguf \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=8 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.70 \
  -e VLLM_MAX_MODEL_LEN=16384 \
  -e VLLM_MAX_NUM_SEQS=1 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --tokenizer mistralai/Mistral-Small-4-119B-2603 \
  --enable-prefix-caching
```

### Quick run (MiMo-V2.5 310B Q3_K_M GGUF on 8x V100 32GB)

Bartowski's GGUF for `XiaomiMiMo/MiMo-V2.5`. Three things make this launch
non-trivial vs Mistral4: (a) `--hf-config-path` routes config through the
full HF repo because transformers' GGUF parser doesn't have `mimo2` in
its arch allowlist, (b) `--hf-overrides` strips the fp8 native-quant
declaration plus the unused vision/audio/processor sub-configs, and
(c) `--trust-remote-code` is needed for the GGUF loader's dummy
meta-model build (transformers ships no native `MiMoV2` class).
Cudagraph capture engages -- do **not** add `--enforce-eager`. Tool
calling works via the `qwen3_coder` parser (MiMo's
`<tool_call><function=...><parameter=...></parameter></function></tool_call>`
envelope is token-identical to qwen3-coder's). Local bench (TP=8,
max_model_len=4096, cudagraph + chunked-prefill + prefix-cache,
max_num_seqs=8): ~42 tok/s single-stream short-decode, ~64 tok/s
aggregate at batch=8.

```bash
docker run --rm --gpus all --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/bartowski/MiMo-V2.5-GGUF/MiMo-V2.5-Q3_K_M/MiMo-V2.5-Q3_K_M-00001-of-00004.gguf \
  -e VLLM_SERVED_MODEL_NAME=MiMo-V2.5-Q3_K_M \
  -e VLLM_QUANTIZATION=gguf \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=8 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.92 \
  -e VLLM_MAX_MODEL_LEN=4096 \
  -e VLLM_MAX_NUM_SEQS=8 \
  -e VLLM_MAX_NUM_BATCHED_TOKENS=2048 \
  -e VLLM_TOKENIZER=XiaomiMiMo/MiMo-V2.5 \
  -e VLLM_HF_CONFIG_PATH=XiaomiMiMo/MiMo-V2.5 \
  -e VLLM_HF_OVERRIDES='{"quantization_config":null,"vision_config":null,"audio_config":null,"processor_config":null}' \
  -e VLLM_TRUST_REMOTE_CODE=1 \
  -e VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Quick run (Qwen3.6-35B-A3B Q8_0 GGUF on 2x V100 32GB)

Bartowski's GGUF for `Qwen/Qwen3.6-35B-A3B`. The text GGUF (arch `qwen35moe`)
carries no vision tensors, so `--hf-config-path` routes config through the full
HF repo (transformers' GGUF parser has no `qwen35moe`) and `--hf-overrides`
binds the text backbone `Qwen3_5MoeForCausalLM` while nulling `vision_config`.
TP=2 is the minimum -- the 35 GiB Q8_0 weights don't fit one 32 GiB card.
`--mamba-cache-mode align` unifies the hybrid Gated-DeltaNet recurrent state
with the full-attention KV pages. Cudagraph capture engages -- do **not** add
`--enforce-eager` (it costs ~11× decode). Local bench (TP=2, max_model_len=8192,
cudagraph): ~100 tok/s single-stream; ~1900 tok/s aggregate across a 4×TP=2
replica fleet (8 GPUs). The native MTP speculative-decode head loads but is
net-negative on V100 (see notes above), so it is left disabled here.

```bash
docker run --rm --gpus '"device=0,1"' --ipc=host \
  -v /path/to/models:/models:ro \
  -e VLLM_MODEL=/models/Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q8_0.gguf \
  -e VLLM_SERVED_MODEL_NAME=Qwen3.6-35B-A3B \
  -e VLLM_QUANTIZATION=gguf \
  -e VLLM_DTYPE=float16 \
  -e VLLM_TENSOR_PARALLEL_SIZE=2 \
  -e VLLM_GPU_MEMORY_UTILIZATION=0.90 \
  -e VLLM_MAX_MODEL_LEN=8192 \
  -e VLLM_TOKENIZER=Qwen/Qwen3.6-35B-A3B \
  -e VLLM_HF_CONFIG_PATH=Qwen/Qwen3.6-35B-A3B \
  -e VLLM_HF_OVERRIDES='{"architectures":["Qwen3_5MoeForCausalLM"],"vision_config":null}' \
  -e VLLM_TRUST_REMOTE_CODE=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000 \
  -p 8000:8000 \
  vllm-v100:latest \
  --mamba-cache-mode align
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
