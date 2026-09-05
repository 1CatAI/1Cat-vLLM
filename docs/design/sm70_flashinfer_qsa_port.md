# FlashInfer CUDA QSA port to Volta

## Purpose and frozen baseline

Port actual FlashInfer computation to SM70, starting with sparse QSA decode.
Changing a backend name or forwarding to the existing Triton/Flash-V100 kernel
does not meet this objective. This first PR is an isolated benchmark prototype;
it does not select a new serving backend or change release defaults.

- Integration: `1CatAI/1Cat-vLLM`, `onecat/main`.
- Base: `755baae1d075ee04fa9096b23fc0225b23589a86`.
- Branch: `codex/v100-flashinfer-qsa-sm70-20260905-164136`.
- FlashInfer source: [6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f](https://github.com/flashinfer-ai/flashinfer/tree/6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f).
- Its CCCL submodule: `16bd510c9b712e82b0ab6cbb630d8e29ba1f7116`.
- The previous WMMA primitive probe has its own older source pin; it is not
  silently upgraded or used as evidence for this new attention path.

Scope is different from the HC/QSA follow-up #504 and M1 QSA #507: this
instantiates FlashInfer's CUDA kernels, not another existing native-FA variant.
No MoE/HC changes from those worktrees are copied here.

The fixed no-MTP concurrency goals remain 238/420/728 aggregate decode tok/s at
C4/C8/C16, respectively (85/75/65% of C times 70 tok/s). The previously recorded
C16 model step was 27.125 ms; meeting 728 tok/s requires 21.978 ms. Component
microseconds must not be presented as a new model result.

## Actual implementation

`flashinfer-sm70/include/flashinfer/attention/sm70/qsa_decode.cuh` provides a
virtual page-size-one sparse cache adapter. It directly instantiates the pinned
upstream `BatchDecodeWithPagedKVCacheKernel` and `MergeStatesKernel` from
`include/flashinfer/attention/decode.cuh` and `cascade.cuh`. The upstream header
tree is unmodified. QK, online softmax, PV, split-state normalization and merge
therefore execute FlashInfer's actual CUDA implementation.

Three GPU operations are measured together:

1. Expand raw logical QSA selections into physical 64-bit offsets and prepare
   persistent split metadata. Preserve order and duplicate weighting.
2. Execute FlashInfer paged decode using these sparse virtual pages.
3. Merge FP32 partial outputs/LSE with FlashInfer's cascade kernel; cast once
   to FP16 at the final output.

SM70 compatibility is obtained by instantiating the upstream SIMT/GQA kernel.
Its `cp_async.cuh` already has ordinary vector-load/store fallback below SM80,
and the decode body retains block barriers. This first candidate does not use
Tensor Cores and does not claim to replace tensor-core instructions by equally
fast software emulation. If SIMT loses, optimize tile reuse and evaluate native
Volta WMMA with the same softmax/masking contract before considering dispatch.

The Torch extension needs a consistent CCCL include set (Thrust, CUB and
libcudacxx) and undefines Torch's disabled CUDA half-conversion/operator macros.
These are build integration fixes, not a global relaxation of FlashInfer's
supported-GPU checks. No FlashInfer Python package is installed or shadowed.

The prototype covers FP16 Q/K/V, D256, 1..32 query heads, GQA groups 1/2/4/6/8,
1..64 splits, variable batch, selection width, page size and request mapping.
It accepts vector-aligned strided tensors and rejects misalignment. This is a
component contract, not a global max-seqs/TP/chunk/prefix-cache restriction.
Other dtypes, fused output gate, indexer, prefill and serving integration are
not implemented by this PR's first iteration.

QSA causality is already encoded by the existing selector/expander. Do not
reinterpret a query row as dense causal attention or physically sort pages.
Negative/out-of-range requests, indices, pages and split padding are masked.
Invalid cache loads use a persistent zero vector, not an arbitrary cache page:
zero probability multiplied by a cached NaN would still produce NaN.

## Test plan

No parameter change, routing-policy change, quantization or model approximation
is allowed as a substitute for a correct faster operator.

- Compare against an independent FP32 oracle that preserves repeated indices.
- Test empty/invalid rows and pages, tail and split padding, GQA layouts,
  non-contiguous tensors, and input/output pointer alignment.
- Poison output/partial/metadata buffers, mutate indices and physical mappings,
  then replay a captured graph. Require replay to equal this implementation's
  eager result and both to match the FP32 oracle (`atol=2e-3, rtol=1e-2`).
- For timed model-shaped cases require relative L2 error <= 5e-3. This is an
  operator screen, not a substitute for task-level quality tests.
- Time current two-warp Triton and FlashInfer with identical inputs, including
  preparation/merge. Record source SHA256, raw paired samples, B1/4/8/16,
  CUDA Graph and SM70. Reject results if another process uses the GPU.
- Only after a stable component gain: sanitizer, runtime metadata integration,
  representative C1/4/8/16 E2E and output-quality regression. No default change
  before these gates; no GPU-serving process left resident after testing.

## Reproduction

Use the task virtual environment backed by Torch 2.10.0+cu128, Python 3.12 and
CUDA 12.8, not system Python. Do not use CUDA 13 to compile SM70.

```bash
bash tools/prepare-flashinfer-sm70-qsa.sh
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export TORCH_EXTENSIONS_DIR="$PWD/.artifacts/torch-extensions"
export MAX_JOBS=2
# CPU-only compilation; no GPU or model loaded.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m benchmarks.kernels.flashinfer_sm70_qsa
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest -q \
  --confcutdir=flashinfer-sm70/tests flashinfer-sm70/tests/test_qsa_decode.py
# Acquire the shared GPU ownership locks and confirm an idle physical GPU first.
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest -q \
  --confcutdir=flashinfer-sm70/tests flashinfer-sm70/tests/test_qsa_decode.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
  benchmarks.kernels.benchmark_sm70_flashinfer_qsa --compare-triton
```

The comparison needs the source tree's usual working vLLM environment; the
FlashInfer-only tests require neither vLLM's native extension nor Flash-V100.

## Test result / status

- CPU compilation and shared-library loading: passed for native `sm_70`.
- `cuobjdump` confirms native `sm_70`; GQA6 decode uses 72 registers/thread
  and reports zero stack/local memory (not a measured occupancy or speed gain).
- CPU tests: **5 passed, 9 GPU tests skipped** (not counted as GPU passes).
- Python Ruff check/format and targeted C++/CUDA formatting: passed.
- GPU correctness, graph, sanitizer and speed: pending GPU ownership; another
  task currently holds the shared lock even when its model memory is released.
- Serving dispatch/defaults/quality or model speed: unchanged, not validated
  by this prototype. Do not advertise a throughput increase yet.

Build and test outputs are stored in this task's unversioned `.artifacts/` directory
(`flashinfer-qsa-build-v4.log`, `flashinfer-qsa-cpu-tests-v1.log`,
`flashinfer-qsa-resources-v1.txt`). The initial
build failures from mixed CCCL headers and disabled half conversions were
localized and fixed; do not rerun those variants as performance experiments.

## Subsequent source-port targets

The same pinned source already exposes useful next candidates, but backend
names alone do not determine implementation or speed:

| Area | Source evidence | Required work before a V100 claim |
| --- | --- | --- |
| GDN state update | `flashinfer/gdn_kernels/gdn_decode_pretranspose.py` uses CuTe DSL and cpasync, FP32 recurrent state | Port load pipeline/state layout to SM70, preserve gate math, state-pool read/write indices and graph lifetime |
| Fused convolution/GDN | `flashinfer/gdn_kernels/experimental/kernel/gdn_fused_decode_sm120.cu` is SM120-specific | Extract reusable fusion/dataflow; native SM70 instruction and reduction implementation, not just a capability bypass |
| TP communication | `include/flashinfer/comm/vllm_custom_all_reduce.cuh` explicitly derives from vLLM/SGLang; TRT-LLM fusion variants are separate | Diff actual algorithms, NVLink/P2P graph buffer ownership and barriers; measure message-size crossover and useful fusion, not duplicated old allreduce |
| Projections/HC | FlashInfer has GEMM/communication implementations, not a generic faster replacement for every pointwise Triton kernel | Select by measured operator shape, precision and collective placement; preserve model arithmetic and current native MoE path |

This work is AI-assisted. Keep implementation and unsuccessful paths in the
owned Draft PR; production admission requires independent correctness and
performance evidence, not compilation alone.
