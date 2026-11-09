# Reproduce the Qwen3.8 NVFP4 single-request baseline from source

This is the quality-repaired **no-MTP** lane, not the experimental FlashInfer,
E4M3-KV, batched, or projection/W13 candidates. The production kernels were
integrated by [PR #525](https://github.com/1CatAI/1Cat-vLLM/pull/525), including
the HC and QSA/router commits from #506/#507. A stacked PR remaining open is
not proof that its commits are absent from `main`.

The integration audit found these accepted components already in the base
`95205a2d9952813aa7469f63ff65b8f2813c027a`; this PR adds their reproducible
build/launch packaging, not a second copy of the optimizations:

| Component | Main-source implementation |
|---|---|
| HC projection/mix and TP4 communication | `vllm/models/qwen4_exp/nvidia/sm70_fp16_hc.py`, `csrc/custom_all_reduce.cu` |
| NVFP4 M1 experts, fused W13 and direct W2 reduction | `csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`, `vllm/_sm70_ops.py` |
| Exact QSA top-k and router | `csrc/qsa_lexicographic_topk.cuh`, `vllm/models/qwen4_exp/nvidia/ops/qsa.py`, `vllm/model_executor/layers/fused_moe/fused_topk_router.py` |
| Flash-V100 logical-page-order quality repair | `flash-attention-v100/kernel/fused_mha_forward_paged.cu` and its public planner regression |
| GDN and fused FP16 projections | `flash_qla/`, `vllm/models/qwen4_exp/nvidia/sm70_fp16_gemv.py` |
| Hybrid PLE and unified prefill/decode configuration | `vllm/v1/ple_offload/`, `vllm/config/vllm.py` |

PR #507 was closed as already incorporated through #525. Open #510 and the
FlashInfer/E4M3 experiments are not part of this accepted single-request lane.

## What this entry point fixes

Previous local launchers selected HC, NVFP4 W13/W2, QSA top-k, FlashQLA and
Flash-V100 binaries from old experiment directories. Source being merged did
not by itself make those launchers reproducible from a clean checkout.

The new builder compiles the production sources in the selected checkout and
records source/binary hashes. Public sidecar bindings replace the formerly
local HC/NVFP4 wrappers; they do not copy or modify kernel algorithms. The HC
communicator lifecycle and every operation using its opaque pointer stay in
the same DSO. The benchmark rejects changed sources, changed libraries, or a
library pointing outside its runtime bundle, and verifies worker mappings.

This is a **source-overlay** reproduction route. A compatible native SM70
vLLM installation is still required for the remaining standard operators;
this command is not a full wheel rebuild. The worker report records those
native dependency paths and hashes instead of concealing that dependency.
An ordinary editable/source build with native extensions in `vllm/` can omit
`--native-extension-dir`. The optional public bootstrap only attaches the
specified native extension directory; it does not patch model execution.

## Fixed contract

| Item | Value |
|---|---|
| Model | RadixArk/Qwen3.8-Flash-Next-NVFP4 |
| Hardware | Four peer-connected V100-SXM2-32GB; TP4/PP1, one request |
| Precision | FP16 activations/KV, checkpoint NVFP4 experts via TurboMind W4A16 |
| Recurrent state | `mamba_ssm_cache_dtype=auto`, resolves to native FP32 |
| Runner | V2, dynamic prefill and full static decode CUDA Graph |
| Context/chunk | 262144 total tokens; prefill chunk8192 |
| PLE | Disk mmap prefill, rank-local pinned-UVA decode |
| MTP/prefix cache | Both off |
| Short speed workload | Fixed8192-token prompt,513 generated tokens,512 decode intervals |
| Sampling | Greedy/ignore-EOS for timing only; official sampling/natural EOS for health |
| Excluded routes | Online QPN8, approximate LM-head, experimental batch sidecars |

The validated environment uses Torch2.10.0+cu128, CUDA compiler12.0.140,
Triton3.6 and NVIDIA driver580.173.02. The compiler/runtime minor versions are
different in this recorded environment; no toolkit or dependency upgrade is
implied by the reproduction recipe. Preserve this environment when comparing
against the recorded baseline. Compiler flags retain the existing production
build recipe; this change introduces no lower-precision kernel route.

## Build and run

Run from the source checkout, with the project Python environment. Replace
the example paths. The build needs no GPUs and does not install into or change
the shared Python environment.

```bash
CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 \
  .venv/bin/python benchmarks/kernels/build_sm70_qwen38_runtime.py \
  --output-dir /path/to/task/runtime
```

Reserve four idle GPUs before running; do not preempt other jobs. The tested
hybrid PLE setup needs at least90GiB of available host RAM at admission. This
is not the model's total host-memory footprint. Keep the output/cache directory
private to this run.

```bash
CUDA_HOME=/usr CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --runtime-dir /path/to/task/runtime \
  --native-extension-dir /path/to/compatible/site-packages/vllm \
  --output /path/to/task/results/result.json \
  --repeats 3 --long-context
```

Without `--long-context`, only the short health and8192-token baseline run.
With it, the same instance also measures261631+513, checks retrieval of a
middle-position record with natural EOS, and exercises262143+1 at the exact
context boundary. This is a focused regression, not broad quality certification.
All workers shut down in `finally`; no API is kept resident. A caller may apply
an outer timeout, for example `timeout --kill-after=35s 1800s ...`.

`--reference-json /path/to/previous/result.json` accepts the retained length
sweep's `cases[].runs[]` format and requires identical complete output tokens
for every requested fixed-prompt case (missing cases are rejected). The warmup
and timed repeats must also agree.
The driver does not change sampling to conceal early EOS or numerical drift.

To test model-aware selection instead of the explicit baseline flag map, add
`--use-defaults`. See [the default-path audit](sm70_qwen38_default_fastpath.md)
for its scope and current validation status. The precision, workload,
reference-token checks and shutdown behavior are unchanged.

## Accepted historical evidence and fresh-build acceptance

The unprofiled `main` source95205a2d9952 sweep used physical GPUs4-7 with the
fixed contract above. Two warmed repeats per case measured:

| Input | Prefill tok/s | Decode tok/s | TPOT ms |
|---:|---:|---:|---:|
|8192|6936.60|97.826|10.222|
|65536|6245.95|90.729|11.022|
|131072|5830.36|83.814|11.931|
|261631|5137.10|73.977|13.518|

These are the previous frozen-library results, not measurements of the new
builder. The fresh short-context reproduction follows below; keep the two
sets of evidence distinct.
Do not substitute an Nsight graph interval for endpoint TPOT, or call a
configured256K maximum a tested256K input.

The follow-up length trace attributed98.989% of the increase in graph kernel
service to QSA Top-K and compressed-key scoring. It did not implement a new
long-context optimization. The current Top-K still has its single-CTA long-row
fallback; these reproduction changes do not claim reduced context decay.

### Fresh-source baseline, 2026-09-06

Source `b2042fb24b78bba131ec9aa0b1a05cdf3f54df60`, physical GPUs4-7, with all
six overlay libraries rebuilt by the public script and independent compilation
caches. This is one model initialization, with two warmed timing repeats:

8192-token input:

| Measurement | Run1 | Run2 | Mean |
|---|---:|---:|---:|
| Decode tok/s |97.71679|97.73243|**97.72461**|
| TPOT ms |10.23366|10.23202|**10.23284**|
| Prefill tok/s |6890.26|7010.06|**6950.16**|

The complete513-token outputs of both repeats and their warmup match the
accepted frozen-library sweep exactly, not just the first token or a text
prefix. The arithmetic and record-copy checks both stop naturally and pass.
All four workers verify this checkout, native FP32 SSM, FP16 KV, no MTP/prefix
cache, `FULL_AND_PIECEWISE`, QSA specialization version1, and the five loaded
optimized DSO hashes. The sixth built library is the paged-KV utility.

The same instance's261631+513 case measures **73.98534 decode tok/s**, **5169.99
prefill tok/s**, and **13.51619ms TPOT** (two warmed repeats). Both complete
513-token outputs and the warmup match the corresponding frozen-sweep output
exactly. This reproduces the existing length-dependent baseline; it is not a
new long-context speed optimization.

Model-free checks of these new builds also pass: HC256 graph replays with zero
bit mismatches on every rank; router1280 bitwise rows; QSA72 exact cases and
128 changing-length graph replays/1536 row comparisons. Eleven CPU helper tests
and the scoped pre-commit suite pass. Long-context completion, detailed runtime
manifests and the final integration audit are recorded in
[PR #532](https://github.com/1CatAI/1Cat-vLLM/pull/532).

The compatible native-base dependency hashes observed on every worker are:

| Native library | SHA256 |
|---|---|
| `_C.abi3.so` | `c35b76ca723ec1a6907e31b2f0fe4f96c2e1b212603e04b976f9a48b85d0916a` |
| `_C_stable_libtorch.abi3.so` | `8c866c3612bbbe2323ff4cc912f5fe3924d6f0517ee675f3f40230c151854890` |
| `_moe_C.abi3.so` | `80d92f26cbed180bf538ce3254534f32581d89741953ee6ee45e4e9e63ece85e` |

This reproduces approximately98 decode/7K prefill tok/s under the stated
contract; it is not a100 tok/s claim or a universal speed/quality guarantee.
The bounded output agreement does not close the broader GDN/W13
actual-input numerical-reference work described in the
[quality repair report](sm70_qwen38_nvfp4_quality_repair.md).

## Focused checks

Pure configuration/provenance tests do not require the GPU test conftest:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q \
  --confcutdir=tests/benchmarks tests/benchmarks/test_sm70_qwen38_baseline.py
```

Use the freshly built runtime paths when running existing HC raw-bit,
QSA/router, and page4 relocation tests. Keep numerical admission separate from
speed: a startup or nonempty response is not an output-quality gate.
