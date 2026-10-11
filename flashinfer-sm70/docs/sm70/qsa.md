# Request-grouped QSA scoring candidate

Status: development, **independent GPU correctness passed; acceleration admission pending**.
The provided V100 passes the 18 standalone QSA cases described in the
[hardware validation record](gpu-validation-20261011.md). This implementation covers the
score stage only. Projection, compression/cache update, exact Top-K, index
expansion, sparse attention and output processing remain open in the
[complete-chain ledger](coverage.md).

## Native implementation and ownership

`flashinfer.sm70.QsaScorePlan` owns explicit score/visibility and grouping
workspaces. `csrc/sm70/qsa.cu` registers a framework-independent TVM-FFI binding;
`include/flashinfer/sm70/qsa.cuh` owns both kernels. The module does not import
vLLM, call `vllm._C`, or borrow an engine-built shared library.

The Volta `mma.sync.aligned.m8n8k4` arithmetic follows the existing single-request
scorer at engine commit `be3722bc095b9792bf9833068595f0b1ef70ad37`. For C1 with
at most eight rows, scoring and visibility use one launch. For multiple requests,
a small GPU kernel creates stable per-request row lists; scoring reuses each key
tile across up to eight query rows from the same request. C4 with five rows per
request is covered, including interleaved row mappings. No host inspection of
request IDs or sequence lengths, host synchronization, `ldmatrix`, `cp.async`,
TMA, BF16 or FP8 Tensor Cores are involved.

Version `.3` accumulates the grouped dot product in two 64-element parts before
combining them in FP32. This fixes three reproduced cancellation cases from the
128K engine comparison without widening the existing tolerance. C1 with at most
eight rows retains the original accumulation order. The normal installed `.3`
library passes the standalone suite and the engine's complete 8K/32K/128K
score/selection screen. Its measurements are recorded separately from `.2`;
complete QSA and model admission remain open.

Supported candidate inputs: FP16 query `[M,4,128]`, M1..32; FP16 cache
`[pages,page_size,1,128]`; int32 page tables with 1..32 requests; contiguous int32
request IDs/lengths and int32/int64 positions. Query/cache/table strides are
explicit. Paged layout, score divisor, compression ratio and causal visibility
remain caller-owned. Invalid physical pages produce negative infinity;
invalid request IDs have non-positive visibility. Only columns below
`min(visible[row], columns)` are defined. Selection must mask all other columns.

The library plan also handles zero rows/columns. Score accumulation and the
nonnegative head sum preserve the inherited native scorer's floating contract;
the independent oracle uses the existing 2e-6 absolute/relative tolerance
and exact selected IDs, without widening thresholds. `fmaxf(dot, 0)` governs NaNs.

The engine's production implementation stays in its existing build until this
candidate passes numerical and complete-chain gates. It is a rollback route,
not a dependency of the new library. After admission, move the production
selection to the library and remove the redundant native build target in a
separate reviewed commit within the consolidated PR.

## Lifecycle and engine selection

Construct `QsaScorePlan(max_rows, max_requests, max_columns, device)` before
capture. Compilation and allocation are refused during capture. `run(...)`
reuses its instance-owned buffers and takes all cache/metadata inputs explicitly.
Dynamic valid lengths and request membership are read on the GPU each replay.
Weight/cache contents may change in place; replacing addresses requires graph
recapture. Eager calls may bind new storage. Results are borrowed plan storage
and must be consumed before the next call. Plans are stream-ordered; independent
concurrent instances/captures need separate plans.

The engine adapter is `vllm/model_executor/kernels/flashinfer_sm70.py`. Configure
the existing kernel configuration with:

```json
{"sm70_flashinfer_qsa": "candidate"}
```

`off` is the default. `candidate` tries supported scoring shapes and records why
an existing route was retained. `force` rejects unsupported shapes/workspace
limits and prevents cuBLAS/request-segmented prefill from bypassing the candidate;
it is for exact-shape validation, not arbitrary whole-model prefills. An official
FlashInfer installation or a mismatched fork version raises before use.
There is no automatic performance-admitted mode.

Each engine indexer has one plan, at most 16 MiB. The first supported eager call
fixes its column capacity. Smaller executions reuse it; larger columns are
rejected rather than replacing captured storage. Warm up the full cache layout
before capturing any graph. Plan provenance records the installed version,
source/compiler identity, **actually loaded** DSO and workspace bytes; later
source edits cannot relabel an already loaded kernel. Total layer/graph workspace
and OOM behavior still require model-level validation.

## Validation scope

Independent tests in `tests/sm70/test_qsa.py` cover M1/M5/M20/M32, C1/C4,
contiguous/strided tensors, real page size 204 and small page boundaries, invalid
pages, ties, NaN/extremes, negative padding positions, empty rows/columns,
changing graph metadata, separate captures and cache rebinding. They use a
library-independent FP64 oracle and exact stable Top-K IDs. Together with three
fixed cancellation fixtures, all 18 cases pass
on V100-SXM2 / CUDA 12.8. They do not replace engine exact-selector, state rollback or model
output/acceptance tests.

The existing engine screen accepts the independent forced provider:

```bash
.venv/bin/python benchmarks/kernels/benchmark_sm70_qsa_shared_key.py \
  --flashinfer --rows 1 4 5 20 --context 8192 --out qsa-screen-8k.json
```

Repeat at 32768 and 131072, then at every supported long-context boundary with
the model workload. The screen first checks scores and exact selected/expanded
indices against the existing route, then performs A/A and paired ABBA graph
timing with retained samples. An A/A median difference above 2% aborts it.
Its scopes are only `score` and `score_select_expand`, across 12 synthetic local
layers. Reports explicitly set `complete_qsa_chain=false` and
`performance_admitted=false`: even a fast result cannot close full QSA or C1/C4
emitted-token performance. The [hardware record](gpu-validation-20261011.md)
records `.2` screening gains, the failed 128K/M20 comparison and the passing `.3`
accuracy/performance screen separately; no complete-chain admission is claimed.

On the provided V100-SXM2 machine use ordinary installed artifacts with Python
3.12, Torch 2.10 and CUDA 12.8. Establish the frozen mainline A/A control first.
Complete QSA timing must include all projection/state/attention/output work, and
preserve cache placement, MTP acceptance/rollback, exact selection, all supported
TP/PP/DCP combinations and final quality. Admission remains at least 5% complete
chain improvement beyond noise and no more than 2% regression at other key
points; integrated text/MTP/vision and shared-model regressions remain separate.
