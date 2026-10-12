# Shared paged QSA codec qualification

QSA now consumes the common ordinary FP16/E4M3/INT8 payload and scale view.
Its page4 planner, grouped union/masks, XQA tail, Triton split-K/merge,
request mapping and workspace owners remain shared. The main attention owner
and backend admit explicit INT8; the ordinary native writer publishes its
payload and FP32 token/head scales. Existing weight, device and numerical
qualification guards remain in place.

## Owner page mapping

Planar INT8 scales make small physical pages non-integral multiples of a
four-token payload tile. Dividing the physical byte stride by tile bytes
therefore cannot identify the corresponding scale row. `KVPageMapping`
describes the logical page4 ID's owner and token offset. New native view
bindings receive original owner tensors and their actual strides for both
payload and scales. The planner and mask/softmax/merge algorithms do not
branch on the codec.

Flat ordinary XQA/grouped instances compile out this mapping. The old XQA
and grouped entry points retain their ABI. Older FP16/FP8 binaries can use
the qualified affine microtile mapping; fractional physical strides fall
back before the old planner. INT8 requires the actual mapped-view binding
before native workspace allocation. Such a fallback does not close a missing
fast-path audit item.

The new public native probes are `decode_paged_xqa_kv_view_abi=1`,
`grouped_sparse_page4_kv_abi=1` and
`grouped_sparse_page4_kv_view_abi=1`. Grouped page4's declared format set
uses the existing compile-time codecs; INT8 is not a second executor.

## Numerical and mask repairs

The reference INT8 encoder now explicitly emulates RN FP32 division. Torch
CUDA may replace scalar division by a rounded reciprocal, producing a scale
one ULP away from the already qualified CUDA/Triton writer. A fixed binary
scale regression and exact payload/scale comparisons preserve the storage
contract. Legacy truncation and its division/multiplication order are unchanged.

New sparse-mask checks exposed a parent-version planner failure for **all
three codecs**, not a new INT8-only arithmetic issue. A negative first
selection incorrectly discarded other valid selections in that four-column
group; visibility and the remainder tail also relied on a dense prefix.
The common grouped planner now visits actual selected tokens, applies the
query visibility bound, and inserts their original page/token masks.
Its union, category packing, logical sort and reduction remain shared.

XQA retains its complete aligned page4 plus causal-tail selection contract.
Its table builder now verifies complete groups and counts actual valid pages,
so empty selections and invalid physical pages cannot become phantom native
pages. Arbitrary within-page masks belong to the grouped/Triton algorithms.
The selection producer declares `selection_group_size=4` for complete aligned
groups; the engine forwards its compressor's grouping. A plain token list
does not acquire XQA compatibility from its width. Unsupported masks reach
the shared grouped/Triton computation without a host synchronization.

The ordinary attention bounds remain `atol=0.004, rtol=0.015`. Existing DCP
output/LSE bounds remain unchanged. No failed threshold was loosened.

## Evidence

The normal Flash-V100 wheel SHA256 is
`74c925d18e17b871e29b3ebd6472a49c79916f0d1f3055959b2174089434c4a2`.
This is a normal component build with a recorded source overlay, not final
whole-wheel qualification. Parent CMake components are installed by their
normal targets; no benchmark-only extension is loaded by these tests.

**79 GPU checks pass**: 48 generated FP16/E4M3/INT8 cases cover page64/2048,
non-contiguous pages, page padding, q1/q8/q17, Triton/grouped/XQA tail and
changed-input graph replay. Further cases cover exact native/Triton/reference
publication, actual CUDA codec-3 XQA and grouped events, sparse masks, empty
rows, unselected non-finite pages, valid-page counting and two-rank DCP
output/LSE composition. Nine existing E4M3 regressions pass in the same run.

**118 affected flat-algorithm and engine-binding checks pass** separately.
This set includes GPU kernels and CPU routing contracts; its total is not
reported as a GPU test count. Physical mapping, legacy-binary admission,
reference division, main-owner cache admission and scale-owner allocation
are checked independently on CPU. CPU engine contracts do not prove complete
engine execution or task quality.

Retained basenames are `gpu-qsa-contract-r5.log`,
`gpu-qsa-flat-regressions-r4.log`, `build-manifest-r2.json`,
`gpu-qsa-full-r1.log`, `gpu-qsa-writer-r2.log` and
`gpu-qsa-contract-r3.log`. The initial reference-scale and parent planner
failures remain in those earlier logs.

## Remaining scope

Native grouped requires eight-row groups, D256/GQA6 and the existing page4
workspace budget. XQA retains its compatible geometry and selection layout;
other valid sparse inputs use the qualified common Triton implementation.
FP16-only dense/resolved-index optimizations remain their original boundary.
The existing direct-QKV prefill policy and independent compressed indexer
representation are unchanged.

Representative QSA engine startup/capture/replay, task quality, latency and
state/offload/transfer acceptance remain open. History's separate scale owner
still requires L04/L05 lifecycle evidence. This component stage does not
complete R01/R02 or the full 43-item migration.
