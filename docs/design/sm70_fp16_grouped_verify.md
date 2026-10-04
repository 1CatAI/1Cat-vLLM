# FP16 KV grouped verification on SM70

The existing grouped verifier requires FP8 KV. A Qwen3.8-27B GGUF target
served with FP16 KV therefore takes per-query XQA even with a 32K or 256K
service window. Its local shape is six Q heads, one KV head and D=256, with
832-token hybrid cache pages. Widening the FP8 guard would reinterpret the
cache incorrectly and would not preserve the existing FP32 probability path.

Add a separate FP16 entry in the shipped FA2 extension. Pack up to eight
query rows per request, share each KV tile across those rows, and retain
FP32 softmax probabilities, PV arithmetic, numerator, maximum and sum. Reuse
the existing ordered FP32 partition reduction. Explicit device row lengths
retain padding and causal boundaries during graph replay. All-zero requests
return zero without reading stale partition storage.

## Admission

Default admission requires SM70, FP16 Q/K/V and output, local Q/KV heads=6/1,
D=256, page=832, full causal context, a capacity no greater than 266240 tokens,
and supported aligned KV strides. B1 supports q2..8; B2..4 supports request-major
q8. Unmeasured pages, head shapes, row counts, sliding windows and missing
operators retain XQA or the ordinary fallback. Startup capability reporting
records a missing operator and lists the runtime guards. No new environment
variable or precision reduction is involved.

Workspaces are allocated once per device/stream and retained for captured
graphs. Growing the batch panel keeps earlier buffers alive; successive layers
reuse buffers in stream order. Engine shutdown clears this workspace bank.

## Operator measurements

The independent prototype uses a 300 W V100, CUDA 12.8, Torch 2.10.0+cu128,
page=832, six/one heads and D=256. Each of eight graph calls is preceded by a
16 MiB L2 eviction buffer; report the median of seven replays with external
CUDA events. The reference uses FP32 QK/softmax/PV with TF32 disabled, starting
from exactly the same FP16 operands.

| Context | M | Grouped µs | XQA P256 µs | Grouped relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 2 | 25.600 | 62.464 | 0.0002045 |
| 1024 | 8 | 39.936 | 63.488 | 0.0002040 |
| 8192 | 2 | 51.200 | 71.680 | 0.0002034 |
| 8192 | 8 | 75.776 | 229.376 | 0.0002062 |
| 32768 | 2 | 118.784 | 257.024 | 0.0002024 |
| 32768 | 8 | 192.512 | 743.424 | 0.0002061 |

The original model trace uses XQA partition=1024, not P256. Repeating the M8
comparison with P1024 and a 262144-token served-window capacity gives:

| Live context | Grouped µs | XQA P1024 µs | Delta for 16 calls ms |
| ---: | ---: | ---: | ---: |
| 1024 | 46.080 | 276.480 | 3.6864 |
| 8192 | 87.040 | 282.624 | 3.1293 |

These are separate cohorts and standalone operators, not a model speedup.
The graph-linked original trace records 16 XQA calls at 235.465 µs each;
using the 1K grouped measurement gives about 3.03 ms less traced attention
service, or about 2.73 ms with the provisional 0.9 profiler correction.
New same-contract model/trace measurements must account for overlap and any
remaining mismatch in native kernel planning.

Unique KV bytes are `2 * context * 256 * 2` (1048576 bytes at 1K,
8388608 at 8K). They exclude partition workspace, Q/output and repeated
cache traffic. Effective unique-KV bandwidth is not a DRAM-counter result.
The independent kernel uses 112 registers and has no spills or stack frame.

## Correctness and graph edges

Six further cases cover strided KV [439296,264,264,1], B1/B4, padding rows,
a wholly padded request, large Q values, and contexts 1091/8192/131072/262144.
All graph replays match eager output exactly. Updating lengths to all zeros
inside the existing graph produces exact zero output. Relative L2 against
FP32 reference stays below 0.000213; ordinary-activation maximum absolute
error is at most 6.07e-5, and the large-Q case is 0.000971 after FP16 output
rounding. Thirteen CPU admission checks pass.

## Normal package validation

The normal CMake FA2 target and complete wheel at source
`96d2b8efb28c2274bba8acad6b43fb3306ccdd7b` build and install successfully.
The wheel passes 100 CPU loading/admission checks in 11.45 seconds and five
GPU reference/graph checks in 3.67 seconds. The latter exercise q2/q8,
strided KV, C4, 128K/256K context and zeroed device lengths.

Five Python modules match source, wheel and installed contents exactly.
All sixteen native libraries match either the qualified base (fifteen) or
the newly built FA2 artifact. ELF dependencies contain only standard
Torch/CUDA/cuBLAS/system libraries. A fresh process registers the native
operator from the normal package without an external library override.

The actual backend method selects the FP16 grouped route. Its graph replay
matches the native entry exactly, with noncontiguous KV strides
[439296,264,264,1]. Cold-L2 normal-wheel timings are:

| Requests | Rows/request | Live context | Backend µs |
| ---: | ---: | ---: | ---: |
| 1 | 8 | 1024 | 39.936 |
| 1 | 8 | 8192 | 76.800 |
| 4 | 8 | 8192 | 237.568 |

The complete model/trace comparison follows the shared floating-projection
loading integration boundary. These package/backend results do not establish
complete-round speed or model-level acceptance equivalence.
