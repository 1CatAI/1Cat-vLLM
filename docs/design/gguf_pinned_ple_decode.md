# Packed GGUF PLE decode from pinned memory

The Flash-Next GGUF PLE table uses IQ4_NL rows with width 160. A complete
320,001,536-row table occupies 28,800,138,240 bytes. TP4 stores a quarter of
these original packed bytes on each worker: 7,200,034,560 bytes per rank.
No floating-point table copy or second packed bank is retained.

## Placement and dispatch

The loader resolves metadata and host capacity before constructing the model.
All local TP workers agree on admission through their CPU process group.
Admission requires SM70, FP16 embedding output, local TP4, one IQ4_NL table
with complete 160-value rows, and at most four scheduled requests. Available
host memory minus the reserve is divided among local workers. An explicit
host budget is also respected. Existing disk-cascade placement takes
precedence.

The route additionally requires Model Runner V2, dual compilation and FULL
decode CUDA graphs. During decode the prepared PLE layer reads its rank-local
pinned table and the connector submits no CPU lookup request. Dynamic prefill
continues to consume the offload worker's result. Both decisions belong to the
prepared layer; changing the current configuration for the MTP model does not
change that ownership. Dummy capture requests retain the existing semaphore
protocol.

Pinned allocation and checkpoint copying temporarily restrict the loading
thread to allowed CPUs on the GPU's nearest NUMA node. The original affinity
is restored afterwards, including on an allocation or loading exception.
Unknown or restricted topology retains the existing placement.

`KernelConfig.ple_pinned_decode` defaults to enabled. The acceleration report
includes the source family, calibrated M interval, packed bytes per rank and
an explicit rejection reason. The resolved active state participates in the
compilation hash because it changes graph topology; diagnostic status does not.

## Row decoder and correctness

One Triton launch gathers and decodes the requested rows. The original FP16
block scale is reconstructed from its bytes. Each 4-bit index selects the
IQ4_NL integer codebook value; FP32 multiplication followed by FP16 storage
matches official GGUF dequantization followed by FP16 conversion. There is no
activation quantization or extra numerical approximation.

The codebook and pinned mappings are prepared before graph capture. Startup
checks the first, middle and final physical row against official dequantization.
M=1/5/20 tests compare every output value exactly, including negative scales,
zeros and FP16 subnormal scales, then replay graphs with changed indices.
The installed artifact passes 17 admission, ownership and affinity checks,
and three GPU row/graph checks.

## NUMA operator measurements

Measurements use four concurrent V100 workers, CUDA 12.8, Torch 2.10.0 and
FP16 output. Two complete pinned copies permit a direct comparison: all
workers read the first NUMA node, or each GPU pair reads its local node.
Physical page placement is recorded from `numa_maps`. Production retains
only the four quarter-table shards.

The changing-index test uses n-gram rows from eight real generated sequences.
Each captured graph traverses thousands of different lookups rather than
repeating one tiny cached row set. Capture is serial per device, followed by
concurrent replay and an ABBA comparison.

| Tokens | Single-node lookup range | Local-node lookup range |
| --- | --- | --- |
| 1 | 5.219–5.628 µs | 5.199–5.243 µs |
| 5 | 10.335–12.723 µs | 10.294–10.510 µs |
| 20 | 25.362–25.959 µs | 25.369–25.835 µs |

These are lookup service times, not round latency. One early M=5 single-node
sample is slower than the remaining samples. NUMA locality has a small effect
in this measurement; removing CPU request submission and waiting is the main
model-level hypothesis. A fixed-index warm test reached approximately 4 µs
at M=5, but that number is not a cold-row or end-to-end claim.

`benchmarks/kernels/benchmark_gguf_pinned_ple_numa.py` accepts the packed-table
GGUF, metadata GGUF, saved acceptance report and output filename. Run it under
the shared GPU ownership locks. It allocates two full table copies and needs
at least twice the packed table size in available host memory.

## Model measurements

Both arms use the same source-complete wheel built from `69e0780a82`, CUDA
12.8, Torch 2.10.0 and four V100s. Only `ple_pinned_decode` changes. The workload
uses Flash-Next IQ3_S, FP16 MTP4, TP4, FP16 KV cache, FP32 SSM state, FULL target
graphs, max length 9216, prefill budget 512, four scheduled requests, memory
utilization 0.95 and no prefix caching. C1 uses 8192 input and 256 output tokens;
C4 uses 128 input and 600 output tokens per request. Startup and prefill are
excluded from steady decode intervals.

| Metric | CPU-offloaded decode | Pinned decode |
| --- | --- | --- |
| C1 mean round, two unobserved runs pooled | 22.501 ms | 18.615 ms |
| C1 median round, before/after observed run | 22.402 / 22.516 ms | 18.602 / 18.593 ms |
| C1 emitted tokens per steady interval | 4.886 | 4.886 |
| C4 mean round | 53.927 ms | 46.072 ms |
| C4 median round | 53.939 ms | 45.967 ms |
| Eight-prompt mean acceptance | 45.845% | 47.205% |
| Eight-prompt mean accepted length including bonus | 2.834 | 2.888 |

The paired prompt bootstrap gives acceptance difference 95% CI
[-0.368, +2.731] percentage points and accepted-length difference CI
[-0.0147, +0.1092] tokens. These fixed prompts show no observed acceptance
regression; the interval does not prove equivalence for arbitrary workloads.
Both short completion checks finish normally with identical answers.

All 64 teacher points use identical saved prefix token IDs, forced token IDs
and positions. Mean KL is 0.0004528, maximum KL is 0.008371, and top-1 matches
62/64 points. Every logit is finite. Natural long outputs are not token-identical:
first differences occur at tokens 37–396. C1 probe outputs are identical;
C4 outputs differ. Packed rows still match official FP32 dequantization converted
to FP16 exactly. The changed graph topology is not a bitwise model-output promise.

The C1 median is approximately 18.60 ms, above the 18.5 ms stage target. The
measured 3.89 ms C1 and 7.85 ms C4 mean savings are end-to-end gains from this
paired run, not a sum of lookup microbenchmark times. GPU entry skew has not
been remeasured by this run.
