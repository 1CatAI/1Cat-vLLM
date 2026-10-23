# Decode ordered GGUF PLE rows directly

Packed PLE row gathering already returns request order, including repeated
IDs and mixed device/host/disk placement. General embedding decoding then
performed a second row gather using an identity index. Expose the existing
row decoder as an opaque operation and use it directly for ordered packets.
Arbitrary embedding lookup retains its original index selection. No decoder,
weight format, activation dtype or precision changes.

The legacy decoder labels its output dimensions in the opposite order to its
row-major payload. Normalize this view before returning ordered rows; general
embedding lookup previously normalized the view after decoding. An initial
legacy-shape test failed while28 checks passed; the corrected normal artifact
passes all29 checks. Tests cover official Q4_0/Q4_1/Q8_0/Q4_K/Q6_K/IQ4_NL/
IQ4_XS/Q2_0 decoding in FP16 and FP32, changed-input CUDA graph replay,
legacy fallback and device/host/mixed-disk/disk-only PLE row ordering.

Flash-Next uses16 n-gram heads of width640. The graph benchmark alternates
old/new order over eight epochs, captures100 operations per graph and replays
ten times per timing sample, so short GPU operations are not starved by a
Python replay loop. Inputs are IQ4_NL packets with valid scales; outputs match
official decoding bitwise. The normal SM70 artifact uses source `7167af4ba8`,
CUDA12.8 and Torch2.10.0+cu128 on V100-SXM2-32GB.

|Verifier M|Packet rows|Identity gather + decode(us)|Direct decode(us)|Local saving(ms)|
|---|---:|---:|---:|---:|
|5|80|7.242|2.153|0.0051|
|20|320|7.831|3.091|0.0047|

Every controlled epoch favors direct decoding. An earlier graph containing
one operation, replayed1000 times from Python, had scheduling gaps and variable
timings: M5 medians were12.983/14.057us (regression), M20 were13.804/4.698us.
Those results are retained; they did not support a M5 speed claim. The revised
measurement batches GPU work within the graph instead of selecting favorable
samples. These are local estimates, not measured full-model gains.

The preceding full model graph contains1,763 GGUF nodes versus1,527 NVFP4
nodes. This change is expected to remove its single identity scatter/gather
node, leaving1,762; that count has not been measured in a new full-model trace.
No extra model restart is needed for a bitwise identity-copy elimination.
Reproduce using `benchmarks/kernels/benchmark_gguf_ordered_rows.py OUTPUT.json`;
`--profile` captures separate old/new graph replays with CUDA node tracing.

The CUDA-only node capture confirms100 old graph replays contain20,000
kernels (10,000 scatter/gather plus10,000 IQ4_NL decoding). The new100
replays contain10,000 decoding kernels with no scatter/gather. Each graph
has100 repeated operations, so this is exactly one node removed per packet;
no full-model node count or end-to-end speed is inferred from the microtrace.
