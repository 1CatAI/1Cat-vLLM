# Native SM70 preparation for host QSA history

Host history previously rejected the existing native QSA preparation route.
The owner therefore split query/key normalization and rotary preparation into
separate operations before calling its history encoder. Reuse the packaged
SM70 preparation operator with local FP16 staging, then invoke the unchanged
host history writer. Keep the query gate in its original strided head layout.

Admission uses the existing `sm70_qsa_prep` policy, SM70 native-operator
availability, FP16 activations, D256, one KV head and partial NeoX rotary
dimensions divisible by 16. Host admission additionally requires a text-only
target; host draft behavior remains unchanged. Batches above 32 rows retain
the previous path. Record the admission and reason in kernel selections;
there is no new environment switch or weight representation.

Allocate 32 local int64 staging slots when binding the cache. These slots
refer only to temporary FP16 key/value rows. They are unrelated to request
pages, authoritative history slots or hot-cache ownership. Prepare only active
rows, including when a graph has extra padding. The existing writer receives
its original physical slot map and performs the same encoding, scale update
and hot-cache write. Calibrators receive prepared keys, rather than raw QKV.

The temporary key/value payload is 5 KiB at M5 and 20 KiB at M20. Query output
remains FP16; normalization and rotary arithmetic retain the existing native
operator's precision. No GGUF-specific decoder or new CUDA kernel is added.

Eighteen CPU tests pass, including contracts for active counts, multidimensional text positions,
fixed local slots, distinct output allocations and rejected empty/oversized
staging calls. Same-wheel CUDA graph screens use real Flash-Next Q/K norm
weights, M5/M20, partial rotary dimension 64 and FP16/FP32 cosine caches.
Two changed-input/position replays preserve key/value values, gate values,
E4M3 history codes and FP32 history scales exactly. M5 query results are also
bitwise equal; M20 query maximum absolute error is 0.00012207, relative L2
5.42e-7.

With the representative FP16 cosine cache, complete preparation plus E4M3
history encoding measures 17.408 / 13.312 us at M5 and 18.432 / 13.312 us at
M20. The approximately 0.049-ms service estimate across 12 target QSA layers
is not a model-round improvement. FP32-cache reference timing includes
explicit activation casts and must not replace the FP16 model denominator.

Fresh packaged route checks and matched C1/C4 model gates remain pending.
The policy remains opt-in and this document claims no endpoint improvement.

## Host partial launch policy

For the admitted SM70 D256/H6 host route, retain two-warp partials through
32 rows. The previous common device policy stops at 16 rows; other device
backends, dimensions, heads and larger batches retain their profiles. Tiles,
split count, decoder and FP32 reduction are unchanged.

Separate-state graph ABBA tests use four distinct physical requests with five
verification rows each. At M20 and contexts 128/512/1024, four-warp control
measures 67.968/100.096/176.896 us for complete host resolution plus attention;
two-warp measures 67.008/89.024/110.592 us. At 8192 tokens per request the pair
is 342.400/255.104 us. A deliberately undersized 64-token hot cache at context
128 measures 453.568/451.456 us. Outputs are bitwise equal in every case,
including changed selections/gates and invalid rows. These synthetic layer
screens do not establish a C4 endpoint improvement.

The two-warp partial uses 226 registers and 24 KiB shared memory, versus
165 registers and 24 KiB for four warps; neither spills. More warps consume
more total registers per CTA and do not improve this small host workload.
