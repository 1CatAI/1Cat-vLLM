# Predictive GGUF expert preparation on SM70

`kernel_config.sm70_gguf.expert_cache_prefill_prepare` defaults to `all`.
The opt-in `predict` mode retains TurboMind's grouped GEMMs and original expert
placement. It prepares a previous pass's active experts ahead of computation,
then converts every current-route miss before either projection consumes its
canonical bank. Predictions never omit a routed expert or change arithmetic.

Correctness admission covers eager FP16 contiguous prefill inputs with
4096–8192 query rows per invocation, at most512 experts, SM70, and the ordinary
DMA preparation route. Context length is distinct from these batch-row units.
Smaller/larger batches, graph capture and other hardware retain complete-bank
preparation. `predict` requires `expert_cache_prefill=turbomind`; it does not
alter MMQ, decode or the configured GPU/host expert fraction.

The existing native lane filters DMA from its locked residency snapshot,
preserving host-slot offsets. Only missed cold experts use mapped-host reads;
resident misses read their ordinary GPU slots. Conversion, grouped GEMM,
weighted unroute and their FP32 accumulation remain unchanged. Original expert
bytes and request/state semantics retain their exact contracts.

Each expert cache owns two pinned hint snapshots and separate GPU prediction,
observation and miss plans. An unfinished CPU snapshot is never read. A new
layer-zero pass retires speculative readers before replacing the GPU hint;
interrupted passes and arbitrary prediction errors use the same exact miss
completion. Cache teardown drains mapped reads and pinned transfers before
unregistering host storage. No extra host thread, private native provider or
process-wide math policy is introduced.

The first pass prepares all experts. Later predictions are performance hints
from completed passes, so cache history and request mix can affect speed but
cannot affect expert values. Prefix-cache clearing does not need to invalidate
these nonsemantic hints.

The additional `predict_resident` candidate admits completed hints into the
same LRU slot budget after each layer's raw readers finish. Hints larger than
the resident capacity skip admission. The existing planner chooses slots and
write-backs; bounded DMA saves incoming bytes before any aliased host-slot
write-back, and the ordinary GPU remap commits the plan. One expert-shard
temporary and one pinned plan per layer are owned by the existing preparation
object. No private provider or host polling thread is added.

Layer-zero boundaries retire prior admission and prior GPU-published decode
work before CPU LRU planning. An external CUDA event makes an already captured
decode graph wait for the latest eager admission and speculative raw readers,
including interrupted prefill. A separate owner stream joins those readers
without delaying admission copies. This synchronization cost is part of the candidate and must be
included in performance evidence. Admission is never submitted by the decode
lane that answers a GPU poll. The original decode copy protocol is retained.
This candidate has component/state qualification; it is not a promoted default.

The measured acceleration case is Flash-Next abliterated IQ3_S, TP4 on four
16GB V100-SXM2 GPUs with1312MHz application-clock settings, CUDA12.8 and
Torch2.10.0+cu128. Actual measured SM clocks are recorded separately and can
reach1530MHz. Its serving
contract retains262144-token context, chunk8192,80% GPU/20% host experts,
host QSA KV, disk PLE, MTP4, prefix caching, tool calling and thinking, with
temperature0.6/top_p0.95/top_k20. Four-rank microbenchmarks use captured router
IDs, real expert weights and synthetic activations/mixture weights. They do
not qualify model throughput or long-context quality. Full-model qualification
is recorded in the performance worklog before any default promotion.

Targeted contracts:

```bash
python -m pytest -q tests/config/test_gguf_prefill_prepare.py
python -m pytest -q tests/models/qwen4_exp/test_gguf_expert_cache.py \
  -k 'cold_stage or predictive_prefill_thresholds or dma_prefill or resident_prefill_fences'
```
