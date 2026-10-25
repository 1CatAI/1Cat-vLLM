# PP transport diagnostics

Set `pp_transport_metrics_interval` to a positive integer in
`--additional-config` to log cumulative counters per pipeline rank every N
nonempty forward batches. Zero (the default) disables hidden-state metrics.
For example, preserve other deployment options and add:

```json
{"pp_transport_metrics_interval": 256}
```

`PP_TRANSPORT_METRICS` reports decode-static hits, the first scheduler condition
causing a fallback, tensor-message counts, and payload bytes by direction and
protocol. Static byte counts exclude local graph-padding rows. Dynamic counts
include the tensors actually submitted; neither count includes serialized
metadata, NCCL framing, or physical network amplification. `legacy_static`
identifies the older fixed-shape transfer independently of the decode option.

`PP_FEEDBACK_METRICS` reports submitted feedback batches, collectives, payload
bytes, packing mode, and queue depth. The packed path submits one collective per
feedback batch; the original path submits two. Submitted work is not a count of
request outputs accepted by the scheduler. A canceled generation can discard
received feedback after its transport has already occurred.

`Worker.get_pp_transport_metrics()` returns the latest counters through the
existing engine collective-RPC interface. It adds no public HTTP route. The
ordinary service does not need developer mode to emit periodic logs. Snapshot
reads and counting inspect tensor metadata only; they do not synchronize CUDA.
The host send-wait counter measures CPU time spent in `Work.wait()`, which may
enqueue stream dependencies; it is **not** CUDA transfer latency or proof of a
communication bottleneck. GPU overlap analysis requires a separate trace.

For a transport comparison, test all four combinations of
`pp_decode_static_transfer` and `pp_packed_feedback`: 00, 10, 01, and 11. Keep
model, GPU order, layer partition, MTP settings, queue depth, metrics interval,
and token-ID inputs identical. Prime every fresh process before measuring;
report cold prefill separately. Include multiple warm repeats, prefix hits,
MTP acceptance, preemptions, temperatures/clocks, TTFT, end-to-end wall time, and
the common interval during which all concurrent streams are producing tokens.
Do not compare a selected steady interval with the entire common interval.

Before timing, run the [real NCCL lifecycle regressions](../../tests/v1/worker/pp_nccl_lifecycle.md).
The forced-reset tests additionally verify that worker request generations are
retired before delayed feedback is consumed, including requests that only
resume in a later batch. Ordinary preemption retains its prior output ordering.
