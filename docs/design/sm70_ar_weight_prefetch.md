# SM70 AR+norm weight-read candidate

The TP4 push all-reduce can issue L2 hints for the next native NVFP4 gate/up
after publishing its peer packets. The compiler connects the weight tensor to
the collective when it finds an N8704/K5120 native QPN2 gated consumer with
eight K partitions. Runtime checks restrict the hints to eight FP16 rows, a
FP32 residual, SM70 and an active full-mesh TP4 push communicator. Other shapes
retain the existing collective and norm path.

The hints cover the first group of each existing K partition. They do not
change the arithmetic, weight storage or kernel count. No additional runtime
environment switch is introduced.

This branch is an end-to-end admission candidate. Its earlier independent
whole-layer screen did not establish a speedup, so correctness alone does not
qualify it for promotion. Before merging, compare the original and hinted
routes in the same source-complete runtime with 256K maximum context, 1K/8K
single-request inputs and a warmed concurrency-four run. Record verify-step
latency, tokens per step, reference-sampling fraction and graph kernel counts.
Confirm compiler matches and captured kernel dispatch, and retain both the
timing result and the bitwise collective/replay checks. Enable the route in the
integration branch only after the serving comparison passes.
