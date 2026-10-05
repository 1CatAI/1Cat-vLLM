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

Serving admission did not establish a benefit. Keep this candidate out of the
default runtime; correctness alone does not qualify it for promotion.

## Serving admission

The source-complete wheel at commit `81cec988650a95e5bc8a2de763c02bbbf3b2cb14`
was built once with CUDA 12.8, Torch 2.10.0+cu128 and Python 3.12.3. It passed
the SM70 release artifact check, 20 compiler routing tests and the four-rank
interleaved CUDA graph replay test. Hinted and original collective outputs
were bitwise equal.

The serving comparison used four V100-SXM2-16GB GPUs, TP4, full NVLink,
300 W limits, the original Qwen3.8-27B NVFP4 checkpoint with its FP8 head,
FP16 DFlash2 draft7, FP8 KV, maximum length 262144, batch-token limit 8192,
maximum concurrency four and memory utilization 0.92. Both arms ran in one
service with recaptured graphs. Each input length used eight fixed-seed
prompts with 600 output tokens, temperature 0.7, top-p 0.9, top-k 20 and
thinking disabled. Serving step timing excludes the first 20 streaming steps.
C4 ran after shape warmup. Observed C1 SM clocks were 1312 MHz on rank 0 and
1530 MHz on ranks 1-3, with 877 MHz memory clocks in both arms. Device process
monitoring sampled every 200 ms and recorded no foreign CUDA process during
the isolated comparison.

| Measurement | Original | Weight hints | Change |
| --- | ---: | ---: | ---: |
| 1K mean step | 14.5713 ms | 14.6281 ms | +0.39% |
| 8K mean step | 14.9655 ms | 15.0280 ms | +0.42% |
| 1K tokens/step | 2.9540 | 2.9540 | unchanged |
| 8K tokens/step | 2.9408 | 2.9408 | unchanged |
| Warm C4 steady throughput | 418.29 tokens/s | 415.23 tokens/s | -0.73% |
| C4 throughput including prefill | 344.77 tokens/s | 349.78 tokens/s | +1.45% |

Prompt-level paired bootstrap intervals (20,000 draws) for the increase in
step time were [0.0491, 0.0645] ms at 1K and [0.0523, 0.0735] ms at 8K.
These intervals describe prompt variation, not service-order or clock drift.
The tokens/step 95% intervals were identical between arms: [2.7885, 3.1287]
at 1K and [2.7756, 3.0755] at 8K. Reference-path fractions also matched:
23/1637 steps (1.405%) at 1K and 26/1644 steps (1.582%) at 8K. C4 was a single
warmed batch per arm, sufficient for the concurrency smoke but not for
establishing a sub-percent throughput difference.

All four ranks captured 55 hinted AR+norm nodes per target graph. Kernel counts
were unchanged: 606 in the target graph and 142 in the draft query graph.
These counts exclude eager launches and rejection graph branches; they are
not a complete verify-step census. All three natural-EOS smoke prompts in each
arm finished normally. The earlier independent whole-layer screen also failed
to establish a speedup, consistent with rejecting this candidate for default
promotion.
