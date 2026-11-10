# GGUF LM-head projection on SM70

Use the existing canonical Q4_K projection for the measured vocabulary-head
shape on SM70, with FP16 operands and FP32 accumulation. The choice uses the
actual row count inside an opaque graph operator. Embeddings retain their
existing storage and lookup method. The raw head remains available outside
the measured M=2–16 band, including M=1 where it is faster.

## Operator evidence

Qwen3.8-27B GSQ-RCO IQ3_S has Q4_K `output.weight`; TP4 gives N=62080 and
K=5120 per rank. Measurements use real rank-0 weights on a 300 W V100,
CUDA 12.8, Torch 2.10.0+cu128 and the normal SM70 wheel at source
`7fd188f4a23a0db8c03e9ac66b0391f4bb51eaa3`. Each of twelve graph calls is
preceded by a 16 MiB L2 eviction buffer, with seven measured replays and
external CUDA events. Bandwidth is weight footprint divided by latency;
it is not a hardware DRAM-counter measurement.

| M | Canonical µs | Raw µs | Canonical relative L2 | Raw relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 336.384 | 243.712 | 0.0006912 | 0.0056942 |
| 2 | 265.216 | 475.136 | 0.0007272 | 0.0054112 |
| 4 | 268.288 | 321.536 | 0.0007450 | 0.0144431 |
| 7 | 270.336 | 619.520 | 0.0007474 | 0.0113763 |
| 8 | 272.384 | 621.568 | 0.0007174 | 0.0161201 |
| 16 | 314.368 | 1246.208 | 0.0007022 | 0.0107231 |

At M=8 the canonical streams occupy 198656000 bytes and reach an effective
729.32 GB/s, versus 178790400 bytes and 287.64 GB/s for raw Q4_K. Against
official FP32 GGUF dequantization followed by FP32 cuBLAS with TF32 disabled,
maximum absolute error is 0.004103 for canonical and 0.064516 for raw.
Both projection outputs are FP16. These sampled activations do not establish
model-level KL or acceptance equivalence.

The pre-change graph ledger contains two raw head calls at about 597 µs each.
Using the canonical microbenchmark projects about 0.65 ms less head service
per round; applying the provisional 0.9 profiler correction gives about
0.58 ms. Comparing both standalone paths gives 0.698 ms. Neither estimate
accounts for stream overlap, and neither is a measured model speedup.

## Graph and dispatch checks

A real-weight Inductor graph exercises M=8, 2, 16, 1 and 32. Canonical runs
inside the admitted band and raw runs outside it. Each compiled and captured
result matches the selected existing operator exactly. CPU checks cover the
separate embedding/head methods and preservation of runtime row-count dispatch.
Unsupported hardware, activation dtype, format or unmeasured head shape keeps
the raw method and records the admission reason. No additional numerical
precision reduction, native kernel or environment variable is introduced.

Model measurements follow the next integration boundary after floating GGUF
projections are restored to the shared dense loading path.
