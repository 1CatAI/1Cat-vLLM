# Short-context FP16 grouped verification

At q8/B1, the K64 plan leaves only 16–32 active CTAs across 80 V100 SMs for
1K–2K context. Use K32 partitions for device lengths 129..2048. Producer and
ordered FP32 merge select the same partition count during graph replay, including
when context grows across the boundary. Other query counts, batches and contexts
keep K64. FP16 inputs and FP32 probabilities/PV/numerator/max/sum are unchanged.

`KernelConfig.sm70_fp16_grouped_short_splits` defaults to true. Its false value
retains the previous plan in the same artifact. The native revision query avoids
passing a new argument to an older binary; startup reports the missing capability.

## Prototype screening

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, SM 1290 MHz / memory 877 MHz,
page 832, q8, local Q/KV heads 6/1, D256, strided FP16 KV and sixteen cold banks.
ABBA CUDA graph replay uses external events. Prototype compiler flags match the
normal FA2 fast-math build. These operator measurements do not establish a model
speedup or an installed-wheel result.

| Live context | K64 us | K32 us |
| ---: | ---: | ---: |
| 640 | 41.950 | 28.203 |
| 1024 | 42.258 | 29.769 |
| 1091 | 43.099 | 30.979 |
| 1536 | 43.550 | 34.903 |
| 2048 | 44.540 | 37.097 |
| 4096 | 53.173 | 55.732 |

The slower 4096 point is excluded. Keeping K64 above 2048 retains the original
partition/reduction layout. The FP32 reference relative L2 stays below 0.000214
at tested activation amplitudes 0.25, 1 and 4. Graph replay is stable and device
zero lengths return zero.

Splitting heads into three interleaved two-head CTAs improves 1091 from 43.021 to
29.061 us, but regresses 8K from 84.954 to 93.026 and 32K from 221.763 to 309.841.
It is not admitted. This retains the negative result so head duplication is not
mistaken for a universally faster plan.

Normal-wheel tests and the same-artifact model comparison remain pending.
