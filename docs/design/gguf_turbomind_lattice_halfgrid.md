# GGUF lattice grouped GEMM tile measurements

## Rejected FP16 shared-codebook expansion

IQ3_XXS and IQ3_S codebooks were expanded into FP16 during CTA initialization,
then read as aligned 64-bit rows. Every value converts exactly: IQ3_XXS has
1024 entries in [4,62], IQ3_S has 2048 entries in [1,15]. Shared storage doubles
to 2/4 KiB. The experiment preserves FP16 activations, canonical scales and
FP32 accumulation, and passes 31 lattice/prefill GPU checks.

Measurements use actual Flash-Next TP4 tensors, four distinct experts,
V100-SXM2-32GB, CUDA 12.8 and Torch 2.10.0+cu128. Graph timing uses 100 ms
warmup and 20 iterations, matching the previous byte-table sweep.

| IQ3_XXS grouped M | Byte table us | FP16 table us | AWQ us |
| --- | --- | --- | --- |
| 1 | 21.11 | 22.22 | 15.28 |
| 2 | 20.33 | 19.50 | 15.38 |
| 4 | 20.38 | 19.35 | 16.15 |
| 8 | 20.49 | 19.48 | 18.42 |
| 16 | 23.97 | 21.77 | 16.29 |
| 32 | 24.72 | 22.56 | 17.44 |
| 64 | 37.61 | 32.84 | 21.71 |
| 128 | 28.15 | 28.42 | 28.12 |
| 512 | 36.97 | 37.93 | 28.20 |
| 2048 | 94.59 | 96.77 | 71.87 |
| 8192 | 342.03 | 354.67 | 228.72 |

M=16–64 improves about 9–13%, while large M regresses slightly. IQ3_S dense
M=8192 changes from 1173.04 to 1131.62 us, while its calibrated canonical DQ
route costs 774.55 us. This does not resolve the grouped gap, so the byte-table
decoder remains the default. Hardware counters are unavailable; increased
shared-table traffic is a hypothesis rather than a measured explanation.

## Smaller accumulator tiles

Static resource usage for the existing IQ3_XXS grouped CTA128/N128/K32 kernel
reports 255 registers with an 80-byte stack frame; FP16 table expansion still
uses 255 registers and raises the stack frame to 96 bytes. These static stack
sizes are not spill measurements or runtime attribution.

The next candidate uses CTA64/N128/K64 with eight warps for narrow IQ2_XS and
IQ3_XXS grouped projections. It reduces output accumulator count per thread.
K64 provides enough group32 metadata rows for the block's operand loader.
Candidate admission is limited to N<=256 and M>=512. Existing schedules cover
other descriptors. GPU correctness, resource usage and timing are pending.
