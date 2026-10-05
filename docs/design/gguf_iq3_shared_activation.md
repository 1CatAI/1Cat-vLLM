# IQ3_S gated pair with shared activation

An M=8 gated pair previously loaded each activation fragment independently
for every output column and both projections. The N32 experiment stages each
K128 activation slice once per CTA, then reuses it for gate and up. The
existing IQ3_S nibble codebook and operand conversion remain unchanged.

The decoder uses a 16 KiB signed codebook derived from the official MIT
licensed IQ3_S table. It retains the original FP16 block scale and original
small-scale nibbles; it produces the same FP16 MMA operands as official FP32
dequantization followed by FP16 conversion. MMA accumulation and split-K
partials remain FP32. The source-byte permutation stores signed 13-bit
indices without expanding the original weight bytes.

## N32 experiment

- 512 threads: two projections, each with eight K warps.
- Shared activation: `[8 K warps][8 rows][136 half]`; the extra eight half
  elements preserve 16-byte alignment and avoid the unpadded row bank stride.
- Cooperative 16-byte activation loads cover each original input vector once.
  A CTA barrier precedes the two projection consumers and another precedes
  the next overwrite. No register prefetch or pipeline is introduced.
- A union holds the 16 KiB codebook during MMA and the 16 KiB FP32 partials
  afterward. A barrier protects the change of lifetime. With the 17 KiB
  activation tile, static shared memory is 33,792 bytes.
- Preferred shared-memory carveout is 100. Compilation uses 48 registers and
  no spills; the control uses 51 registers and 16 KiB shared memory.
- Research registration is conditional on `GGUF_IQ3_PAIR_RESEARCH`. The
  `staged` argument selects shared activation or the matching decoder with
  original global activation loads. Model dispatch is unchanged.

## Measured result

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, TP4 rank-zero partition,
M=8, N=4352, K=5120. Both IQ3_S projections are actual layer-six weights from
Qwen3.8-27B-GSQ-RCO-IQ3_S. The NVFP4 control uses actual weights from the
same layer of Qwen3.8-27B-QUASAR-NVFP4. SM and memory clocks were
1290/877 MHz; temperature was 36 C. Each CUDA graph point uses a 16 MiB
L2 flush outside the kernel event interval, 12 external event pairs and
seven replays, reporting the median of 84 samples.

| Path | ABBA median, us | Original source bytes | Source GB/s |
| --- | ---: | ---: | ---: |
| Same decoder, original activation loads | 63.488 / 63.488 | 19,148,800 | 301.61 |
| Shared activation, N32 | 50.176 / 50.176 | 19,148,800 | 381.63 |
| Earlier nibble-book pair control | 62.464 | 19,148,800 | 306.56 |
| NVFP4 native pair, split 8 | 51.200 | 25,067,520 | 489.60 |
| NVFP4 native pair, split 16 | 52.224 | 25,067,520 | 480.00 |

All shared-activation outputs were bitwise equal to the retained pair
control. The actual Float and Half weight oracle was bitwise equal to the
official GGUF dequantization. Relative L2 error against the FP32 GEMM
reference was 0.00049515 and maximum absolute error was 0.00390625 for
both activation paths. No activation, operand, or accumulation precision
was reduced.

The complete K128 loop contains 659 static SASS instructions, or
82.375 per K16, versus 625 / 78.125 for the global-activation control.
Global-load instructions in the loop decrease from 22 to 7; shared loads
increase from 32 to 48 and two CTA barriers are added. This is a measured
activation-reuse benefit despite the slightly larger static instruction
count. It does not meet a 40 us target.

The improvement is 12.288 us per gated pair against the earlier nibble-book
control. Applying it only to the eight measured same-type layers gives
0.098304 ms per round before integration costs. The other mixed-format
layers retain their existing implementation until each shape and format
combination is measured. This result is an operator experiment; it does not
establish model-level latency or dispatch behavior.

## N64 with two K partitions

The follow-up gives each CTA two N32 output tiles and half of K. Four K
warps share one activation tile across both projections and both N32 tiles.
The signed book stays live across persistent tasks; a separate partials/
completion-flag union retains the existing last-CTA reduction protocol.
There is no additional reduction launch. `__launch_bounds__(512, 2)` produces
62 registers, zero spills and 41,472 bytes of shared memory. The K128 loop
has 647 static instructions, or 80.875 per K16. The complete persistent-task
backedge includes 866 instructions with the reduction and completion path.

The same session and controls produce:

| Path | Two medians, us | Source GB/s |
| --- | ---: | ---: |
| N32 shared activation | 50.176 / 50.176 | 381.63 |
| N64, grid 80 | 58.368 / 58.368 | 328.07 |
| N64, grid 160 | 53.248 / 53.248 | 359.62 |
| NVFP4 native pair, split 8 and 16 | 51.200 / 51.200 | 489.60 |

The completion counters return to zero after each invocation, including all
84 graph replays. Every output is finite. The changed FP32 reduction tree
has relative L2 difference 0.0000071532 and maximum absolute difference
0.00048828125 against the retained pair. Relative L2 error against the
FP32 GEMM reference is 0.00049519; maximum absolute error remains
0.00390625. Weight operands and accumulation precision are unchanged.

Both N64 configurations are slower than the N32 candidate. The N64 source
is retained only as an experiment; the N32 candidate remains the choice for
the measured shape. No additional GPU investigation is required to reject
this N64 configuration.
