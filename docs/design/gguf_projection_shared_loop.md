# Mixed-format GGUF projection execution on SM70

The current mixed-format pair kernel expands a separate load/decode/MMA loop
for each format. This research screen keeps the shipped reader formulas,
KW4/TN4 tile geometry, FP32 accumulation order, FP16 scale expansion and
SiLU epilogue. Its unchanged original kernel remains the control.
Only a measured whole-chain improvement can justify serving integration.

## Ceiling and numerical contract

The existing diagnostic trace assigns 2.997 ms of service to gate/up and
1.524 ms to qkvz with a/b. This experiment initially covers mixed gate/up
pairs only. It cannot remove all projection costs or predict a multi-ms
model saving. The complete projection payload floor is approximately
3.270 ms versus 7.888 ms profiled service; these sums are not unprofiled
model round times. The previous same-wheel model control remains
14.431/14.804 ms at 1K/8K on 54633. No separate model baseline is run.

The shape is M8/N4352/K5120, TP4, four V100-SXM2-32GB GPUs with full NV2
connectivity, CUDA 12.8 and Torch 2.10. Six real gate/up type pairs in layers
22, 38, 39, 42, 36 and 47 cover IQ3_XXS/IQ3_S, both IQ3/IQ4 orders and
both IQ3_S/Q4_K orders. Six independent layers exceed L2 and vary the
instruction stream. All shared GPU locks are held during testing.

## Rejected common-loop variant

Moving the format dispatch from the complete body into its load and decode
helpers reduces static code size but increases dynamic work. The six-layer
same-run ABBA chain changes from **255.754 to 273.474 us**, a 6.93%
regression. Four ranks pass bitwise comparison, official-dequantization
checks and twenty changed-input graph replays. Post-timing SM clocks are
1447–1470 MHz and memory clocks 877 MHz; full clock logs and timing windows
are retained. These are microbenchmarks, not an accepted model speed result.

For the IQ3_XXS/IQ3_S pair, SASS code size falls from 47,744 to 43,392 bytes.
NCU observes 7,294,768 versus 7,824,488 warp instructions, with 97/98
registers per thread. The no-instruction stalled-warps/issued-warps ratio
falls from 0.960 to 0.411, but the long-scoreboard ratio rises from 0.272
to 1.907. Less instruction footprint does not establish a net speedup.
The candidate is not admitted to model A/B.

The NCU probe runs with cache flushing and unmodified clocks. Its measured
GPC frequencies differ (1.237/1.303 GHz), so its kernel durations are not
used as a matched speed comparison. The instruction counts and stall
measurements describe this probe; they do not prove that instruction cache
is the only model bottleneck. Root-owned NCU successfully reads counters
without changing driver settings.

The [retained record](data/gguf_common_loop_rejected_54633_20261008.json)
includes all paired chain samples, numerical checks, code footprints and
counter units. No serving source or production wheel changes in this screen.

## Related reasoning

The [Volta whitepaper](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf)
places an L0 instruction cache in each SM processing partition.
[NVIDIA's instruction-cache investigation](https://developer.nvidia.com/blog/improving-gpu-performance-by-reducing-instruction-cache-misses-2/)
explains why static code size alone does not measure hot instruction-cache
pressure and why eliminating fetch stalls can still lose performance through
other resource costs. This screen uses those diagnostics rather than assuming
that merging instruction bodies guarantees a speedup. No external kernel
code is imported.

## Unified IQ3 metadata screen

IQ3_S and IQ3_XXS share an IQ3-family loop by padding IQ3_XXS metadata
with a zero high-index word. The original block coefficient is unchanged.
Its 0.25 factor is still applied in FP32 before the expanded coefficient is
rounded to FP16; there is no extra scale rounding. Codebook offset and the
coefficient multiplier are invariant across the K loop. Existing decoder
operations are mechanically reused.

One IQ3_XXS N4352/K5120 matrix gains 696,320 bytes of metadata. Code bytes
are unchanged. The two real opposite-order layers 22 and 38 pass bitwise
checks, official-dequantization comparisons and twenty changed-input graph
replays on four ranks. Their two-layer chain changes from 78.654 to 75.507 us,
with post-timing clocks at 1522/877 MHz. The 3.147 us saving across two layers
is a microbenchmark result. Only seven target MLP layers use these opposite
IQ3 pair combinations, so the implied MLP increment is approximately 0.011 ms
before model effects. It is not a multi-ms model gain and no serving route is
admitted on that evidence.

The [IQ3-family record](data/gguf_iq3_family_54633_20261008.json) retains the
paired samples and numerical checks. A subsequent packet-loop screen tests
whether sharing the four K32 decoder steps can reduce hot code footprint;
it must preserve rounding and accumulator order and avoid dynamically
indexed register arrays or spills.

## Rejected fixed-packet loop

Sharing the four K32 steps uses a fixed decoder packet and advances its
index/sign/scale words in registers. This avoids dynamically indexed register
arrays. The compiler reserves 79 registers per thread and reports zero spills,
compared with 97 registers in the original mixed IQ3 pair. Numerical and
changed-input graph checks remain bitwise on all four ranks.

Nevertheless the two-layer chain regresses from 78.389 to 82.916 us. Its
post-timing clocks are 1522–1530/877 MHz. Smaller code and lower register
reservation do not establish a winning execution structure. This version is
also withheld from serving integration; no model A/B or latency credit is
assigned. The [negative packet record](data/gguf_iq3_packet_loop_rejected_54633_20261008.json)
retains every paired sample. Further tile/warp/split variations of these
screens are not planned.

The packet-loop counter probe nearly eliminates the no-instruction ratio
(1.025 to 0.028), but increases executed warp instructions from 7,294,768
to 9,493,480, or 30.1%. This establishes a concrete tradeoff: removing fetch
stalls by looping the decoder adds too much dynamic work. It does not justify
more packet-loop or tile tuning. The counter probe requests NCU's base clock
control; recorded GPC-cycle frequencies still differ, so only the unprofiled
ABBA chain is used for the speed decision.
