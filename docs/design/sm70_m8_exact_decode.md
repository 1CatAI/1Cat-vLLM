# Exact M8 gate/up and TP4 norm scheduling

For native bundled QPN2 M8/K5120/N8704 gate/up with split eight and one
accumulator chain, each warp computes both projections and reuses its A
registers. It preserves dequantization, partial sums, FP16 gate/up rounding,
SiLU and FP16 multiplication. Other shapes, layouts and configurations retain
their existing kernels. The normal operator selects this route automatically.

TP4 M8 Gemma RMSNorm retains its forty CTAs and five CUB128 partials per row.
Each CTA publishes its variance and generation atomically in an aligned
64-bit packet, computes the ordered five-part sum and writes its own output.
There is no leader inverse handoff or partial reset. The local IPC pointer
is a separate kernel argument, avoiding a thread-local RankData array copy.
Peer payload, sum order, residual and FP16 output semantics are unchanged.
The normal topology/shape admission remains unchanged.

Packet metadata occupies a separate 512-byte region per rank. Existing IPC
payload offsets are preserved, and benchmark-reference metadata does not alias
the new variance packets. The ordinary buffer initializer clears this region.
The first packet generation is one; all CTAs read the generation before their
partial can be observed by the row's generation owner.

## Validation status

The independent screen compiles the production translation units, calling
the actual gated operator and norm kernel rather than a rewritten copy.
Real unsloth layer-0 weights, TP4 V100-SXM2-32GB fully connected by NV2,
300 W, 1290/877 MHz, Torch 2.10/cu128 and CUDA 12.8 are used. CUDA graphs
evict 128 MiB before the external timing events. All four ranks pass output,
residual, FP32 rollback-state and convolution-history bit checks at input
amplitudes 0.01, 0.125, 1 and 4.

The maximum rank per paired sample improves from 159.442 to 155.986 us:
3.456 us saving, bootstrap 95% interval 3.011 to 3.891 us (200 samples,
seed 123). Ten timed compute nodes remain in both arms. This is approximately
0.166 ms across 48 identical GDN layers; it is not an end-to-end speed claim.
The earlier separate projection screen overestimated the complete-layer gain.

`build_sm70_exact_decode_screen.py` builds a private research module. It is
not installed as a serving overlay. A complete wheel was built with CUDA 12.8,
Torch 2.10/cu128 and GCC 12. Its native libraries load in a fresh runtime
without a preload or private library override. The wheel SHA256 is
`33eb061b73f149d4d92ad1543bfece1a51b9c608903931320133cc61b57c3960`.
The production dispatch adds no environment flag or extra weight allocation.

## Complete artifact admission

Unprofiled TP4 V100-SXM2-32GB at 300 W and 1530/877 MHz uses the original
unsloth NVFP4 checkpoint, original FP8 target head and DFlash2 with seven draft
tokens. The shipped profile starts with max length 262144, max batched tokens
8192, max sequences four and E4M3 KV. Eight fixed prompts each generate 600
speed-fixture tokens at temperature 0.7, top-p 0.9, top-k 20, seed 123 and
thinking off. Only the speed fixture ignores EOS; three separate natural
requests finish normally with `stop` and nonempty output.

| Input | Original artifact, fresh process | Candidate | Saved |
| --- | ---: | ---: | ---: |
| 1K | 13.586 ms/round | 13.456 ms/round | 0.130 ms |
| 8K | 13.994 ms/round | 13.860 ms/round | 0.134 ms |

The prior qualified original artifact measured 13.620/14.015 ms. Use the
fresh-process comparison above for the conservative increment. This does not
meet the 12-ms whole-round objective.

Mean accepted tokens per round and bootstrap prompt-level 95% intervals are
3.0996 [2.9304, 3.2689] versus 2.9828 [2.7037, 3.2275] at 1K, and 2.9200
[2.8405, 3.0102] versus 2.9237 [2.8427, 3.0158] at 8K. Both intervals overlap;
sampled token-sequence equality is not the quality criterion. Device counters
record 1.22%/2.18% reference-branch rounds over the measured 1K/8K requests.

Initial C4 rates in separate processes were 448.91 versus 447.52 tokens/s.
This comparison alone did not pass the nonregression gate. A same-artifact,
same-service follow-up switches only the shipped old norm reference and new
norm protocol, recaptures the model and warms each C4 shape before measurement.
The respective two-run rates are 442.90/429.91 and 450.26/453.51 tokens/s,
means 436.40 and 451.88. The follow-up passes nonregression; its 3.55% rate
difference is not claimed as a norm speedup. The M32 paired-kernel fallback
remains unchanged. Untimed critical-rank CUDA-event calibration gives target
M8 graph means 10.427 versus 10.289 ms, and M32 means 18.396 versus 18.371 ms;
these are graph envelopes, not whole-round latency.

A read-only graph census confirms 56 paired gate/up and 122 packet norm calls
per target M8 graph. Its kernel count stays 606 to 606. The six observed steady
graph components contain 775 executed kernels on a compact round or 810 on a
reference round; eager launches are excluded from those counts. Do not compare
this partial graph census with a full profiler kernel count.

`benchmark_sm70_tp4_norm_packets.py` checks actual FP16 model norm weights
and their FP32 equivalents against the shipped ordered reference using four
input amplitudes, four repeated graph replays per amplitude and alternating
metadata generations. Normalized outputs and FP32 residuals are bitwise on
all ranks. It uses only the clean wheel's ordinary operators. At 1290/877 MHz,
critical-rank 64-call bursts measure 8.565 to 7.304 us for FP16 weights and
8.623 to 7.349 us for FP32 weights. These are isolated protocol measurements.

The focused graph test compares bundled M8 against the unchanged unbundled
kernel and also checks non-M8 and alternate-chain fallbacks.

Raw samples, compiler logs, source archive and GPU snapshot are retained in
the `qwen38-qpn2-effective-scale-20261008/exact-*` artifact collection.
Integration base: `47600e948cca28f528e0ae523f19899ab547a0a3`.
