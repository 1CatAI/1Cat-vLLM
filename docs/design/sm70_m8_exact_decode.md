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

`build_sm70_exact_decode_screen.py` builds a private research module. It must
not be installed as a serving overlay. A normal source-complete wheel,
1K/8K C1 and C4 admission are pending; this change remains Draft until then.
The production dispatch adds no environment flag or extra weight allocation.

The focused graph test compares bundled M8 against the unchanged unbundled
kernel and also checks non-M8 and alternate-chain fallbacks.

Raw samples, compiler logs, source archive and GPU snapshot are retained in
the `qwen38-qpn2-effective-scale-20261008/exact-*` artifact collection.
Integration base: `47600e948cca28f528e0ae523f19899ab547a0a3`.
