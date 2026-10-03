# IQ3 fragment packet candidate

The installed canonical lattice path retains a grouped prefill gap for
Flash-Next TP4 N=160/K=2560: IQ3_XXS E512/M=8192 takes approximately 562 us,
versus native AWQ 354 us. A CUDA trace confirms the active lattice
CTA32/N128/K32 mainloop with 127 registers and no local memory. Hardware
counters are unavailable; this trace does not establish which instructions
stall. A matched AWQ trace selects CTA8/N256/K64 with 162 registers, no local
memory and about 353 us kernel time. With sixteen rows per expert, the lattice
M32/N128 tiles pad M and use two N tiles, while native M8/N256 tiles use two
M tiles. Source-format dispatch and existing mma884 scheduling remain in use.

This candidate moves each eight-value fragment's two codebook indices, eight
signs and two high index bits into one 32-bit U4 operand packet. The separate
coefficient stream holds one FP16 scale per group of 32. IQ1/IQ2 storage is
unchanged. IQ3's former U2 packet plus 64-bit metadata uses 16 bytes per group;
the candidate uses 18 bytes, a 12.5% increase. It removes wide metadata sign
and index extraction from the common fragment decoder used by GEMM, vector
and canonical dequantization.

Codebook indices, signs and FP16 coefficient bits remain unchanged. Activations
and reconstructed weights remain FP16; GEMM and vector accumulation remain
FP32. All seven CPU carrier/oracle contracts pass (22 cases). Normal extension
compilation, GPU oracle/capture checks and matched real-weight timing are
pending. The candidate is not a default-selection recommendation until those
measurements establish a useful tradeoff. Existing measured dispatch bands
must be rechecked against the changed physical representation.

The candidate also registers the existing native U4 grouped tile geometry
(CTA8/N256/K64) for IQ3. Measurement outside capture decides whether to reuse
it; there is no expert-count/shape threshold or new environment variable.
This addresses the observed padding mismatch within TurboMind rather than
adding a separate scheduling path.
