# IQ3 fragment packet candidate

The installed canonical lattice path retains a grouped prefill gap for
Flash-Next TP4 N=160/K=2560: IQ3_XXS E512/M=8192 takes approximately 562 us,
versus native AWQ 354 us. A CUDA trace confirms the active lattice
CTA32/N128/K32 mainloop with 127 registers and no local memory. Hardware
counters are unavailable; this trace does not establish which instructions
stall. A matched AWQ trace selects CTA8/N256/K64 with 162 registers, no local
memory and about 353 us kernel time. The higher register count in the faster
AWQ kernel means register count alone does not explain the gap. With sixteen
rows per expert, the lattice
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

For the primary Flash-Next checkpoint's first-shard linear IQ3 tensors,
excluding the token embedding, the header inventory contains 22,113,157,120
IQ3_S and 10,066,329,600 IQ3_XXS values. Uniform TP4 slicing adds an estimated
502,804,480 bytes (about 480 MiB) of packed weights and coefficients per rank
relative to the previous canonical representation. This is a storage estimate,
not a measured model memory peak; PLE and runtime workspaces are excluded.

The first GPU attempt aborted in IQ3 GEMM while IQ2 checks passed. The widened
packet's second low-index extraction lacked its byte mask, so sign bits entered
the codebook address. The extraction now masks both low indices before adding
high bits. CPU packet inversion had already masked both bytes; GPU oracle and
capture checks must pass after rebuilding this correction. No timing from the
failed attempt is used.

## Matched packet measurements and rejection

After the byte-mask correction, all 40 lattice GEMM, grouped, canonical
DQ, vector, graph and tracing checks pass. Matched real-weight timing uses
FP16 activations/FP32 accumulation, V100 32GB, CUDA 12.8, Torch 2.10.0+cu128,
TP4 N=160/K=2560, 100 ms warmup and 100 graph timing iterations.

| Experts | M | Existing U2 GEMM us | Candidate U4 GEMM us | Candidate vector us | AWQ us |
| --- | --- | --- | --- | --- | --- |
| 4 | 8 | 22.52 | 29.78 | 16.50 | 18.38 |
| 4 | 16 | 24.02 | 29.85 | 27.76 | 16.25 |
| 4 | 8192 | 300.29 | 305.42 | 17854.88 | 227.80 |
| 512 | 128 | 138.50 | 126.24 | 77.96 | 72.19 |
| 512 | 512 | 306.76 | 287.65 | 266.77 | 186.66 |
| 512 | 8192 | 558.37 | 561.97 | 3973.79 | 352.96 |

IQ3_S dense N=1536/K=2560/M=8192 fused GEMM costs 1136.12 us; canonical
DQ plus FP32 cuBLAS still costs 770.53 us. The packet does not close the
large-M grouped gap and regresses small grouped descriptors despite increasing
storage. The U4 packet representation is rejected; the final candidate restores
all existing U2/metadata layouts and decoder formulas.

One timing launch initially imported the installed baseline wheel, identified
by its two-bit carrier report. Those measurements are excluded as candidate
evidence and retained only as the explicitly labeled installed-base control
for the table above; both control and candidate use 100 graph iterations. Source timing was repeated with the owned tree explicitly selected.
Benchmark output now includes the loaded core SHA256 so artifact identity is
checkable alongside carrier width. The passing source GPU oracle suite was
retained; only the invalid timing launch was repeated.

## Isolated native grouped tile

The remaining operator candidate registers CTA8/N256/K64 for IQ3 while keeping
existing canonical weights, metadata and decoders. It is admitted only for
grouped descriptors with M at least 512, where the shared framework measures
candidate tactics outside capture. Small-M and dense descriptors retain their
existing candidates. No new expert-count/shape condition or environment variable
is added. This isolated candidate requires new GPU and timing checks; the packet
results above do not establish a speedup for it.
