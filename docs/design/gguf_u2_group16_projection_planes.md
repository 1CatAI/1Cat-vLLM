# Q2_K projection planes for SM70

Q2_K uses two-bit integer codes and a separate scale/min pair for every
16 weights. The small-M projection kernel previously lacked this group16
reader, so mixed GDN inputs and the Q2_K down projection retained their
earlier canonical or original-record implementation.

The new reader uses the existing QPN shared-activation main loop. Loading
reorders the canonical integer codes and FP16 coefficient bits into coalesced
planes. Each K32 step loads two 32-bit code words; eight scale/min words cover
a K128 group. Operand reconstruction uses the same half2 affine FMA as
canonical TurboMind, followed by FP32 MMA accumulation.

Codes occupy two bits per weight and coefficients another two bits per
weight. This equals the existing Q2_K canonical storage size. The plane bank
replaces that storage. Other M values restore the original packed codes and
coefficient bits into existing shared scratch before calling canonical
operators; a second permanent weight bank is unnecessary.

Admission uses `kernel_config.sm70_gguf.u2_group16_planes`, enabled by
default, and requires the packaged `gguf_dmv_u2_group16_sm70_supported`
operator. The measured M8 candidates use KW4/TN2/split1:

| Projection | K | Local output widths | Source types |
| --- | ---: | --- | --- |
| Down | 4352 | 5120 | Q2_K |
| GDN qkvz | 5120 | 2560, 1536 | IQ3_S, Q2_K |
| GDN qkvz | 5120 | 2560, 1536 | IQ3_XXS, Q2_K |

Mixed lattice operands retain the original-record reader's single final
FP16 rounding: the grid times the local scale is exactly representable, then
the original block coefficient is applied. Changing the K schedule does not
introduce an extra FP16 rounding of the combined lattice coefficient.

GDN beta/alpha FP16 weights share the input launch. Adjacent source rows may
be coalesced without changing their output order. Unmeasured shapes,
three-format attention inputs, and TP2 shapes retain their existing paths.
Startup admission reports the missing operator, disabled policy, or unmeasured
shape/source combination. Both TP sizes use the same implementation and
capability checks.

Initial research measurements used real rank0 TP4 weight shards on
V100-SXM2-16GB, SM clock 1530 MHz, memory clock 877 MHz, CUDA 12.8,
Torch 2.10.0, FP16 inputs, CUDA graph replay, rotating more than 64 MB of
weights, and same-process ABBA ordering:

| Projection | Earlier route (µs) | Group16 reader (µs) | Relative L2 vs official dequantization |
| --- | ---: | ---: | ---: |
| Q2_K down | 27.830 | 17.062 | 5.90e-4 |
| IQ3_S + Q2_K qkvz and beta/alpha | 28.870 | 21.379 | 4.84e-4 |
| IQ3_XXS + Q2_K qkvz and beta/alpha | 30.612 | 20.866 | 4.90e-4 |

These prototype measurements used canonical rounded lattice coefficients.
The mixed-source production implementation retains the original-record
operand precision and requires a new comparison. The numbers do not
establish installed-wheel or end-to-end gains. Promotion requires a normal
native build, exact canonical restoration, changed-input graph tests, and
same-wheel model distribution comparisons. Complete-round latency must be
measured separately from isolated projection time.
