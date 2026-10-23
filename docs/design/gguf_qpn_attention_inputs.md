# Joint GGUF attention input projections

TP4 27B full attention has M8/K5120 and Q/K/V output widths 3072/256/256.
Mixed canonical projections require forty-one matrix launches over sixteen
layers. The measured shared-A QPN body now processes the three sources in
one launch, with per-source readers and ordered FP32 same-launch reduction.
Q2_K and Q4_K alias existing canonical code/stat views and their full stats
stride. Other formats reuse original-record readers. No decoded-weight cache,
activation copy, concatenation or separate reduction launch is added.

## Admission and fallback

Default admission requires SM70, FP16 activations, this exact M8/K5120/N3584
shape and one of fourteen measured logical Q/K/V type combinations. Prefix,
source widths, source storage and available canonical fallback are checked at
load time. An opaque runtime-M operator preserves the original single or
mixed canonical call for other M; compile-time prefill does not fix its route.
FP32 source coefficient products and accumulation remain unchanged.

## Same-machine results

Real rank0 TP4 weights, V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128,
Python 3.12.14, 300W, 1290MHz SM and 877MHz memory. Every graph timing follows
16MiB L2 eviction, with canonical/candidate/candidate/canonical arms together.
The candidate is the installed, prefill-first compiled model-wiring call.
Read bytes are actual operand stream sizes, not measured NCU DRAM traffic.

The corresponding projection in the NVFP4 checkpoint is channel FP8: one
QPN call reads 18357248B and takes 31.744–34.816us at the same shape and clock.
No speedup is attributed to differing checkpoint weights.

| Q/K/V type IDs | Layers | Canonical us | QPN us | Read bytes/rank | Effective GB/s | Estimated saving us/round |
| --- | ---: | --- | --- | ---: | ---: | --- |
| 16/21/21 | 1 | 68.608–69.120 | 38.912–38.912 | 5181440 | 133.2 | 29.696–30.208 |
| 10/23/21 | 2 | 92.160–92.160 | 37.888–37.888 | 9123840 | 240.8 | 108.544–108.544 |
| 10/22/18 | 1 | 97.280–98.304 | 37.888–37.888 | 8785920 | 231.9 | 59.392–60.416 |
| 10/12/21 | 1 | 87.040–87.040 | 35.840–35.840 | 9246720 | 258.0 | 51.200–51.200 |
| 10/18/21 | 1 | 93.184–93.184 | 36.864–37.888 | 8929280 | 238.9 | 55.296–56.320 |
| 21/12/12 | 1 | 61.440–61.440 | 37.888–37.888 | 8396800 | 221.6 | 23.552–23.552 |
| 23/23/12 | 1 | 59.392–59.392 | 45.056–45.056 | 9871360 | 219.1 | 14.336–14.336 |
| 10/23/12 | 1 | 83.968–83.968 | 38.912–38.912 | 9379840 | 241.1 | 45.056–45.056 |
| 21/23/12 | 1 | 89.088–89.088 | 38.912–38.912 | 8273920 | 212.6 | 50.176–50.176 |
| 21/23/23 | 1 | 66.560–67.584 | 38.912–38.912 | 8151040 | 209.5 | 27.648–28.672 |
| 23/12/12 | 1 | 58.368–58.896 | 44.032–44.032 | 9994240 | 227.0 | 14.336–14.864 |
| 18/23/12 | 1 | 87.040–87.040 | 39.936–39.936 | 7536640 | 188.7 | 47.104–47.104 |
| 18/12/12 | 2 | 61.440–61.440 | 39.936–39.936 | 7659520 | 191.8 | 43.008–43.008 |
| 12/23/21 | 1 | 88.064–88.064 | 39.936–39.936 | 11089920 | 277.7 | 48.128–48.128 |

The layer-weighted saving is **0.617472–0.621584ms per round**,
an operator estimate rather than a model-level measurement.

## Correctness and packaging

All fourteen combinations pass official GGUF FP32 reconstruction for Q, K and
V separately, three seeded inputs and one thousand bitwise-stable CUDA graph
replays with reset counters. Runtime rows 512/8/1/5/16/20/32/8 verify
prefill-first compiled selection, graph/eager equality and bitwise canonical
fallback for other M. Sixteen CPU checks cover calibrated capabilities and
single/mixed canonical boundaries.

The existing joint QKVZ+B/A specialization also passes its compiled/eager,
other-M, oracle and graph replay regression after its launch is templated.
The complete `dev57+g2f0ef058ed` wheel contains both the new input route and
main's output projection route. Installed native hashes match the normal wheel;
no source overlay or private native dependency is used.

The [retained operator record](data/gguf_qpn_attention_inputs_20261006.json)
contains every timing arm, runtime-M and oracle check. Model-level measurement
follows integration of the projection batch.
