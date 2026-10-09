# Lossless compact candidate communication on SM70

DFlash2 candidate extraction gathers values and token IDs separately. Packing
both into one int32 message removes a collective while preserving the original
FP16/FP32 value bits, int32-range vocabulary IDs and each caller's existing
TopK implementation. The unpacked IDs remain int64.

`KernelConfig.sm70_packed_topk_gather` controls the shared GGUF/NVFP4 route.
Modules capture this policy during construction: candidate sampling runs outside
an active model configuration context. A late global configuration lookup would
ignore an explicit disable. The captured selection dictionary reports both
successful admission and fallback reasons.

Admission covers SM70, TP2/TP4, rows 1/7/8/28/32, 20 or 64 local candidates,
contiguous FP16/FP32 values and int64 vocabulary IDs. Other shapes retain separate
gathers. No activation, accumulation or KV precision changes.

## Validation

Source `aebeadc65d` builds a normal wheel with unchanged qualified native
extensions. Five targeted GPU tests cover original float bits, distinct NaN
payloads, signed zero, subnormals, sentinel IDs, unsupported layouts and explicit
policy use outside a configuration context. Twenty distributed graph cases
cover TP2/TP4, the admitted row/candidate pairs and both value dtypes. Candidate
transport and global TopK results are bitwise equal, including repeated replay.
Captured NCCL graphs are destroyed before communicator shutdown.

Measurements use four V100-SXM2-16GB GPUs with full NV2 connectivity, CUDA 12.8,
Torch 2.10.0+cu128 and sampled SM/memory clocks 1530/877 MHz. ABBA graph measurements
report the maximum TP rank, with twelve samples per setting.

| TP | Rows/candidates | Value dtype | Separate gathers, us | Packed gather, us |
| --- | --- | --- | ---: | ---: |
| 4 | 8/64 | FP32 | 28.40 | 20.23 |
| 4 | 7/20 | FP16 | 25.52 | 18.31 |
| 2 | 8/64 | FP32 | 22.39 | 16.06 |
| 2 | 7/20 | FP16 | 20.99 | 15.27 |

Run the installed operator benchmark under the agreed GPU locks:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  benchmarks/kernels/benchmark_sm70_packed_topk_gather.py --output result.json
```

The complete-wheel model check uses IQ3_S target weights, a Q8_0 DFlash2 draft,
seven speculative tokens, TP4, FP16 activations, FP32 GDN state, target E4M3 KV,
draft FP16 KV, CUDA Graph, maximum length 32768 and prefill budget 1024. Sixteen
matched prompt probes contain 128 matched logit rows: mean KL 6.31e-14, maximum
KL 4.80e-12 and top-1 agreement 100%. Both natural EOS checks pass.

A short 1K-input/128-output diagnostic measures 14.297 ms/round disabled and
14.270 enabled. Steady emitted tokens/round differ, 3.60 versus 4.18; the short
request is not the sixteen-prompt speed cohort. The isolated collective gain
must not be extrapolated into a multi-millisecond model improvement. Complete
TP4 verification remains above the 12 ms target.
