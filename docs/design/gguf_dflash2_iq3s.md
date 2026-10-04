# GGUF target and DFlash2 on SM70

Load Qwen3.8-27B GSQ-RCO IQ3_S as the target and the Q8_0 DFlash2 GGUF as
the default draft. Preserve FP16 projection operands, FP32 accumulation and
the existing BF16 range-preserving draft arithmetic. Shared embedding and
LM-head weights remain target-owned. The complete verification round includes
target forward, rejection sampling and draft forward; its single-request goal
is below 12 ms on four fully connected V100s with TP4.

## Loading boundary

The native DFlash metadata adapter recovers five layers, 32/8 attention heads
of dimension 128, a non-causal 2048-token sliding window, an eight-token
trained block, convolution and selector dimensions, and the mask token ID.
GGUF extraction layers are 1-based; convert them to HF indices
`[5, 19, 33, 47, 61]`. Declare BF16 in the config so existing SM70 range
preservation and output scaling remain active under FP16 runtime transport.

Map all 81 checkpoint tensors directly to the draft loader. Retain canonical
quantization for the backbone and context projection. Decode the convolution
projections and selector tables into the dense parameters those modules own.
Keep Qwen3 norm weights unchanged. The SM70 context projection uses TP4
output sharding with compact all-gather for GGUF as well as dense drafts.
Dequantization already allocates a new array; avoid a second dense copy before
dtype conversion, especially for the target embedding table.

Name and format references: llama.cpp `conversion/qwen.py` and
`gguf/tensor_mapping.py` at `bed0a856606ee4a24a164066f73d2379447033f5`
(MIT). No llama.cpp kernel shape tuning is part of this work.

## Checkpoints

| Checkpoint | Bytes | SHA256 |
| --- | --- | --- |
| Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf | 11771546784 | `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810` |
| Qwen3.8-27B-DFlash2-Q8_0.gguf | 2056414816 | `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e` |

Both complete files pass SHA256 verification. The target ModelScope revision
is `fc62412a020e0c1ee3a8f9a12f10aea2102478f0`; the draft Hugging Face revision
is `2d9571f8ce46e151f61c6499c99dee6079e1d610`. An incomplete HF transfer is
retained as a failed path; the ModelScope mirror has the same draft SHA256.

## Measurement contract

Develop and measure on the same four SM70 V100s, all pairs NVLink connected,
with power limit 300 W. Set PCI bus device ordering and expose GPUs 0–3.
Single-GPU probes hold that GPU's lease. Four-GPU runs hold the aggregate
lease, the shared 0–3 lease and all four individual leases. Preserve other
processes and use separate source, runtime and mutable compiler caches.

Operator probes use actual checkpoint weights and TP4 shapes, M=8 with
M=1–16 coverage, cold L2 and repeated calls inside CUDA graph replay. Report
physical bytes, GB/s, latency delta and error against official FP32 GGUF
dequantization. Layer graphs cover full GDN, attention, draft and head tails.
Use full-model runs at integration boundaries or after at least about 1 ms of
predicted savings. Final runs use maximum length 262144, 1K/8K inputs,
temperature 0.7 and thinking disabled.

The kernel ledger records calls per round, physical bytes, the 750 GB/s
bandwidth floor, measured service and launch grids. Project savings as latency
delta times calls, distinguishing overlap and using the measured profiler
correction rather than treating service sums as wall time. Investigate
prediction errors above 15%. Compare Q8_0 and BF16 drafts with at least eight
seeded prompts and 600 emitted tokens per prompt, reporting mean tokens per
round and confidence intervals.

Numerical changes require teacher-forced mean KL ≤0.001, p99 ≤0.01,
maximum ≤0.05, top-1 agreement ≥99% and maximum logit difference ≤0.5.
Validate rejection samples against the dense reference token by token. Check
fixed prompts against llama.cpp, the fixed quality suite including 128K and
258K needles, and C4 before promotion. No additional precision reduction is
authorized. Source checks pass; GPU loading, graph, numerical and model-speed
results remain pending.

## Installed loading checks and storage budget

The installed normal wheel passes 45 CPU checks on Python 3.12.14,
Torch 2.10.0+cu128 and CUDA 12.8. Seven changed modules match the source,
wheel members and installed files exactly; all sixteen native libraries match
the qualified normal base wheel. The release profile accepts local GGUF draft
files. Packed context projections expose their declared operand dtype to
auxiliary-state conversion without changing the existing runtime transport.
These are package and CPU checks; they do not establish GPU model quality or
throughput.

The actual target has 40 mixed gate/up layers and 35 mixed GDN qkv/z layers.
There are 72 distinct checkpoint role/type/TP4-shape combinations across the
target and draft. Predicting storage from the current canonical code and
metadata streams gives:

| Storage | Per-rank bytes |
| --- | --- |
| Target checkpoint body excluding embedding/head | 2659536384 |
| Canonical target projection buffers | 3175956480 |
| Checkpoint head, one read | 178790400 |
| Canonical head, one read | 198656000 |

The canonical body estimate is about 19% larger than checkpoint storage.
Expanded coefficients and index/sign metadata therefore need to be included
in bandwidth accounting. IQ3_S grows from 0.4296875 to 0.5 bytes per weight;
IQ2_XS grows from 0.2890625 to 0.5. These are CPU format predictions for
aligned projections, excluding norms, codebook loads, inputs/outputs and
workspace traffic. Verify them against the loaded GPU buffers before using
them in a bandwidth or complete-round speed claim.
