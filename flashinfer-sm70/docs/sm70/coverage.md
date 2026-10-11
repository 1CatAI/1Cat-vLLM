# Qwen4 path coverage and admission ledger

Audit baseline: `8d4fd9401ce0fd713821be0f045dc1f1f718b57b`.
Implementation base: `be3722bc095b9792bf9833068595f0b1ef70ad37` on `onecat/main`.
The increment is PR #1172's mHC prenorm/output optimization; it is retained.
Canonical dirty checkouts and other worktrees are not development inputs.

This is a **source audit / implementation ledger**, not a completed execution
audit. For every row, native registration, loaded artifact identity, actual
GPU kernel, numeric correctness and complete-chain timing must be closed on the
target machine. GPU validation is in progress on the supplied V100 machine;
the earlier user-requested hardware deferral has ended.

Paths below are relative to the engine repository, except `flashinfer.*` APIs.
Both target and draft traverse the relevant rows. Production baselines exclude
disabled experiments. Passing a primitive or delegating to an existing engine
kernel does not close a row.

| Chain | Engine call / native owner at implementation base | Shape / state constraints | Fork status and upstream design reference |
| --- | --- | --- | --- |
| RMS/Gemma norm, residual norm, activation, position-ID RoPE | model layers → `flashinfer` public dispatch → `flashinfer.sm70.ops` → TVM-FFI `basics` | FP16, FP32 reduction, explicit output; no BF16/FP8 Tensor Core or PDL | Independent native implementation / upstream norm and activation reuse; 16 GPU cases pass on V100/CUDA 12.8; performance pending |
| QSA projection, Q/K norm, rotary, compressed/raw-key updates | `nvidia/indexer_qsa.py`, `nvidia/ops/qsa.py` | H4/D128, compression 4, original calibration and ring layout | Pending extraction; upstream `experimental/b12x/attention/qsa` plan/bind/run contracts inform lifecycle, its BF16/CuTe kernels cannot run on SM70 |
| QSA score | `ops/qsa.py:qsa_mqa_paged` → `model_executor/kernels/flashinfer_sm70.py` → `flashinfer.sm70.QsaScorePlan` → TVM-FFI `qsa_scores` → `group_qsa_rows` / `mqa_multiquery` | Candidate FP16 M1..32 H4/D128; arbitrary request membership, at most 32 requests, int32/int64 positions, existing paged strides | Independent Volta MMA candidate, default off; C1 M1..8 retains one launch, C4×5 shares each request's keys; 18 standalone GPU cases pass, including three fixed cancellation cases. The `.2` partial chain improves C4/M20/8K and 32K, but 128K/M20 stopped on score tolerance; `.3` engine revalidation and complete-chain gates remain pending |
| QSA exact Top-K and expansion | `ops/qsa.py:qsa_select_paged_tokens` → `csrc/qsa_lexicographic_topk.cuh` / Triton expansion | Exact score order, lower logical ID on ties, sorted selected IDs; visible-length mask | Preserve #1171; migration and complete-chain qualification pending |
| QSA sparse attention / output gate | `nvidia/qsa.py`, `ops/device_kv_attention.py` → `_sm70_qsa_device_C` or existing host/paged route | Main D256; qualified device-history M1..20 H6; target vs draft placement differs | Pending; generic FlashInfer attention is not a substitute for QSA admission |
| HC mix, norm, projection, combine, TP boundary | `nvidia/hyperconnection.py`, `sm70_hcx.py`, `model_executor/kernels/mhc/triton.py` | HC4, hidden 2560, rank 320; TP4 and topology constraints | Pending; retain #1172 and distinguish M5 from M20; upstream b12x HC is BF16-specific |
| Shared/routed MoE, gate/Top-10, merge | `nvidia/model.py`, quantization layers, `csrc/sm70_turbomind` | 512 experts, Top-10, existing GGUF / NVFP4 layouts and TurboMind packing | Pending independent native ownership; no requantization or Marlin substitution |
| GDN convolution/gating, prefill, decode, verification, state selection | `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` and existing SM70 kernels | FP32 recurrent state, variable valid rows, MTP accept/reject and rollback | Pending; upstream GDN SM90+/SM100 replay cannot be directly enabled |
| PLE n-gram, decoded row conversion, short convolution | `nvidia/ple.py` and existing PLE helpers | GPU hash already optimized; device/pinned/disk rows, existing encoded storage | Pending compute extraction; row scheduling and file I/O remain engine-owned |
| MTP projections / QSA reuse / rollback | `nvidia/mtp.py`, `v1/worker/gpu/spec_decode/eagle/speculator.py` | FP16/FP8 draft routes; reuse currently still updates indexer state | Pending; removing preprocessing requires proof that partial accept/all reject restore all state |
| LM head, sampling, exact reductions | existing model LM head / sampler and `_sm70_sampler_C`, `_sm70_exact_reduce_C` | Full vocabulary, unchanged sampling semantics and TP groups | Pending; operator ownership must not move sampling policy into the library |
| Vision patch projection, positions, norm, QKV | `Qwen4ExpForConditionalGeneration` → `model_executor/models/qwen3_vl.py`, `qwen2_5_vl.py` | 27 layers, hidden 1152, 16 heads, **D72**, patch16/temporal2, output2560 | Pending; local full NVFP4 checkpoint has visual weights; language-only deployment is not a visual baseline |
| Vision ragged noncausal attention, MLP, merger | `layers/attention/mm_encoder_attention.py`, `v1/attention/ops/vit_attn_wrappers.py` | Image/video mixed lengths, original `1/sqrt(72)` scale, mask and TP layout | Pending dedicated SM70 route; existing H3 D128 probe does not qualify D72 |

## Known negative results to preserve

- `docs/design/sm70_flashinfer_native_backend.md`: old planner/primitive probes
  and fallback wrappers are not native all-path acceleration.
- Draft #523: prior GDN/QSA batch work has adverse quality evidence and unmet
  whole-model targets. This standalone source module is a different ownership
  change; do not import its measurements as acceptance or duplicate its probes.
- #1168 / commit `764afb73937d4c5d53436aa55dc6d69138627f9d`: historical SGLang
  comparison localizes multi-request scoring, but uses a different Torch/runtime,
  synthetic data and TP-local single GPU. It is not the current C1/C4 baseline.
- `docs/design/sm70_flashnext_hc_boundary.md`: full HC fusion won at M5 and lost
  at M20; per-CTA polling and several split-K/MoE scheduling candidates lost.
- Existing GDN verify is off: prior complete-model C4 latency/output regressions
  prevent automatic selection. Existing fused convolution/gating and GPU PLE
  n-gram wins must not be rediscovered or counted twice.
- MTP currently reuses Top-K while maintaining caches. Moving the reuse guard
  ahead of indexer updates is unsafe without new rollback evidence.

## Frozen validation contract

Primary: Python 3.12, Torch 2.10, CUDA 12.8, V100-SXM2. Freeze GPU set/topology,
TP/PP/DCP, quantization, effective compute and KV storage precision/placement,
cache capacity, context/input/output lengths, sampling, MTP, prefix caching,
attention backend and graph mode. Model families include current GGUF and
ModelOpt NVFP4 target, FP16/FP8 supported drafts, image/video, and Qwen3.6-27B /
35B-A3B AWQ/FP8 production regressions.

QSA matrix: C1/C4, ordinary decode and MTP4 verification, mixed prefill,
8K/32K/128K and existing supported long-context boundaries; actual page layouts.
State matrix: empty/padded rows, changing valid lengths, multiple captures,
replay, cache replacement, request end/reuse, prefixes, partial acceptance and
all rejection. Exact integer outputs/ties/state choice cannot use numeric
tolerances. Floating tolerances inherit existing reference tests; do not widen
them. Include NaN/extreme inputs and noncontiguous/quantized layouts.

First establish stable A/A outputs and timings. Then paired alternating repeats
must show **at least 5% reduction** for each required complete chain and integrated
text C1/C4 actual output-token time / multimodal total latency, beyond noise.
Other critical points must regress no more than 2%. Report chain time, TTFT,
pure decode, MTP round time, actual output-token time and acceptance separately.
Summed kernel times are not an end-to-end result. Visual tests need real weights,
mixed resolutions/lengths and videos, encoder and overall TTFT, and final outputs.

No automatic admission entries exist. Foundation, QSA, HC/MoE, GDN, PLE/MTP,
vision and full integration retain separate test gates within the single
consolidated PR #1177, as requested by the user. Retire an
engine-native duplicate only after the new owner passes its equivalence and
performance gates. The required text/MTP/vision rows must all close before
announcing “Qwen4 全路径打通”.
