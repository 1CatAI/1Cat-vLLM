# TP4 integration findings (2026-09-15)

## What exists
- 1Cat-vLLM-sm70 (branch main, tip e95e350): the sm70 work is already
  IN THIS FORK, not in the plugin. Commits c5293fd8c (RoutedExperts
  backport) + e95e35081 (Exl3MoEMethod E2E test).
- vllm-exl3 plugin (references/vllm-exl3): NOT installed in the env.
  Its src/vllm_exl3/exl3.py provides Exl3Config + Exl3MoEMethod +
  the expert-apply plumbing. Written for GB10/sm_121 TP=1.
- exllamav3 (our sm70 tree): the env's exllamav3 already points at
  /home/nvidia/Dev/exllamav3-sm70 (editable). exl3_moe exists but
  is arch-gated cc>=8 (correct — the fused kernel is sm80+).

## E2E verified on V100
Exl3MoEMethod E2E (tests/exl3_sm70_moe_e2e.py) PASSES with:
  PYTHONPATH=<1cat-vllm>:<vllm-exl3>/src  EXL3_FUSED_MOE=0
  → loop backend, rel err 7.3e-4.
The fused path (exl3_moe) correctly rejects cc<8; the loop backend
(LinearEXL3 per-expert) is the working sm70 path today.

## TP=4 gap analysis
The plugin's TP plumbing (_narrow_tp, _resolve_tp_geometry) shards
gate/up column-wise and down row-wise with an all-reduce — the
standard MoE TP pattern. It is size-agnostic (narrow + contiguous),
so the sharding math itself is TP=4-ready. What is missing for TP=4:
1. The E2E test only exercises tp_size=1. The sharding path is
   untested at tp>1 (expert-map + all-reduce interaction).
2. The loop backend runs one LinearEXL3 per (token, expert) pair —
   correct under TP but each rank reconstructs its own shard; the
   all-reduce is handled by RoutedExperts/moe_output layer.
3. Our sm70 GEMV kernels (exllamav3_ext) are per-matrix and
   shape-agnostic — they work on sharded tensors unchanged.

## Next steps for TP=4
1. Extend the E2E test to tp_size=4 (4-rank or simulated 1-rank
   with tp_size=4 shard shapes) to validate the sharding math.
2. Serve the real model: DeepseekV4ForCausalLM exists in this fork
   (vllm/models/deepseek_v4/nvidia/model.py) and the registry maps
   our arch. The exl3 quant config (quant_method: exl3) needs the
   plugin's Exl3Config registered — install the plugin
   (pip install -e references/vllm-exl3) or add the entry point.
3. vllm serve with --tensor-parallel-size 4.


## Verified install path (2026-09-15)

- Plugin installed editable: `pip install -e references/vllm-exl3
  --no-deps --no-build-isolation` (required a setup.py fix: relative
  source paths — committed a317eb3 upstream). `VLLM_EXL3_NO_CUDA=1`
  skips the CUDA extension build; the native p2b kernels are
  optional and the loop backend does not need them.
- Verified: `vllm.plugins.load_general_plugins()` registers exl3,
  `get_quantization_config("exl3")` resolves to Exl3Config, and the
  E2E test passes through the registered path (loop backend,
  EXL3_FUSED_MOE=0).
- The E2E test requires `EXL3_FUSED_MOE=0` on V100: the fused
  exl3_moe kernel is sm80+ and the plugin's fused path does not
  arch-gate before calling it (it relies on exl3_moe's own
  TORCH_CHECK). Loop backend is the correct sm70 default.

## TP=4 serving command (ready to test)

    vllm serve /home/nvidia/Dev/model/DeepSeek-V4-Flash-Vision-Exp-exl3-3.04bpw \
        --tensor-parallel-size 4

Open items before a real TP=4 run:
1. The plugin's TP sharding (_narrow_tp) is size-agnostic but only
   exercised at tp_size=1 by the E2E test.
2. Model memory: 110 GiB weights / 4 GPUs = ~27.5 GiB per GPU +
   KV — fits 32 GiB V100s, but KV cache size needs tuning.
3. The loop backend is slow (per-token expert loop); TP=4 e2e
   numbers will be launch-bound until the sm70 fused path lands.

## MTP/DSpark readiness (2026-09-15)

The fork has the full DSpark speculative-decoding path:
- `DSparkDeepseekV4ForCausalLM` (vllm/models/deepseek_v4/nvidia/
  dspark.py) loads the mtp.0/1/2 checkpoint weights as the draft.
- The draft's linear layers take `quant_config=vllm_config.quant_config`
  — with the exl3 plugin registered, the draft's exl3-quantized
  weights resolve through Exl3Config like the target model.
- The DFlash2 proposer (vllm/v1/spec_decode/dflash.py) drives the
  DSpark draft on sm70 — the README's Flash-V100 DFlash2 numbers
  are this path on this hardware class.
- Serving: `--speculative-config method=dspark` (use_dspark() →
  DFlash2 proposer).

Our model: num_nextn_predict_layers=3, mtp_bits=3 (exl3-quantized
MTP weights present — 9447 mtp.* keys in the checkpoint). The draft
is a dense 3-layer model (no MoE), so its exl3 layers are plain
LinearEXL3 served directly by the sm70 GEMV kernels — the same
kernels the E2E-verified loop backend exercises.

Untested: an end-to-end serve with spec decode on V100. The
remaining risk is the draft/target KV and sampler interaction under
the sm70 attention backends, not the quant path.

## TP=4 config parse verified (2026-09-15)

`EngineArgs(model=<DeepSeek-V4-Flash pack>, tensor_parallel_size=4)
.create_engine_config()` succeeds: quant resolves to Exl3Config, TP=4
accepted. Required fixing the plugin's `get_min_capability` (80 → 70):
the sm70 GEMV decode path serves LinearEXL3 on V100; the fused MoE
path still requires Ampere and falls back to the loop backend below
cc 8.0 (EXL3_FUSED_MOE=0 or the plugin's backend selection).

Automatic entry-point loading verified through the engine path:
`EngineArgs.create_engine_config()` → `load_general_plugins()` →
`register_quantization_config("exl3")` → Exl3Config. A bare
`import vllm` does NOT load plugins (by design — plugins load at
engine-config time), so the earlier bare-import probe failing was
expected behavior, not a gap.


## Real-model load test (2026-09-15) — precise gap identified

Attempting the actual TP=4 weight load surfaced the real blocker,
in sequence:

1. `KeyError: 'scale_fmt'` — the dsv4 model reads
   `config.quantization_config["scale_fmt"]`, which exl3 packs
   omit (the value lives in original_quantization_config: ue8m0).
   Trivial fix, reverted pending the larger work.
2. `DeepseekV4 only supports fp8 kv-cache` — needs
   `kv_cache_dtype='fp8'` (the model's DSA attention requires it).
3. **The real gap**: OOM in `UnquantizedFusedMoEMethod.create_weights`.
   The dsv4 nvidia model's MoE layer is `FusedMoE` (model.py:533),
   but `Exl3Config.get_quant_method` only handles `RoutedExperts`
   and `LinearBase` — FusedMoE falls through to the unquantized
   method, which allocates BF16 expert weights (hundreds of GiB).

Closing it means one of:
- (a) adapt `Exl3MoEMethod` to FusedMoE's param contract
  (w13/w2 shard shapes, expert_map, weight_loader) — the trellis
  tensors don't map 1:1 onto FusedMoE's param shapes; or
- (b) switch the dsv4 model's MoE to `RoutedExperts` (backported
  in c5293fd8c for exactly this) — changes the model's
  weight-load and forward paths.

Either is real engineering (days, not config). The verified state:
config parse + quant resolution + TP=4 acceptance all work; the
blocker is precisely the Exl3MoEMethod↔FusedMoE param contract.


## FusedMoE bridge — starting-point analysis (for next session)

Two checks decide adapter-vs-rewrite:

1. **create_weights signature**: compare FusedMoE's call site
   (what kwargs it passes to quant_method.create_weights) against
   Exl3MoEMethod.create_weights' expectations. If the kwargs map,
   the bridge is thin.
2. **Expert-params naming**: model.py's make_expert_params_mapping
   (line 121-127) generates `experts.{id}.{weight_name}.` prefixes;
   the plugin's weight loaders already match this pattern —
   evidence the plugin was designed for this model's FusedMoE
   naming, suggesting the isinstance check (RoutedExperts-only)
   is the actual gap, not the weight-loading machinery.

Validation template: tests/exl3_sm70_moe_e2e.py (loop backend,
EXL3_FUSED_MOE=0, rel err 7.3e-4 baseline).


## FusedMoE bridge landed (2026-09-15) — next gap: per-shard bits

Two plugin fixes committed (vllm-exl3):
1. `get_quant_method` accepts FusedMoE layers (isinstance branch
   extended; create_weights signature-compatible via
   **extra_weight_attrs — audited, bodies are layer-agnostic).
2. Dense linears default to exl3 when the pack carries no
   non_routed_exl3 spec (exllamav3 end-to-end packs quantize all
   linears; the old default crashed weight loading with
   KeyError fused_wqa_wkv.mul1).

Load test progression after the fixes: TP=4 load now reaches
shard-1 weight matching on the fused attention linear and fails
with a REAL shape mismatch:

    EXL3 linear load shape mismatch shard=1 suffix=trellis:
    dest (256, 32, 48) != loaded (256, 32, 80)

k_words 48 = 3bpw (global bits), 80 = 5bpw (head_bits). The fused
fused_wqa_wkv linear merges wq_a (3bpw) and wkv (5bpw) — the
checkpoint uses per-tensor bits (head_bits=5 for attention output
projections), but Exl3LinearMethod sizes ONE param set with the
global bits for all shards.

Next step: per-output-partition bits in Exl3LinearMethod —
k_words per shard (the trellis param is already per-shard sized
via out_tiles_list; bits must become per-shard too). The bits
per shard are derivable from the checkpoint tensor shapes
(loaded k_words / 16) or from a per-layer bits map.


## Per-shard bits — DISPROVED; actual fix: head_bits selection (2026-09-15)

Checkpoint shapes decided it: wq_a.trellis (256,64,80) and
wkv.trellis (256,32,80) — BOTH 80 k_words = 5bpw (head_bits).
Uniform K; the mismatch was bits SELECTION (dense default used
global bits=3). Fixed: head_bits stashed in Exl3Config, selected
for attention/compressor prefixes in the dense-default path.
Committed in vllm-exl3 (head_bits selection + duplicate-kwarg fix).

## Load progression after head_bits fix

Next failure: KeyError 'layers.0.attn.wo_a.slice.0.mul1' — wo_a is
a BATCHED linear (bmm_batch_size = n_local_groups); the checkpoint
stores per-group tensors under wo_a.slice.N.{trellis,suh,svh,mul1}.
Same param-class gap as fused_wqa_wkv: the plugin's flat
single-linear param layout vs the model's batched/merged attention
linears. The per-shard/per-slice param design (recorded below)
applies to both, with slice-index routing in the loader.

## Per-shard bits — scoped design (2026-09-15)

The fused trellis param is one tensor (in_tiles, total_out_tiles,
k_words) with uniform k_words = bits*16. The fused_wqa_wkv layer
merges wq_a (3bpw) and wkv (5bpw = head_bits) — per-shard bits
cannot fit the uniform layout.

Fix design (next session):
- Per-shard trellis params: list of tensors, one per shard, each
  with its own k_words. Changes:
  1. create_weights: allocate per-shard trellis (out_tiles_list
     already exists per shard; k_words per shard from a bits map)
  2. weight loader: dest shape check per shard (already per-shard
     in the loader — the mismatch error proves it)
  3. apply: run the GEMV per shard with each shard's K (the fused
     linear's apply already iterates shards for suh/svh)
- Bits per shard source: the checkpoint tensor shapes (loaded
  k_words / 16) at load time, or a per-layer bits map in the
  config. Deriving from shapes is robust — no config change.
- Alternative rejected: padding shard 0 to 80 k_words with zeros —
  the decoder reads exactly bits*16 words per tile; padding works
  only if apply passes per-shard K to the kernel, which is the
  same per-shard-bits plumbing as option 1 with wasted memory.


## wo_a slice loading — marker stage passed, geometry next (2026-09-15)

Slice-name branch added to the model's load loop (committed):
checkpoint `wo_a.slice.N.{suffix}` names strip to the flat param
(`wo_a.{suffix}`) with the slice index passed as loaded_shard_id —
the plugin's weight_loader already handles marker span (mul1/mcg
apply to all shards) and trellis shard splits.

Progression: the mul1 marker KeyError is gone; the load now fails
at svh geometry — `shard=0 suffix=svh: dest (2048,) != loaded (256,)`.
The checkpoint's per-slice geometry differs from the plugin's
uniform model: slice.0.suh is (4096,) = full input, slice.0.svh is
(1024,) = per-group output. The batched-linear layout (in-sharded
suh, out-per-group svh) needs the loader to understand the slice's
own geometry rather than the layer's uniform shard model.

Remaining work: teach the plugin's linear weight_loader the
batched-linear geometry (per-slice in/out sizes), or model-side
narrowing before the loader call. The checkpoint shapes are the
authority: slice.suh covers the full input; slice.svh covers the
slice's output partition.


## wo_a slice geometry — runtime metadata settled (2026-09-15)

Debug metadata from the failing launch: `n_shards=1 out_parts=[2048]
row=False merged=False tp=4`. The wo_a layer at TP=4: 8 groups / 4
ranks = 2 groups per rank → the rank's param covers 2048 out (2
slices × 1024), allocated as ONE shard (n_shards=1).

Slice count corrected: 8 slices (slice.0..7, one per o_group) —
an earlier listing truncated at 12 keys and showed only 4. The 2×
arithmetic is resolved: constructed n_groups(8) × o_lora_rank(1024)
= 8192 = checkpoint 8 slices × 1024 ✓; per-rank at TP=4 = 2048 =
dest ✓.

The slice mapping: checkpoint slice N belongs to rank N//2, local
index N%2. The model's slice branch must:
1. Skip slices not owned by this rank (N//2 != tp_rank).
2. svh: narrow locally — local slice i covers param[i*1024:(i+1)*1024]
   (the loader's uniform-shard dest math can't address halves of a
   single-shard param).
3. suh: pass through the loader's row-parallel narrowing (suh
   covers the full input; shard_exl3_row narrows by TP).

The loader's span logic (loaded_shard_id tuples) handles spanning
tensors; here the checkpoint ships per-slice tensors so the model
narrows and routes per slice. All arithmetic verified against the
checkpoint shapes (slice svh 1024, slice suh 4096, slice trellis
(256, 64, 80)).


## Batched-layer load: flat param layout confirmed insufficient (2026-09-15)

Empirical close-out of the loader-patch route: with the slice branch
passing full per-slice tensors (verified via debug print — svh
arrives (1024,) complete), the loader's fallback still fails because
the PARAM LAYOUT itself is flat: one suh (1, 4096), one svh (2048),
one trellis — but a batched layer needs per-slice params (2 local
slices × their own suh/svh/trellis). The loader's segment math
cannot address halves of a single-shard param.

Confirmed requirement: per-slice parameter sets in Exl3LinearMethod
(the design recorded earlier — group shards by owning rank, one
trellis/suh/svh set per slice, apply() runs per-slice GEMVs). The
loader-patch route is exhausted; the redesign is the path.

Scope for the redesign:
- create_weights: allocate per-slice params (slices_per_rank × each
  suffix), shaped from the checkpoint's per-slice geometry
  (in 4096, out 1024 per slice for wo_a)
- weight loader: route slice.N → param[N_local], direct write (no
  TP narrowing — verified: all suffixes are rank-local complete)
- apply: per-slice GEMV over each slice's params, concatenated
  output (or batched GEMV — the bmm structure the model's forward
  expects)


## Per-slice suh distinctness — redesign definitively required (2026-09-15)

The last open question: are the slices' suh tensors distinct (each
group has its own input-side scale) or identical (one write +
broadcast)? Checked the checkpoint directly: slice.0.suh ≠ slice.1.suh
(4096-element tensors, values differ). Each of the 8 groups carries
its own suh.

This settles the design question raised by the refined analysis (the
output_partition_sizes derivation fix would address svh/trellis
segment addressing, but suh needs 8 distinct 4096-element slots —
the flat (1, 4096) param cannot hold them regardless of segment
math). The per-slice param redesign is confirmed REQUIRED:

- create_weights: per-slice params — 8 slices → per-rank 2 slices ×
  (trellis (256, 64, 80), suh (4096,), svh (1024,), markers)
- loader: slice.N → param[N_local], direct write per suffix
- apply: per-slice GEMV (the model's bmm forward structure)


## Redesign perf note (for implementation)

The per-slice apply path (8 sequential GEMVs per layer vs one merged
GEMM) costs throughput — the merged-GEMM optimization is why
fused_wqa_wkv exists. Preserve it where possible: group same-K
slices into batched GEMV calls rather than fully sequential
per-slice applies. Correctness first (the notes' design), then this
perf refinement once the load path is green.


## MAJOR MILESTONE: weight loading complete, forward runs (2026-09-15)

The TP=4 load path is GREEN: all weights load (LOAD OK), the model
constructs, and the first forward pass runs — the failure moved to
APPLY time (determine_available_memory's profiling forward), at the
shared_experts gate_up_proj's reconstruct reshape:

    shape '[8192, 2048]' invalid for input of size 33554432
    (reconstruct_hgemm, exllamav3/modules/quant/exl3.py:176)

The shared_experts gate_up geometry: the layer's registered
in_features/out_features don't match the loaded trellis's decoded
shape. The reconstruct's y.view(rows, out_features) fails — the
layer's geometry attrs need checking against the loaded trellis
(in 4096, out 4096 = gate 2048 + up 2048).

Fixes landed this round (vllm-exl3 + 1Cat model.py):
- bmm_prefixes map (Exl3Config.__init__ + from_config skip set +
  model-side registration before construction) → per-slice suh
  allocation for wo_a
- shape-based rank-local write path in the loader (segment math
  from loaded vs param shapes)
- head_bits family extended to shared_experts
- compressor quant threading (quant_config=None → vllm_config.
  quant_config) — the compressor's exl3 params now register
- lm_head quantized (VocabParallelEmbedding branch in
  get_quant_method + quant_config passed to ParallelLMHead)
- tile-aligned padded+TP relaxation (pad % 16 == 0) with zero-init
- span math round() for padded geometry (129280//32384=3 skew)
- gate bias name mapping (ffn.gate.bias → ffn.gate.e_score_
  correction_bias), unconditional noaux_tc bias registration
- vision-side key filter (aligner./image_/vision.) at the
  ForCausalLM level


## Next session's opening item: apply-time reshape at shared_experts

The load path is green; the first forward fails at
reconstruct_hgemm (exllamav3 exl3.py:176): reshape [8192, 2048]
invalid for 33554432-element input. The shared_experts gate_up
layer's registered in/out geometry doesn't match the loaded
trellis's decoded shape (in 4096, out 4096 = gate 2048 + up 2048).
Check the layer's in_features/out_features attrs against the
trellis dims — likely a doubled or swapped geometry from the
create_weights args (MergedColumnParallelLinear output_sizes
[2048, 2048]).


## Apply-time diagnostic plan (next session's opening item)

The failing view: `x.view(rows, self.in_features)` at
reconstruct_hgemm (exllamav3 exl3.py:176). in_features comes from
the LinearEXL3 object built at apply-side construction
(exl3.py:3172-3184, make_linear_exl3 with in_features =
int(suh.numel())).

Diagnostic: print self.in_features and x.shape at the failing call.
The mismatch direction traces to what suh slice the shared_experts
shard got at construction — suspects: (a) merged gate_up suh
handling (gate+up share one suh or have distinct ones), (b) padded
suh derivation.

Carry-forward cleanup items (pre-ship, non-blocking):
- span-math round() fragility: shard counts computed via round() on
  padded sizes — correct for this geometry, fragile for other
  pad/TP combinations; use true (unpadded) sizes when threaded
- vision-key filter duplicates: inner-model filter at 1192-1199 is
  dead code (the ForCausalLM filter at 1476-1478 is operative);
  consolidate at next touch


## Milestone update: engine initializes, prefill runs, decode crashes (2026-09-15 late)

Fixes this round (all verified by test runs):
- batched wo_a apply: group-paired slicing — the apply ran ALL
  slices on ALL groups and concatenated (doubling z width);
  now slice i pairs with group i (dim -2), outputs cat along
  last dim. This fixed the [8192, 2048] reshape crash.
- bmm shard construction: process_weights_after_loading now
  builds _bmm_n linears for bmm layers (trellis out-tile span
  per slice, suh row i, svh span i), markers read at [0]
  (uniform). Previously built n_shards=1 linear → list index
  OOR in the apply.
- compressor_kv_score / indexer_compressor_kv_score: guard
  .weight access — quantized (EXL3) fused_wkv_wgate has no
  .weight; call the layer forward instead (return_bias=False
  → bare tensor).

State: engine initializes, KV cache allocates (2.02 GiB at
0.95 util / 256 ctx / 4 seqs), prefill completes, GENERATE
starts — decode crashes with an ILLEGAL MEMORY ACCESS in
sm70/sparse_kernels.py:919 (sparse paged fp8 decode kernel).

Memory regimes mapped: 0.90 util → KV 0.03 GiB (too small);
0.95 → KV 1.6-2.0 GiB ✓ but decode workspace tight (0.97
OOMs the workspace); the IMA is the current blocker, not
capacity.

Next session's opening item: the IMA in the sparse paged fp8
decode kernel — FIRST rerun with CUDA_LAUNCH_BLOCKING=1 to
confirm the faulting kernel ("crashes on the first decode
step" is exactly the scenario where the fault may be DELAYED
from a prefill-side kernel, reported at the next sync). Once
the faulting kernel is confirmed: check the paged KV index
bounds under max_model_len=256 (4841 KV tokens, 4 seqs) and
the fp8 page table indexing.


## IMA diagnostic plan (carry-forward, next session's opening)

Memory-regime table (util-only is a dead end — it moves the
failure point, never fixes the workspace/KV competition):
- 0.90 util: pool after weights ~1.6 GiB → KV 0.03 GiB, init
  fails
- 0.95 util: KV 1.62-2.02 GiB ✓ but decode workspace OOMs in
  the leftover
- 0.97 util: carves MORE KV → workspace margin worse

Ratio levers (one-line harness changes): max_num_seqs=4
(shrinks num_decode_tokens → workspace linearly), SPLITK_C128
off (drops the fp32 partial buffers entirely), max_model_len
=256 (shrinks num_partials). Shrink the workspace FIRST, then
util has room to move.

The fp32 splitk partial_acc (num_decode_tokens, heads,
num_partials, 512) fp32 × 3 buffers is the allocation that
OOMed — num_partials scales with main_width (SWA window +
compressed context), not the request length.

Discriminator before splitk index-math work: rerun with
CUDA_LAUNCH_BLOCKING=1 — the current traceback (sparse.py:145
forward_mqa → :272 _forward_decode → sparse_kernels.py:919)
may be a DELAYED fault from a prefill-side kernel, reported
at the next sync. If the IMA fires on decode step 1, look one
frame earlier for a prefill-side writer; if after several
clean steps, the splitk decode indexing is the real suspect.

Then: buffer-shape-vs-index-range check — the workspace specs
at sparse.py:212-235 vs the kernel's index math at
sparse_kernels.py:919 (main_width/num_partials derivation
source: layer attribute vs actual index buffer width).


## IMA root-cause narrowing (2026-09-15 late, tuning session)

Discriminator results:
- CUDA_LAUNCH_BLOCKING=1: the IMA reports AT the exl3 GEMV
  launch (exl3_gemv.cu:284, the cooperative launch in
  exl3_gemv_try_launch) — the faulting kernel IS the sm70
  trellis GEMV, NOT the sparse attention kernel (the earlier
  sparse_kernels.py:919 traceback was the next sync point).
- VLLM_EXL3_RECONSTRUCT_MIN_ROWS=1 (GEMV bypassed): the FULL
  pipeline runs — tokens generated end-to-end (2.91 toks/s,
  gibberish text). The GEMV is the sole crash site.
- EXL3_GEMV_SMEM=1: same IMA — not the extraction-style
  variant.
- VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=0: same IMA — not
  the stream-overlap race.
- Standalone sweeps (random + REAL checkpoint tensors, all
  traced shapes incl. out=256/32384, rows 1-2): ALL PASS.
  The IMA needs the full-runtime context.
- compute-sanitizer: blocked — NCCL fails under the
  sanitizer (cudaErrorNoKernelImageForDevice at nccl init),
  the model never loads.

Failing call signature (per-worker last GEMV before the
assert): K=5 in=4096 out=512 mul1 — the wkv shard shape;
but standalone passes with real tensors, so the trigger is
runtime state, not shape alone.

Static bounds verified: C/A/B indexing bounded by size_m/k/n
and the grid cap; the ws partial buffer unused at cfg 0
(KS=1); the locks buffer sized MAX_TILES_C + barriers.

Working config for a green pipeline (reconstruct-only):
VLLM_EXL3_RECONSTRUCT_MIN_ROWS=1 + the 0.95/256/4 harness.
~3 toks/s — the GEMV is ~10x faster per the dispatch
docstring, so fixing the IMA is the perf lever.

Next tool: kernel-level debug asserts (printf the OOB index
+ thread/block) in exl3_gemv_sm70_kernel.cuh, rebuild the
ext, rerun — a build+run cycle. Alternatively cuda-gdb on a
minimal multi-layer repro that still IMAs.


## Correctness finding (tuning session close)

The reconstruct-only pipeline generates tokens but the text
is GIBBERISH (mixed-script garbage, repeated tokens like
'awsze', consistent across runs and prompts). This is a REAL
correctness bug, not an untrained model — the checkpoint is
a quantized production model.

Ruled out: the padded-vocab leak (the LogitsProcessor trims
logits[..., :org_vocab_size] after the all-gather — the pad
rows never reach sampling).

Suspects (in order of tractability):
1. Weight-layout corruption in loading/slicing — a wrong
   tile order or shard span silently produces valid-shaped
   but wrong weights. Check: round-trip test — dequantize
   the loaded trellis via ext.reconstruct, requantize with
   the dsv4 compressor's quantize path, compare error
   magnitude. A transposed tile order shows as a huge
   round-trip error.
2. The batched wo_a group-paired apply (slice i ↔ group i)
   — if the pairing or the group dim is wrong, attention
   output is garbage.
3. The K=3 cb=2 reconstruct path (the K=3 layers dominate
   the GEMV trace).

The IMA work (GEMV kernel) and the correctness bug are
INDEPENDENT tracks: the GEMV IMA blocks the fast decode
path; the correctness bug affects even the reconstruct-only
path. Fix order: correctness FIRST (fast tokens are
worthless if wrong), then the GEMV IMA for throughput.


## Correctness narrowing (round-trip + scale-correlation results)

CORRECTED: the earlier round-trip verdict was wrong. The
quantize_tiles output IS the decoded codebook value in
fixed codebook units (quantize_tiles_kernel.cuh:321,
decode_3inst) — no per-tile scale. The ~0.87/1.0 errors
were a UNIT MISMATCH: outer-domain tiles (de-Hadamarded,
rescaled by suh/svh, std 0.023) fed to the fixed-unit
codebook. The mul1 quantization path requires pre-
normalized input (quantize_rows: rms → cs → scale0 →
normalize → quantize → scale back).

Scale-correlation check (PASSES): the dequantized W's
column norms correlate with |svh| at cosine 1.0000 and row
norms with |suh| at 1.0000. The tile layout is NOT
scrambled — a transposed/shifted tile order would
decorrelate the per-column norms from the output scales.
The weight dequantize path (trellis → tiles → Hadamard →
scales) is self-consistent.

Correctness suspicion shifts to the COMPUTE path:
1. The Hadamard sign/phase convention between quantize-time
   and dequantize/apply-time — a phase flip keeps scale
   correlation but corrupts activations (the correlation
   check cannot catch it).
2. The attention compute graph (the group-paired batched
   apply, the compressor path).
3. The GEMV/reconstruct decode math for K=3/K=5.

Next experiment: apply the dequantized W as a dense fp16
matmul vs the fused path on the same input — if they
disagree, the fused compute (Hadamard convention) is the
bug; if they agree, the attention graph is the suspect.

ROUND-TRIP RESULT (inner domain, decisive): taking
get_inner_weight_tensor() (raw ext.reconstruct output =
codebook units, no had/suh/svh), extracting tiles with the
canonical permute, requantizing with K = trellis.shape[-1]//16
and the layer's mul1 flag: error 0.000000 — EXACT. Codebook
values are exactly representable, so the tile element-order
convention and the encode-decode pairing are VERIFIED
CORRECT. This closes the fused-vs-dense blind spot (both
paths share the same decode; only a true round-trip can
test the tile order).

Correctness conclusion (final for the weight domain): the
quantized weight representation is EXACTLY self-consistent
through quantize → encode → decode → reconstruct →
requantize. The gibberish is definitively NOT in the
quantized weight representation. Remaining suspects: the
RUNTIME Hadamard application to activations (the forward's
had_l/had_r convention vs quantize-time), the attention
compute graph, or model wiring.


## Fused-vs-dense parity (correctness narrowing complete for the linear path)

The fused GEMV and the dequantized-weight dense matmul AGREE
on the same input: relative diff 0.0005 (fp16 noise floor).
The quantized linear compute — Hadamard convention, GEMV
math, reconstruct math, scale application — is CONSISTENT
end-to-end at the single-layer level.

Correctness conclusion: the linear/quantization stack is
CLEARED. The gibberish output originates at the attention
or model-wiring level: the group-paired batched apply's
integration, the compressor path, the sparse attention, the
KV cache, rotary embeddings, or residual/norm wiring.

Next session's correctness plan (in order):
1. Layer-level parity: one decoder layer's forward with exl3
   weights vs the same layer with dequantized bf16 weights
   on identical input — diverge ⇒ the layer's compute graph;
   agree ⇒ higher (model wiring).
2. Hidden-state probe: constant input, track per-layer
   hidden norms — norm explosion/collapse localizes the
   broken component (attention vs MLP vs norm).
3. The attention path suspects in order: the group-paired
   batched apply integration, the compressor's kv-score
   guard change (the quantized fallback path), the sparse
   attention's topk indices.


## First-token probe result (correctness narrowing continues)

Greedy decode (temperature=0) produces garbage from the
FIRST token ('occ S Q'sef intensity ofifiable'). The argmax
token is already wrong — the corruption is in the PREFILL
forward, not the KV cache or decode loop.

Cleared by this session's tests: the quantized linear stack
(weights self-consistent, fused compute parity 0.0005), the
padded-vocab leak (trimmed), the stream race (IMA persists
without overlap).

Prime suspect: the MoE expert weight path — the FusedMoE
bridge's w13/w2 slicing of the exl3 expert trellis. A wrong
expert-weight slicing produces valid-shaped but wrong
expert weights → garbage output from layer 0 (the hash-MoE
layers use the same expert compute). Fits: first-token
garbage, all layers affected equally.

Secondary suspects: the attention compute (group-paired
apply integration, compressor), the hash-MoE + bias routing
combination (the bias is now registered for hash layers 0-2
matching the checkpoint — verify fused_topk_bias handles
the hash+bias combination as the model intends).

Next session's plan:
1. MoE bridge parity: one expert's weight through the
   bridge's slicing vs direct checkpoint dequantize —
   compare.
2. Layer-level probe: hidden-state norms per layer on a
   constant input (norm explosion/collapse localizes).
3. If MoE clears: the attention compute graph.


## MoE w2 slicing verified (correctness narrowing continues)

The w2 trellis param (E, out_tiles=32, in_tiles=256, 48)
matches the narrowed loaded (32, 256, 48) exactly: the
create_weights' tile vars are named for the w13 orientation
(in=hidden, out=intermediate) and the w2 param reuses them
swapped — dim1 = intermediate-local tiles, dim2 = hidden
tiles. shard_exl3_row narrows the checkpoint's dim 0
(intermediate FULL 128 → 32 per rank) ✓. The strict shape
check (dest vs sharded, raising on mismatch) passed during
the load — the w2 slicing is correct.

w13 similarly: shard_exl3_col narrows the OUT dim (128 → 32
per rank) matching (E, 2, 256, 32, 48) ✓.

MoE weight path CLEARED. The first-token garbage lives in
the attention compute or model wiring. Next: the
layer-level parity test (exl3 weights vs dequantized bf16
weights through one decoder layer) and the hidden-state
norm probe — the plan documented above.


## Session close (tuning + correctness narrowing, 2026-09-15)

Additional verification this session:
- The group-paired batched apply's semantic is correct:
  contiguous head sharding makes local group 0 ↔ global
  slice 2r hold, so the pairing is right under the model's
  sharding scheme.

Correctness status: the quantized linear stack (weights,
slicing, fused compute) is fully CLEARED by parity and
consistency tests. The gibberish output lives in the
attention compute or model wiring — the layer-level parity
test and the hidden-state norm probe are the next
experiments (both need forward-hook work inside the vLLM
workers).

Throughput status: reconstruct-only path ~2.2-2.9 toks/s
(correctness-blocked, so tuning is premature). The GEMV IMA
fix would unlock the fast decode path (~10x per-call claim
in the dispatch docstring) — kernel-level debug asserts in
exl3_gemv_sm70_kernel.cuh are the prepared next step
(compute-sanitizer is blocked by NCCL).

All changes committed across the three repos (1Cat-vLLM-sm70,
references/vllm-exl3, exllamav3-sm70).


## mhc staging verified (correctness narrowing continues)

The sm70 mhc prenorm staging (sm70_mhc_prenorm_staging)
compared against a torch reference (split-K GEMM + sqrsum):
max relative error 0.000000 on both outputs — EXACT. The
custom V100 port's staging kernel is correct. (Test
contract: x (tokens, hc_mult*hidden) fp16, fn (mix_hc,
hc_mult*hidden) fp32, outputs (n_splits, tokens, mix_hc)
and (n_splits, tokens) fp32.)

Correctness scoreboard after this round:
- Weight representation: EXACT (round-trip 0.0)
- Linear fused compute: parity 0.0005 ✓
- MoE w13/w2 slicing: shape-verified ✓
- mhc prenorm staging: exact ✓
- Vocab trim: verified ✓

Remaining suspects for the first-token garbage: the mhc
sinkhorn/mixing shared code, the attention compute (the
group-paired apply integration, the compressor, the sparse
attention topk, the rotary), or model wiring. The next
experiment is the forward-hook harness: per-layer hidden
norms on a constant input (norm explosion/collapse
localizes the broken component), then the layer-level
parity test.


## mhc post verified (correctness narrowing continues)

sm70_mhc_post vs mhc_post_torch reference on identical
input: max relative error 0.00021 (fp16 noise floor) — the
comb indexing convention ([input, output] with the
einsum '...ij,...ih->...jh' pairing) matches the reference
exactly. The mhc port's both custom kernels (prenorm
staging, post) are verified.

Correctness scoreboard: weight representation EXACT,
linear compute parity ✓, MoE slicing ✓, mhc prenorm
staging EXACT, mhc post consistent ✓, vocab trim ✓.

Remaining untested: the sinkhorn mixing (shared code), the
attention compute (group-paired apply integration,
compressor, sparse attention topk, rotary), model wiring.
The forward-hook harness (per-layer hidden norms on
constant input) is the next experiment — it localizes the
broken component in one run.


## mhc fused pipeline verified (correctness narrowing continues)

The production fused path (mhc_fused_post_pre_tilelang —
the model's actual call, including the sm70 triton staging
and the shared sinkhorn mixing) compared against the
pure-torch reference (mhc_fused_post_pre_torch) on
identical inputs: all four outputs (x, residual, post_mix,
res_mix) agree at fp16 noise level (max rel err
0.0002-0.0015). The mhc stack is FULLY CLEARED — prenorm
staging exact, post consistent, fused pipeline consistent.

Correctness scoreboard: weight representation EXACT,
linear compute ✓, MoE slicing ✓, mhc stack (all three
paths) ✓, vocab trim ✓.

THE ATTENTION COMPUTE IS THE REMAINING SUSPECT — the only
major component without a differential test. Concretely:
the group-paired batched apply's integration in the real
model (the standalone apply math is verified; the
integration — the o tensor's group layout, the reshape
(num_groups, -1) pairing with the slice order — is not),
the compressor's kv-score path, the sparse attention's
topk indices, the rotary embedding's inverse application.

Next experiment (unchanged): the forward-hook harness —
per-layer hidden norms on a constant input, then the
layer-level parity test. The attention's components are
testable the same way the mhc stack was: differential
tests against torch references.


## Inverse RoPE verified (correctness narrowing continues)

The sm70 inverse RoPE kernel's rotation (even' = even*cos +
odd*sin; odd' = odd*cos - even*sin) is the exact transpose
of the standard forward RoPE convention, and the cos_sin
cache layout (cos at [pos*rope_dim + pair], sin at
[pos*rope_dim + half_rope + pair]) matches the standard
vLLM layout. The grouped projection's slice↔group pairing
is also confirmed: contiguous head sharding makes local
group 0 ↔ global slice 2r, matching the loader's
global%gpr → local mapping.

Attention internals status: inverse RoPE ✓ (convention
analysis), grouped projection pairing ✓ (sharding
analysis), compressor kv-score guard ✓ (mathematically the
same projection). Remaining untested: the sparse
attention's topk indices and the full attention
integration — both need the forward-hook harness.

The hook harness (per-layer hidden norms on constant
input, then layer-level parity) is THE next experiment —
it localizes the broken component in one or two runs. All
component-level differential tests available without
hooks have passed.


## Hook-probe channel recipe (prepared for next session)

The forward-hook probe needs these pieces (each was a
failure mode this session):
1. Script placement: the probe script must live IN the repo
   directory (sys.path[0] = the script's dir for file
   scripts; a /tmp script imports the STALE site-packages
   vllm without routed_experts.py).
2. Serialization: llm.collective_rpc(callable) requires
   VLLM_ALLOW_INSECURE_SERIALIZATION=1 and a callable CLASS
   instance (a plain function fails serialization).
3. Stats channel: the rpc return path hung in testing —
   write the per-layer stats to a FILE from the worker
   instead of returning them.
4. GPU cleanup: a timed-out run leaves workers holding
   31 GiB/GPU — kill the PIDs from
   nvidia-smi --query-compute-apps before rerunning.

The probe itself: register forward hooks on
model.model.layers[i] (the decoder layers), record
out.float().norm() per layer on a constant input, then
walk the norm profile: explosion/collapse localizes the
broken component (attention vs MoE vs mhc wiring).


## Hook harness deployed — first norm profile captured

The env-gated hook probe (VLLM_HOOK_PROBE=1 in
gpu_model_runner.py's load_model) works: file-based
per-layer stats at /tmp/mhc_layer_stats_rN.txt. The rpc
channel (collective_rpc) hung consistently — the env-gated
in-process hook is the reliable channel.

First profile (43 layers = the full model, greedy 1-token
run, rank 0):
- Norms grow steadily 35 → 207 through the layers
- One extreme ratio: layer 19 (5.01x jump)
- min 8.6 (layer ~4), max 207.4 (final layers)

Interpretation: steady growth may be normal for mHC
(unbounded residual mixing) or the bug; the layer-19 jump
is the standout. Without a reference profile, ambiguous —
the decisive test is the LAYER-19 PARITY: run layer 19's
forward with the exl3 weights vs the same layer with
dequantized bf16 weights on identical input (the hook
harness can capture the layer's input tensor for the
replay).

Next session's concrete steps:
1. Capture layer 19's input tensor via the hook.
2. Replay layer 19 with dequantized bf16 weights (the
   get_weight_tensor path) — compare outputs.
3. Diverge ⇒ bisect layer 19's components (attention vs
   MoE); agree ⇒ walk to the next extreme-ratio layer.


## Layer-19 IO captured (hook harness working)

The extended hook (VLLM_HOOK_CAPTURE_LAYER=19) captured
layer 19's input and output tensors:
- input (256, 4096) fp16, norm 2886, std 2.82
- output (256, 4096) fp16, norm 5198, std 5.08
- No NaN/inf in either ✓

Norm-profile corrections from the full data:
- The 4 passes in the stats file: 3 profiling runs + 1
  generate. The profiling passes' norms go NaN (synthetic
  inputs → rsqrt(0)) — normal, not a bug signal.
- The generate pass's trajectory: oscillating 8-207 with
  several 2-5x swings (layer 5: 3.96x, 19: 5.01x, 21:
  0.41x) — characteristic of mHC residual stream energy
  exchange, NOT obviously a bug. The forward is
  numerically STABLE (no inf/nan in the real pass).
- The corruption is SUBTLE (wrong values, not overflow) —
  consistent with a convention/indexing bug.

The decisive next test: the IN-WORKER REPLAY — at the
captured layer, recompute the forward with dequantized
bf16 weights (get_weight_tensor path) on the captured
input, compare against the captured output. Divergence
localizes the bug to that layer's quantized compute;
agreement walks to the next layer. The replay runs inside
the worker (the weights are live there) — extend the hook
with a replay branch.


## CORRECTNESS BUG FOUND: fused_wqa_wkv's wkv partition never loads

The replay divergence (cosine 0.066) traced to the model's
loaded weights: the param dump + checkpoint comparison
shows the fused_wqa_wkv merged layer's trellis param is
(256, 96, 80) — the FULL output size (wq_a 64 tiles + wkv
32 tiles) — with:
- partition 0 [0:64]: wq_a.trellis at 100% match — loaded
  UN-NARROWED (the full wq_a tensor, no TP slicing)
- partition 1 [64:96]: ALL ZEROS — the wkv tensors never
  loaded (torch.empty → zeroed by the padded init or never
  written)

The model computes the wkv projection with zero weights →
the attention's KV path is dead → garbage from token 1 ✓
explains the first-token gibberish completely.

Root cause: the fused_wqa_wkv merged layer's TP handling
is broken end-to-end — the param allocated at full (un-
sharded) output size, the wq_a write bypassed the TP
narrowing, and the wkv write never fired. The un-narrowed
wq_a + zero wkv combination also explains the norm
profile's layer-19 oscillation (dead KV path changes the
residual mixing).

Fix direction: the merged layer's create_weights must
allocate the SHARDED output sizes (output_partition_sizes
already carries the per-rank sizes — verify what the
fused layer passes), and the loader's stacked-mapping
path must narrow both partitions (shard_exl3_col for
wq_a shard 0, and the wkv shard 1 write must fire —
trace why it doesn't: the stacked mapping entry exists
('attn.fused_wqa_wkv', 'attn.wkv', 1), so the loader's
shard_id=1 branch is the suspect).

After the fix: re-run the replay parity (cosine should
jump to ~1.0) and the greedy probe (coherent token).


## wkv partition FIXED — output changed, attention suspect remains

The vocab-parallel branch gate (the fix): the branch was
catching EVERY tp_size=1 trellis call — the merged
layers' wkv write landed at [0:32] (the wq_a region) and
was overwritten by the wq_a write, leaving the wkv
partition zeros. Gated on the LM head prefix.

Verification: the merged param's tail [64:96] now matches
the checkpoint's wkv.trellis at 100% (was 0%/zeros) ✓.

Output change: 'occ S Q'sef...' → 'ifiableifiableifiable'
— the fix had an effect but the output is still garbage
(REPETITION pattern now — classic broken-attention/KV
symptom).

Remaining suspect: the attention compute/state — the
compressor's kv-score quantized fallback, the sparse
attention's topk, the rotary's runtime application, or
the KV cache. The layer-19 replay parity (the harness is
deployed) is the next decisive test: with the weights now
correct, a replay divergence isolates the compute bug.

Also note: one run hit a flaky 'We expected the number of
MOE layers' assertion — a MoE-forward counting race, not
reproduced on the clean rerun.


## Post-cleanup verification (trace strip)

The diagnostic traces stripped from both repos; the clean
tree reproduces the same state: the pipeline runs, the
output is still garbled ('  ifiable obifiable  $' —
repetition/fragment pattern). The wkv fix is live (the
merged param verified 100% both partitions).

The remaining correctness bug: the attention compute or
model wiring. The layer-19 replay parity (the hook
harness captures the IO; the replay swaps the exl3
forwards for dense dequantized matmuls) is the prepared
next experiment — with the weights now verified correct,
a replay divergence isolates the compute bug to a
specific component.


## Replay methodology correction (KV cache contamination)

The layer-19 replay divergence (cosine 0.066, 1.4x norm)
is CONTAMINATED: the replay ran the layer's forward a
second time — the attention read the KV cache entries
written by the first pass, so the replay's attention
output differs from the captured output REGARDLESS of
the linears' correctness. The replay's linear-swap
methodology is invalid for attention layers as built.

Also confirmed: the layer-19 output is bit-identical
before and after the wkv fix — expected, since the wkv
projection feeds the KV cache (affecting subsequent
tokens' attention), not the current layer's output. The
wkv fix's effect shows in the GENERATION (the output
pattern changed) and the re-baseline norm profile
(early-layer norms dropped 35 → 13.7, the trajectory
reshaped).

Re-baseline profile on FIXED weights: bounded early
(4-7), gradual mid growth (5 → 20), accelerating late
(28 → 53 → 120 at the final layer). No non-finite
values. The final-layer jump (2.2x) is the new
localization candidate — the last decoder layer's output
feeds the lm_head.

Correct replay methodology (next session): either replay
a non-attention component (the MLP/MoE part only), or
isolate the KV cache (clear/restore around the replay),
or compare the LINEAR outputs directly (hook the linears'
inputs/outputs, not the layer's) — the linear-level
parity is immune to the cache state.

## Session: attention-block verification + exl3 head-order finding (2026-09-16)

### vLLM attention block CLEARED end-to-end
The exact-math reference computed from the vLLM's own captured q/kv
now reproduces the captured o_a stage at cos 0.981 (residual 0.217
rel_diff = the FP8 cache quantization, measured 2.7% on the roped
kv). The complete verified chain:
  scores = q_roped . k_roped * scale, causal mask
  softmax with the sink folded into the denominator (m = max(sink,
  scores); sink weight exp(sink - m) in the denominator)
  values = roped V (K=V shared-KV MQA)
  derot: CONJUGATE rotation on the output's rope slice (last 64
  dims per head) at the query position — applied to the attention
  OUTPUT, not the values
  then o_a (grouped) -> wo_b.
Earlier reference attempts omitted the derot-on-output term and
under-attributed the value path; the apparent cos-0.43 divergence
was the reference's composition error, not a vLLM bug.

### Verified-cleared components (vLLM side)
- mHC gate math: tilelang, torch fallback, hc_mix/hc_apply CUDA
  kernels — all match the HF transformers 5.15.0 reference (split
  order [H,H,H*H], Sinkhorn axis order, comb^T @ streams combine,
  2*sigmoid post, pre sigmoid+eps).
- Attention sink loading: bit-exact, checkpoint-ordered.
- q/kv projections: bit-exact at rope-identity positions.
- q post-norm+rope: bit-exact per-head.
- Shard convention: runtime-verified self-consistent (wq_b narrow
  takes the correct tile chunk; sink = heads 0-15; wo_a = groups
  0-1 on rank 0).
- FP8 KV cache: e4m3 round-trip error 2.7% on the roped kv — far
  too small to explain any observed divergence.

### exllamav3 multi-GEMV row bug (FIXED, verified)
exl3_gemv_multi hardcoded size_m=1 in its cooperative launch
(exl3_gemv.cu:542) while the caller (_project_o_grouped) passed
multi-row inputs into torch.empty outputs — row 0 computed, rows
1+ uninitialized garbage on every multi-token forward. Fixed by
driving the GEMV row by row with per-matrix pointer tables
(o[g][0][r], A_had[g][r], C[g][r]). Standalone generation flipped
from garbage to correct output ("The capital of France is Paris").

### exl3 head-order finding
The exl3 runtime head order is rotated +32 heads (+4 groups) vs
the checkpoint (activation-level: the exl3's q head slot h holds
the checkpoint head (h+32)%64's computation, cos 1.0 exact,
uniform, rope-consistent). The exl3's sinks are checkpoint-ordered
(verified) — mispaired with the rotated head slots; bounded effect
(delta_max 1.81 vs score spread ~9, ~20% relative). The exl3's
o_a groups are checkpoint-ordered. The exl3 model tolerates this
(coherent output) but is degraded — it is NOT a canonical
reference for per-head comparisons.

### Slice-order test (inconclusive, reverted)
The +4-group slice remap (wo_a only, wo_b left identity) produced
different-but-still-garbled output — expected, since fixing wo_a
alone breaks the wo_a<->wo_b pairing. The checkpoint's wo_b is
flat (not slice-qualified), so a slice-order fix would need the
wo_b column blocks remapped identically. The test cannot decide
the slice order without completing both sides.

### Current status
The vLLM still garbles end-to-end with the attention block,
mHC, shard convention, and FP8 cache all verified correct. The
bug lives OUTSIDE the verified set: MoE/hash routing, the mHC
apply composition at runtime, deeper-layer effects, the final
norm/head, or a layer-type-specific path. The per-layer stream
comparison infrastructure (variance-gated captures, per-site
mHC states, MLA q/kv hooks) is in place for the next bisect.

## ROOT CAUSE FIXED: TP slice selection missing the rank term (2026-09-16)

### The bug
`load_weights`'s slice-qualified branch (wo_a.slice.N.*) kept slices
0..gpr-1 on EVERY rank — the keep condition `int(_idx) // _gpr != 0`
had no rank term, so every rank loaded rank-0's slices and skipped
the rest. At TP=4 with o_groups=8, gpr=2: all four ranks ran their
attention's grouped o_proj with slices 0-1's weights against their
own (correctly sharded) head-groups 2-7's activations. Every layer's
attention output was wrong at TP>1; TP=1 was unaffected (which is
why the standalone path worked).

### The fix
Rank-gated slice selection: rank r keeps slices gpr*r..gpr*(r+1)-1,
_local = _idx % gpr. The checkpoint's slice order is identity
(slice N pairs with head-group N) — the +4-rotation hypothesis was
tested and rejected (worse output, and the wo_b flat layout
confirmed identity).

### Verification
- "The capital of France is" -> " Paris" (+ coherent multi-capital
  continuations, correct code answers on the coding prompt).
- Hash table (tid2eid) verified loading bit-exact (int64->int32
  cast clean) — was a suspect, cleared.
- The full attention chain verified independently: exact-math
  reference (roped K scores, sink-folded softmax, roped V,
  derot-on-output) reproduces the captured o_a at cos 0.981
  (residual = the FP8 cache's 2.7% quantization).

### Cleared along the way
- FP8 KV cache: e4m3 round-trip 2.7% — too small to matter.
- Softmax variants (sink fold, sink-as-key, no-causal, scale
  variants): all equivalent under the exact reference — the
  differentiator was the derot-on-output term.
- hc_head_fuse tilelang kernel: matches HF HyperHead exactly.
- tid2eid hash routing kernel: matches HF HashRouter (gather +
  renormalize + scaling).

### Remaining known issues
- exllamav3 head-order rotation (+32 slots vs checkpoint): the
  exl3 runtime is degraded-but-functional; separate from the vLLM.
- exllamav3 multi-GEMV size_m=1 row bug: FIXED this session
  (row-by-row driving in _project_o_grouped).
- GEMV IMA (fast decode path): still open, next up.

### exl3 head-order resolution (final)
The group-major store math in dsa_attn (dsa_triton.py:283-285:
`out[h // HPG, row, (h % HPG) * D + d]`) settles the compensation
question: the o_proj's group g consumes the exl3's head slots
g*8..g*8+7 — the same (rotated) slots the q occupied. Since the
rotation is present in EVERY layer's q slots AND every layer's
o_proj grouping identically, it is a consistent relabeling: the
model computes a valid (permuted) function, not the checkpoint's.
The sinks are the only absolute-indexed component that does not
permute with the slots — their mispairing is the bounded ~20%
effect. The exl3's degradation is bounded and cannot explain the
cos-0.43 gap; that gap was the vLLM's slice-selection bug (every
rank loading slices 0-1), fixed and verified above.

## GEMV IMA: hypotheses narrowed (2026-09-16, post-fix session)

With the slice-selection fix landed, the GEMV IMA work resumed.
New discriminator results:

- Shard-view alignment: DEAD. The trellis's tile stride is 160
  bytes (80 int16 words per tile for K=5), so every tile-granular
  shard view is 4-byte aligned — the narrowed-B-view alignment
  hypothesis cannot produce the fault. Tested empirically: the
  exact failing shape (K=5, 4096x512, mul1) with narrowed views
  at every rank offset passes standalone.
- Cooperative-launch concurrency: WEAKENED. The cooperative GEMV
  co-exists with a busy side-stream kernel (matmul chain
  occupying SMs) — the driver queues the launch rather than
  faulting. Not the mechanism (at least not in this simple form).
- Buffer sizing re-verified: the ws (16 MiB) covers the ksplit
  partials (4 * size_n max); the locks buffer is unused by the
  sm70 GEMV (cooperative grid.sync instead of lock-based
  reduction); the multi/dual variants' (matrix, group) indexing
  stays within the buffers.
- Static bounds re-verified for the exact failing shape: the B
  index max = the tensor's last element; the C/A/suh pointer
  alignments hold for the wkv shard's offsets (3072 B, 768 B —
  both 4B-aligned).
- The dual/multi GEMV variants (exl3_gemv2, exl3_gemv_multi) and
  the tiled sm70 path (exl3_gemm_gr's 8-row tiling) share the
  same kernel and launch checks — no variant-specific hazard
  found.

Confirmed next step (unchanged from the plan): the dynamic repro —
build the ext with bounds-printf debug asserts in
exl3_gemv_sm70_kernel.cuh (printf the faulting index + thread +
block on the B/C/A accesses), then run the default-env repro
(rows 1-16 route to the GEMV; RECONSTRUCT_MIN_ROWS unset). The
static analysis is exhausted; every remaining hypothesis needs
the faulting address from the dynamic run.

## GEMV IMA: full elimination record (2026-09-16, continued)

The in-vivo fault (default RECONSTRUCT_MIN_ROWS: rows 1-16
route to the GEMV) is a deterministic IMA at the m=1 K=5
n=512 wkv-slice GEMV launch (exl3_gemv.cu cuda_check at the
cooperative launch), on the first decode step, on every TP
rank. With CUDA_LAUNCH_BLOCKING=1 the error surfaces at
exl3_gemv.cu:289 (the launch's cuda_check).

Eliminated with evidence:

- B-load OOB: device printf bounds-check on every ld_b
  access across all runs — zero prints.
- Static extents: TORCH_CHECKs on A_had/suh/svh/C numel vs
  the launch extents in exl3_gemm_gr — all pass.
- Standalone GEMV with the REAL checkpoint wkv tensors
  (m=1, K=5, cb=2, exact shapes): compute-sanitizer clean,
  correct output.
- Concurrent two-stream GEMVs sharing the A_had cache
  tensor (wqa n=1024 + wkv n=512): sanitizer clean.
- Custom all-reduce disabled: fault persists.
- Env bisections (fault persists in every combination
  tested): MHC_FP32_STAGE=0, QNORM_KV_FUSED_TP4=0,
  FP13_GEMV=0, PRIVATE_COMPRESSOR_STATE=0.
- Aux-stream concurrency: attn_gemm_parallel_execute forced
  serial (aux_streams=None) — fault persists.
- Non-contiguous trellis: TENSOR_DBG dump of every K=5
  GEMV's trellis in-vivo — all contiguous with the exact
  expected strides ((2560, 80, 1) for (256, 32, 80)).
- Triton sparse-attention kernel: the earlier trace
  attribution was the error SURFACING at the next CUDA call
  (Triton's module load), not the origin.
- Latent bug fixed en route: the cached xh (A_had) workspace
  was sized (1, in_features) while the m>1 GEMV path's
  input-Had phase writes m * in_features halves — sized to
  EXL3_GEMV_SM70_MAX_M (8) rows now. Not the in-vivo
  fault's root cause (the fault persists with the fix) but
  a real overflow for every batched GEMV call.

Sanitizer on the full TP=4 run is blocked: NCCL ships no
sm70 SASS; under compute-sanitizer injection the PTX JIT of
NCCL's kernels fails (cudaErrorNoKernelImageForDevice at
ncclInitKernelsForDevice) and NCCL init aborts the workers.
CUDA_FORCE_PTX_JIT=1 breaks the SASS-only exl3/custom-AR
kernels (468 symbol-not-found errors). TP=1/2 are impossible
(111 GB checkpoint).

Status: the GEMV fast path stays opt-in. The green config
(VLLM_EXL3_RECONSTRUCT_MIN_ROWS=1: all rows reconstruct,
the GEMV never fires) is verified end-to-end: ' Paris. The
capital of Spain is Madrid. The capital of Italy is Rome.'
Root-causing the in-vivo fault requires a sanitizer-
compatible TP=4 run (NCCL with sm70 SASS, or a
sanitizer-tolerant NCCL build).

## Correction: the green-run mechanism (2026-09-16, final)

The earlier "green config" runs used `VLLM_EXL3_RECONSTRUCT_MIN_ROWS=1`
— an env var that DOES NOT EXIST in either tree (the routing constant
is `AUTO_RECONSTRUCT_THRESHOLD = 144` in exllamav3's exl3.py, and the
only other knob is `no_reconstruct` in infer_params). Those runs'
env var was inert; their success mechanism was unverified.

Re-verified with a REAL bypass: `AUTO_RECONSTRUCT_THRESHOLD = 0`
(temporarily, in exllamav3's exl3.py — routes every row to
reconstruct_hgemm; the GEMV never fires) on the current clean build:

  OUT: ' Paris. The capital of the United States is Washington, D.C.
  The capital of Germany is Berlin. ...' — exit 0, zero GPU asserts.

So the core claim survives with corrected evidence: bypassing the
GEMV entirely yields green; the default routing (rows 1-16 to the
GEMV) faults deterministically at the m=1 K=5 wkv-slice GEMV launch.
The GEMV fast path remains the crash site; it stays opt-in until
root-caused under a sanitizer-compatible TP=4 setup.

Recommended interim default for this stack: keep the GEMV fast path
disabled (e.g. `AUTO_RECONSTRUCT_THRESHOLD = 0` or an equivalent
shipping knob) until the in-vivo fault is root-caused.

## Interim operating recommendation

Until the in-vivo GEMV fault is root-caused under a sanitizer-
compatible TP=4 setup, run this stack with the GEMV fast path
disabled. The verified-green configuration: `AUTO_RECONSTRUCT_
THRESHOLD = 0` in exllamav3's exl3.py (or an equivalent shipping
knob) — every row reconstructs, the GEMV never fires, output is
coherent end-to-end. The GEMV path itself remains
sanitizer-clean standalone with the real checkpoint tensors;
the in-vivo-only fault is documented in the elimination record
above.

## Serve-config throughput probe (2026-09-16)

First end-to-end throughput measurement on the verified-green
config (EXL3_SM70_GEMV_DISABLE=1, the new env kill switch in
exllamav3 — commit 55d9744):

- Config: TP=4, max_model_len=256, max_num_seqs=4,
  gpu_memory_utilization=0.90, enforce_eager, fp8 KV.
  (0.95 OOMs at the compress_norm_rope Triton launch with a
  longer prompt — the driver needs ~1.6 GiB/GPU of headroom
  beyond the 0.95 profile; 0.90 is the working ceiling.)
- Result: 3.52 tok/s decode (26 tokens in 7.4 s, greedy,
  ignore_eos), coherent output, zero asserts.
- Context: the documented reconstruct-only range was
  ~2.2-2.9 tok/s; the ~10x per-call GEMV claim is in the
  dispatch docstring. The measured 3.52 tok/s on the
  bypassed path is consistent with the reconstruct-only
  range (the probe's prompt is long, so prefill dominates
  less than in the earlier correctness runs).

Throughput verdict: the GEMV fast path's ~10x per-call claim
would put decode at ~30+ tok/s — a material gap. The IMA
root-cause work (NCCL sm70 rebuild + sanitizer) is justified
by this measurement whenever decode throughput matters.
