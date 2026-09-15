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
