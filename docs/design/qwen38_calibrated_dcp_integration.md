# Qwen3.8 calibrated E4M3 and disk-PLE DCP integration

## Scope and prerequisites

This is a qualification/integration of [DCP2 PR #696](https://github.com/1CatAI/1Cat-vLLM/pull/696),
not a competing cache layout. Base main: `b034648012244ab712df05b93e6d8fff877a6f2f`.
Imported DCP head: `9e4236c2a6f5a15c44c1a0542ab53a779eade730`.
The high-precision and PLE placement prerequisite is
[PR #701](https://github.com/1CatAI/1Cat-vLLM/pull/701),
commit `71c7de134e9107655ad01a1e329f0813c8b0c1b8`.
Neither dependency is represented here as already merged into main.

New integration work:

- Preserve main's CPU-index selection for interleaved GDN speculative rows,
  while retaining #696's slice fast path for leading speculative rows.
- Preserve per-group MTP prefix-cache semantics alongside global/local DCP
  block-span conversion.
- Admit disk-PLE with the Qwen3.8 TP4, PP1, no-MTP DCP2 contract. DCP does not
  shard hidden/input tokens consumed by the PLE process. Other previously
  unsupported PLE topologies remain rejected.
- Default only SM70, Qwen3.8 NVFP4, TP4, no-MTP, DCP1/2 from `auto` to
  `fp8_e4m3`. Explicit cache dtypes win. Automatic selection enables strict
  QSA scale loading unless explicitly overridden; unit scales are not quality
  evidence. Existing FP16 HC/GDN compute and FP32 reductions are retained.
- Retain the model-aware fast decode defaults with E4M3 KV. This is route
  selection, **not** a claim that the FP16 97 tok/s baseline transfers to DCP2.

Only target main QSA K/V is sharded. Selector, compressor, GDN and PLE state
stay replicated. Therefore neither E4M3 nor DCP doubles the entire pool by
definition: usable capacity must come from actual allocator/worker records.

## Calibrated model view

The original RadixArk checkpoint has no main QSA K/V scales. Use the published
[24-tensor scale pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales)
at revision `edaf669f134747a6d3535956e42c36fa75be52f9`
(`v1.1-base-7b719225`). Its materializer checks both hashes and creates a
separate standard checkpoint index with symlinks; it never modifies weights.

```text
Base index SHA256:
da5ca9c3b65e48e151329e64e141c2fa700bf2f99aec53cc014e4b52a6ff7a84
Scale shard SHA256:
bbf767cd46fe3ac52793c87ec069a964bf6fb64c5b6183959c734b9772936cc9
```

Both hashes match this machine's artifacts. The scale pack's published
validation extends to 128K; that does not certify this integration at 256K.
The MTP scale overlay in #696 is a separate contract; automatic E4M3 selection
and disk-PLE admission in this follow-up are deliberately no-MTP first.

## Reproducible bounded model check

Build/install the native extensions from this source, including Flash-V100
and FlashQLA. Do not use private sidecars, `LD_PRELOAD`, another worktree's
binaries or a precompiled older wheel to claim full-model reproduction.

Run in a fresh process with four exclusively available V100-SXM2-32GB GPUs,
Torch 2.10.0+cu128, CUDA 12.8 and enough host RAM for the hybrid PLE views.
Use task-owned compiler caches. The driver never starts an API and shuts
workers down in `finally`.

```bash
export CUDA_VISIBLE_DEVICES=4,5,6,7
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS=1
export VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
export TRITON_CACHE_DIR="$PWD/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"

.venv/bin/python -m benchmarks.benchmark_qwen38_dcp_quality \
  --model /path/to/calibrated-model --dcp 1 --kv-gib 4 \
  --long-context --output /path/to/dcp1.json
.venv/bin/python -m benchmarks.benchmark_qwen38_dcp_quality \
  --model /path/to/calibrated-model --dcp 2 --kv-gib 4 \
  --long-context --reference /path/to/dcp1.json --output /path/to/dcp2.json
```

The controls keep max context 262144, chunk8192, C4 admission, no MTP,
no prefix caching and a fixed 4 GiB/card KV budget. They record physical
cache allocations, per-owner geometry, resolved graph/precision/PLE settings,
official-sampling short output health, deterministic token comparisons,
8K/32K/near-256K retrieval and the exact 262143+1 finite-logprob boundary.
Prefill and decode metrics are recorded separately. Short health responses
are not sustained-decode speed benchmarks. Omit `--long-context` only for a
short preliminary check; never label that result 256K quality acceptance.

## Qualification status

On the integration worktree, before full native rebuild completion:

- 98 operator/metadata cases passed: localization, partial attention vs FP32,
  FP16/E4M3, empty owners, packed-page writes/reads, GDN row selection.
- 213/214 cache/GDN/config cases initially passed; the remaining old test
  unpacked a now-four-field `SpecGroup` as a three-tuple. It now uses named
  fields and passes in the targeted rerun.
- 35 PLE/cache cases passed, including the corrected cache-layout case.
- Real four-process TP4/DCP2 NCCL eager/graph gate passed: FP16/E4M3,
  rows 1/5/33, both combine backends, changed inputs between replays, FP32 oracle.
- 11 selected allocator/offload regressions and the precision-policy test passed.
- After adding #701, 87 combined default-selection/PLE tests passed. The
  late-default test now isolates environment writes, preventing false failures
  in subsequent PLE dispatch tests.
- Merge pre-commit checks passed, including mypy and forbidden CUDA API checks.

These are component gates, not full-model accuracy or throughput acceptance.
Full-model DCP1/2, exact 256K quality, actual usable capacity and matched speed
remain promotion gates. Imported PR author measurements are not our results.
