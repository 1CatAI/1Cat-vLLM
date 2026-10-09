# 1Cat-vLLM architecture and layering

This document is the entry point for how 1Cat-vLLM is organised on top of
upstream vLLM and how new work must be placed. It replaces "add another branch
where it is needed" with explicit layers, ownership and a ratchet that keeps the
generic code from absorbing more project specifics.

## Why

A 2026-10-08 audit of `main` (#1041) against upstream vLLM found that hardware
(SM70/V100) support, model families (Qwen3.8 Flash-Next, DFlash2, QUASAR, ...),
quantisation formats and acceleration paths are written directly into generic
vLLM modules and into each other:

| Measure | 1Cat-vLLM | upstream |
| --- | ---: | ---: |
| `VLLM_*` environment variables | 981 (688 model/platform specific, 313 default-off) | 296 |
| `vllm/envs.py` | 18,675 lines | 2,456 |
| `vllm/v1/worker/gpu_model_runner.py` | 13,163 lines | 7,540 |
| `mamba/gdn/qwen_gdn_linear_attn.py` | 8,120 lines | 2,059 |
| `vocab_parallel_embedding.py` | 1,309 lines | 639 |
| Flash-V100 attention backend | 9,870 lines, 160 KV-dtype branches, 50 route sites | — |
| Generic modules with model/platform/env coupling | 182 files: 2,325 model, 3,964 platform, 335 raw env references | — |

Consequences we have already paid for: a KV page-size change silently disabled
the DFlash2 split draft attention; `--block-size 2048` silently made the 75T
prefill route unreachable; E4M3 decode uses a different partition/reduction
policy than FP16 and is slower than it; every new KV or weight format has to be
copied into each acceleration path; four parallel SM70 MoE implementations
(AWQ, NVFP4, FP8, MXFP4) repeat the same structure.

## Layers

Dependencies point downwards only. A layer never names anything that lives in a
layer above it.

| Layer | Owns | Lives in |
| --- | --- | --- |
| 6. Model adapters | Model families and their fusions (HC, QSA, PLE, MTP heads, DFlash drafts) | `vllm/models/<family>/`, `vllm/model_executor/models/`, spec-decode proposer packages |
| 5. Engine (generic) | Scheduler, KV manager, runners, sampler, config, distributed state | upstream vLLM modules — kept as close to upstream as possible |
| 4. Dispatch | Selecting an implementation from (stage, shape, format, platform); capability declarations; route accounting | backend / kernel-selector modules (e.g. `kernels/linear` selectors, attention backend routing) |
| 3. Operators | Attention, GEMV/GEMM, MoE, collectives, norms — parameterised by format codecs | `vllm/model_executor/kernels/`, attention backends, `csrc/`, `flash-attention-v100/` |
| 2. Formats | Weight formats (GGUF types, NVFP4, AWQ, FP8) and KV formats (FP16, E4M3, INT8) as codecs: byte layout, scales, read/write | one definition per format, shared by every operator |
| 1. Platform | SM70 capabilities, policies, profiles, observability | `vllm/config/kernel.py` (`KernelConfig.sm70_*`), `vllm/sm70_profiles/`, platform hooks |

Rules:

1. **Generic modules do not name models or the platform.** They expose a hook
   (a method on the platform, a field of `KernelConfig`, a registry entry, a
   model-state class) and the specific code registers itself. `check-layering`
   enforces this with a per-file ratchet (below).
2. **Policies are configuration, captured once.** Behaviour is read from a typed
   `KernelConfig` field resolved at engine construction — never from
   `os.getenv` at forward/capture time. Legacy `VLLM_*` names are adapters into
   `KernelConfig` (the existing `check-sm70-linear-policy` pattern). New
   environment variables are for debugging only.
3. **Formats are codecs, not branches.** An operator is written once against a
   codec interface; adding a format means adding a codec, not editing every
   path. Route names never contain a format.
4. **Dispatch is declared and loud.** Every implementation declares what it
   supports. When nothing matches, the fallback is logged and counted — never
   silent. The "path x format x shape" matrix is generated from the
   declarations and checked by tests.
5. **Experiments do not live in `main` forever.** A default-off path either
   graduates (becomes the default, with data) or is marked deprecated with its
   negative result linked in the design log. Retain experimental and `_old` /
   `_fixed` / `_vN` paths for reproducibility; do not delete them.
6. **One home per concept.** A behaviour is implemented once; variants are
   template/codec parameters, not copied files.

## The ratchet

`tools/pre_commit/check_layering.py` counts, for every non-owner module under
`vllm/`, references to model families, to the SM70 platform, and raw
`VLLM_*` environment reads. The current debt is recorded per file in
`tools/pre_commit/layering_baseline.json`.

- A change that **adds** coupling to a generic module fails pre-commit. Move the
  code behind a hook instead.
- A change that **removes** coupling passes; run
  `python tools/pre_commit/check_layering.py --update` to lock the reduction in.
  `--update` refuses to record growth.
- `python tools/pre_commit/check_layering.py --report` prints the remaining
  debt, largest first. This is the progress metric of the refactor.

Owner modules (excluded from the model/platform counts) are model packages and
files whose path names their platform or feature (`*sm70*`, `*turbomind*`,
`*gguf*`, `*dflash*`, `qwen4_exp`, ...). `vllm/envs.py`, `vllm/envs_metadata.py`
and `vllm/config/kernel.py` are configuration registries and are excluded
entirely.

## Where things live

| Concept | Home |
| --- | --- |
| KV-cache storage formats | `vllm/v1/attention/kv_codecs.py` (`KVCodec`: FP16, BF16, FP8-E4M3, FP8-E5M2). Routes admit codecs, never `kv_cache_dtype` strings. |
| Flash-V100 attention | `vllm/v1/attention/backends/flash_v100/`: `ops` (native operator loading), `routing` (route accounting, decode partition/XQA admission), `kv_layout`, `masks`, `dense_prefill`, `spec/` (feature metadata hooks and device preparation), `metadata` (common metadata), `impl` (initialization/forward), `decode`, `prefill`, `verify`, `debug_compare`, `state` (shared flags), `backend`. Modules reach each other through the module object (`_routing._record_route`), so rebound globals and monkeypatches have one owner. `flash_attn_v100.py` is a compatibility module that forwards reads and writes, including wildcard imports. Public package re-exports resolve the owning module dynamically. The original logger name and shared one-shot flags are retained. |

A pure move between files keeps every coupling total constant; record it with
`python tools/pre_commit/check_layering.py --accept-moves`, which refuses to
run if any total grows.

## Refactor roadmap

Every step is behaviour-preserving (bitwise outputs, identical route hits, no
performance regression), lands as its own PR and can be reverted alone.

1. **Guardrails** — this document, the ratchet, the debt report. *(done)*
2. **Configuration** — finish moving runtime `os.getenv` reads in generic modules
   into `KernelConfig` / `envs` captured at init; mark default-off
   experiments deprecated when their negative results are recorded.
3. **Extract platform and model code from generic modules**, largest first:
   `gpu_model_runner.py`, `qwen_gdn_linear_attn.py`, `config/vllm.py`,
   `vocab_parallel_embedding.py`, `gdn_attn.py`, `config/speculative.py`,
   `v1/core/sched/scheduler.py`. Metric: diff against upstream for these files
   and the ratchet totals.
4. **Format codecs and dispatch registries** — KV formats first (FP16, E4M3,
   then INT8 on the same interface), then weight formats and the SM70 MoE
   family; split the Flash-V100 backend into routing / metadata / decode /
   verify / prefill / KV-codec modules. *(KV codecs and the module split done;
   declarative route table and shared XQA admission introduced;
   next: one grouped/XQA operator family per codec,
   metadata feature hooks and the implementation method split done;
   next: remove feature names from forward orchestration.)*
5. **Kernels and build** — consolidate extension modules, mark retired variant files deprecated,
   split multi-thousand-line kernels by responsibility.
6. **Process** — PRs state their coverage in the generated matrix; new knobs go
   into `KernelConfig`; design notes become one overview plus per-component
   READMEs and decision records, with data files out of `docs/`.
