# Retained SM70 MTP optimizations

This PR is synchronized with main
`c39f53abae51df6bf8ef122e34975a9f3596a93c`.
The full serving comparison remains frozen on main
`4e5357b59294edd31c46e66ae8219503a843b2bd`. Its measured E commit is
`d686ca549b6087ad36211d4a2cb771c840399925`. The subsequent report-only rebase
onto `16628e2f0a70761ba6525bf46e5e844c04b3e66f` replaced the equivalent
serializer with upstream PR1160; that deployed runtime is
`69801806572ea094b91d5f18864beb22c6c8b6fb`.

This later conflict synchronization includes upstream PR1162 startup-policy
checks, PR1163 GDN lifecycle sharing and PR1164 native GEMM ownership changes.
It has CPU/static checks only: no native rebuild, GPU run, model-performance
remeasurement or production deployment was performed. In particular, the
historical results do not validate PR1164's changed native implementation.
The measured numbers and their original source revisions remain unchanged.

This change retains the parts of PR903 that are absent from main. Ordinary
upstream collective, graph, GDN projection, attention and native resource
owners remain authoritative. Core native sources, CMake and package build
rules have no residual diff from the base.

## Scope

| Area | Retained behavior | Upstream behavior preserved |
| --- | --- | --- |
| Sparse attention | Segmented page4 queries and grouped E4M3 multi-KV-head FP32 accumulation | Existing attention runtime and unsupported-shape fallback |
| GDN verification | Fused small speculative recurrence, BA preparation and shared state metadata | Higher-priority context-core and mixed-QKV routes |
| Auxiliary MTP kernels | Exact shared gate, n-gram, short-convolution and MoE permutation fusions | Native PLE admission and ordinary fallback |
| Drafter | TP-local greedy exchange, padded-row routing and selective eager prefill MoE | Captured communication policy and full prefill for unsupported topology |
| MoE tuning | Additional drafter row counts through M40 | Native-compatible M1/M5 tiles and temporary legacy warmup scope |
| Startup | Request-count/kernel warmup coverage and optional JIT telemetry | Ordinary compilation and loading |
| Mixed prefill | Existing opt-in fixed-step pacing compatibility | Disabled by default and in the comparison serving configuration |

Selective prefill requires a verified single-block MTP model. It preserves
all draft attention/KV writes, computes MoE only for rows that can be sampled,
and omits discarded draft decode for batches whose prompts remain incomplete.
Graph captures retain the ordinary full path. Tests include unsupported
topologies, mixed batches, exception cleanup and padded-row scope restoration.

The MoE selector reads the value of upstream's legacy-scope ContextVar.
Treating the ContextVar object as a boolean would disable tuned tiles. The
selector also uses the initialized upstream MoE policy. GDN/PLE admission
and diagnostic exclusions consume their corresponding upstream owners.

## Configuration and validation

Legacy `ONECAT_*` aliases remain available for existing deployments. The
production comparison enables the retained GDN, auxiliary and drafter
fusions, segmented page4 attention, grouped short-convolution metadata and
selective eager MTP prefill. Fixed-step prefill pacing remains zero. No model
format, weights, quantization or global runtime dtype is changed.

Build normal `_C` and `_moe_C` extensions for native policy ABI 67 and runtime
ABI 1. Rebuild Flash-V100 for this tree. Old native binaries are not valid
substitutes for the initialized upstream runtime interfaces.
The bundled FlashQLA extension must expose GDN policy ABI 1, as required by
the current upstream GDN owner.

Upstream PR1160 serializes initialized diagnostic filter sets without
changing policy objects or computation hashes. The frozen comparison used
the same equivalent local repair in both arms because the earlier main
failed while creating the internal MTP draft after MoE initialization.
That duplicate local repair is absent from the final implementation.
Both arms also preserve explicit Inductor combo settings through SM70
platform defaults and use identical ordinary deterministic compiler options.

Qualification uses the same FP16/TP4/MTP4 serving configuration, request bytes
and weights on both arms. Prefill-only and normal generation are measured
separately, with repeated C1/C4/C8 runs and full 32K/64K uncached prompts.
Report queue, prefill, TTFT and total completion time separately. Verify
cached-token counters and computed KV work. A total-time improvement does
not excuse a reproducible prefill or TTFT regression.

Kernel choices are compared only for matching generated source and candidate
sets. Each arm has a separate ordinary compiler cache. Timing excludes code
warmup; cache changes during measurement are reported. No loader interception
or kernel allowlist is introduced.

The [fixed comparison and final-source equivalence](https://github.com/areslp/1Cat-vLLM/tree/evidence/pr903-e-166-20261010/docs/sm70-e-qualification)
record exact measured commits, commands, per-case results and limitations.
All 26 historical C1 inputs improve against the fixed control, with identical
output and MTP work. Cold prefill and TTFT improve too. Some paths remain
slower than the older production runtime; those differences are disclosed
separately from patch-only gains. Earlier measurements are retained as
[historical evidence](https://github.com/areslp/1Cat-vLLM/tree/b352d3967a166bdc468b21b959ebe98529032d1a/docs/sm70-c73-qualification)
and do not qualify this revision. AI assistance was used.
