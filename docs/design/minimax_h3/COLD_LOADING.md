# H3 cold loading and reusable prepared weights

## Scope and baseline

Integration base: `fe67339ddf862df0441a4e844fc664d9cafdffd0` (`onecat/main`).
This scope covers original H3, INT8 ConvRot and LightX2V/FastH3 adapters;
checkpoint precision, attention, sampling and output semantics stay unchanged.

The deployed four-card V100 host has 62 GiB RAM and uses disk-backed masters.
A real original FastH3 request spent 973.19 seconds preparing the service and
104.95 seconds generating. At 604 seconds into loading, worker disk counters
already totalled 99.57 GiB read and 109.08 GiB written. The old path loads row
storage, transforms it to column storage and snapshots it again. Temporary
masters are reconstructed at each service launch. Shared text encoder loading
also repeats across H3 variants. These observations are not a new benchmark.

## Intended contract

- Prepare final projection strides before loading, avoiding a second full copy.
- Reuse exact per-component, per-TP-rank prepared CPU storage, including the
  shared text encoder, without copying a second complete model for publication.
- Key entries by source/checkpoint identity, precision, topology and relevant
  adapter/configuration. Retain original model files and all existing variants.
- Publish atomically only after loading/validation succeeds; validate metadata
  and data on restore. Use private mappings to prevent writes changing cache.
- Bound disk use and evict only inactive cache entries under leases. Corrupt or
  incomplete entries must rebuild, never silently serve unverified weights.
- Report preparation subphases and distinguish first construction from reuse.

## Validation status

Implementation and targeted tests are in progress. No new startup speed or
quality claim is made. GPU validation waits for the user's current generation
to finish; no active user task is interrupted. Real first-load/reload timing
and output comparison are required before promotion.

AI assistance: OpenAI Codex.

## Candidate implementation

`--prepared-weight-cache` enables exact rank-local prepared transformer and
text-encoder storage under `VLLM_CACHE_ROOT/h3-prepared`. The default budget is
128 GiB (`--prepared-weight-cache-gib`). Native callers remain opt-in. Studio
uses the capability probe to enable it independently of generation fast mode.

The initial fill allocates the final projection strides before loading. A
completed entry includes checksums of all storage files, tensor metadata and
its manifest. Subsequent loads validate it before binding private mappings.
Keys include checkpoint identity, source, Torch/CUDA, processing configuration
and TP rank/topology. Ordinary LoRA sidecars are applied after the base cache;
FastH3 fusion is keyed by its exact adapter. Active entries are protected by
lifetime leases. Only inactive prepared entries can be evicted; original model
files are never removed. Missing, incomplete or corrupt entries are rebuilt.

Loading reports actual tensor or byte counts separately from component counts.
No overall time percentage or performance prediction is synthesized.

CPU validation: 80 passed, 3 skipped across prepared weights, host residency,
service, progress, FastH3 and Studio fast-path suites. These cover exact signed
INT8 bytes, FP16 layout, shared storage offsets, cache corruption and active
lease protection. Real GPU startup, output parity and speed remain pending;
this candidate must not be promoted based only on the CPU result.
