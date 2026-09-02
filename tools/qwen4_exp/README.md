# Qwen4Exp QSA E4M3 KV calibration

This directory contains the offline-only tooling for the first supported
contract:

- SM70, TP4, MTP0;
- FP16 activations during calibration;
- per-layer, per-tensor E4M3 K and V scales;
- E4M3 main QSA KV cache and FP16 raw/compressed indexer caches.

`qsa_kv_calibration.py prepare` freezes source records into exact prompt token
IDs. For an existing audited corpus, pass both `--corpus-manifest` and
`--processor-audit-jsonl`; every processor tensor hash must match before an
output is created.

`qsa_kv_calibration.py collect` runs the final checkpoint through the normal
offline inference engine with FP16 KV. The `COLLECTING` marker is created only
after engine initialization, excluding dummy/profile/graph-capture forwards.
Use `--shard-manifest` to collect several named shards in one engine lifetime.
The conservative V100 defaults are 2,048 batched tokens, 32 sequences, and
0.75 GPU memory utilization; record any overrides in the resulting manifest.

`summarize` requires all expected QSA layers and emits independent K/V max-abs,
scale, percentile, saturation, and per-shard stability data. When nested runs
produce slightly different extrema because batching changes floating-point
order, `envelope` preserves one report's distribution statistics while taking
the observed max across the other reports. `overlay` creates a derived
checkpoint directory, writes `model-kvscales.safetensors`, and merges its 24
entries into the standard `model.safetensors.index.json`. It never modifies the
base checkpoint.

Smoke-only or incomplete-corpus reports must not be promoted into a production
overlay. A unit-scale overlay is available only as an explicit negative control
via `--unit-scale-negative-control`.
