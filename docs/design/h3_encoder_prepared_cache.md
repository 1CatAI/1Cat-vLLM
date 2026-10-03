# H3 encoder and DiT checkpoint reuse

The H3 text encoder and DiT automatically reuse completed CPU weight snapshots.
Checkpoint identity, source code, Torch version, dtype and rank topology
separate entries. Tensor shapes, dtypes, strides and aliases are validated
before reuse. Publication checksums the bytes; unchanged file identities
allow subsequent loads to avoid rereading them. Changed files are checked
again and damaged or unfinished entries are rebuilt.

Snapshots use private mappings, so runtime writes cannot modify the reusable
weights. Active entries retain a lease and are not evicted. The cache has a
128 GiB default capacity and keeps 1 GiB free. Storage failures retain the
ordinary checkpoint-loading path; startup logs explain misses and reuse.

DiT snapshots contain the TP-local logical checkpoint tensors, after the
ordinary QKV/scale loader and completeness checks and before quantizer layout
conversion. Warm loads still run every quantizer post-load method and the
mixed-precision validation. INT8 row scales therefore retain their constructor
shape in the snapshot and are flattened by the same post-load method on both
cold and warm starts. FP16 and INT8 row/column layouts keep their ordinary
conversion and compute paths. Fused or reconstructed adapter checkpoints use
the ordinary loader, with the reason reported during startup.

Use `vllm video ... --disable-prepared-weight-cache` to disable this cache.
Programmatic callers can set `H3Config(prepared_weight_cache=False)` or adjust
`prepared_weight_cache_gib`. This does not change parameter initialization,
sampling, kernel selection or VAE loading. First-load publication adds disk
work; the benefit is reuse on subsequent loads.

Validation uses CPU checkpoint roundtrips, exact output comparisons, RNG
checks, malformed-entry cases and storage-failure fallback. A CPU fixture
loading comparison is not a full H3 startup or video-quality benchmark.
