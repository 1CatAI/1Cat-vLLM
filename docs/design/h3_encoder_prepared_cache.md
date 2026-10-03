# H3 encoder prepared weights

The H3 text encoder automatically reuses completed CPU weight snapshots.
Checkpoint identity, source code, Torch version, dtype and rank topology
separate entries. Tensor shapes, dtypes, strides and aliases are validated
before reuse. Publication checksums the bytes; unchanged file identities
allow subsequent loads to avoid rereading them. Changed files are checked
again and damaged or unfinished entries are rebuilt.

Snapshots use private mappings, so runtime writes cannot modify the reusable
weights. Active entries retain a lease and are not evicted. The cache has a
128 GiB default capacity and keeps 1 GiB free. Storage failures retain the
ordinary checkpoint-loading path; startup logs explain misses and reuse.

Use `vllm video ... --disable-prepared-weight-cache` to disable this cache.
Programmatic callers can set `H3Config(prepared_weight_cache=False)` or adjust
`prepared_weight_cache_gib`. This does not change parameter initialization,
sampling, kernel selection or VAE loading. First-load publication adds disk
work; the benefit is reuse on subsequent loads.

Validation uses CPU checkpoint roundtrips, exact output comparisons, RNG
checks, malformed-entry cases and storage-failure fallback. A CPU fixture
loading comparison is not a full H3 startup or video-quality benchmark.
