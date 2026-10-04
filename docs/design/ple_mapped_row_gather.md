# Native CPU reads for disk-mapped PLE rows

The C1 PLE phase trace identifies mmap gathering as the largest CPU lookup
phase: median 1.610 ms across 1196 single-token requests. Key computation is
0.144 ms; result fan-out and four flag publications take 0.089 ms. These are
instrumented CPU intervals, not GPU critical-path wait measurements.

The new CPU operator resolves every requested row, sorts and deduplicates
only its covering pages, advises read-ahead for cold pages, then copies the
original bytes directly into the result buffer. It retains all checkpoint
shards as mmap. It never allocates a whole-table copy or prefaults the whole
mapping. Existing page-release policy, output publication and consumer waits
remain in their existing order. Dense weights, activations and accumulation
precision are unaffected; existing quantized embedding bytes are unchanged.

`MappedRowGatherKernel` lives in the PLE kernel framework and is selected by
Linux/file-mapping/operator capability. Startup verifies actual mapped row
bytes and records warm reference/candidate timings without consuming RNG.
`KernelConfig.ple_disk_row_gather` defaults on; it is CPU I/O policy and does
not change the compiled model hash. No environment variable, architecture,
tensor-parallel, model-name or batch-width restriction is added. The native
operator validates indices and output geometry before writing results.

The normal `_C` extension ships the CPU registration. A missing operator or
failed startup check retains the existing reader. Both whole-table output
gather and cascade disk-tier reads consult the provider. CPU-reader admission
is included in the offload-worker READY message and startup logs.

Research-only real-row probes use 16 retained row sets, nine alternating
repetitions, and byte comparisons. Cold arms evict only selected file pages;
warm arms retain them. The table stays file-backed in both arms.

| CPU lookup | Warm us/case | Cold us/case |
| --- | ---: | ---: |
| Serial ctypes row copies | 28.27 | 1446.62 |
| Native cold-page advice and gather | 53.95 | 282.08 |

The native research measurement includes creating its ID and output tensors;
the production path writes the existing output directly. Cold controls
average 16.69 major faults/case; native prefetch initiates I/O ahead of copies.
All compared row bytes match. These numbers do not establish a model speedup.

Fourteen research C++ correctness cases cover M=1/5/17/33, 17/160/8193-byte
rows, shard/page boundaries, repeated IDs, buffer reuse and rejection of
invalid or strided IDs. Installed-artifact dispatch, matched C1 distributions,
fixed-input step timing, the natural quality suite and a short C4 smoke remain
required before default promotion. UVA hot-row caching remains a separate
follow-up: the current 8192-row trace simulation has only 14.5% all-row hits,
so its miss path must be efficient as well.
