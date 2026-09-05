# Qwen3.8 Flash Next PLE disk offload on SM70

## Decision

Qwen3.8 Flash Next's PLE is a learned trigram embedding, not prompt-lookup
speculation. The NVFP4 checkpoint stores 47.684 GiB of PLE E4M3 rows in 128
tensors across ten safetensor files. Keeping the table in an independent CPU
process is fast, but consumes about 48 GiB of host memory.

This change adds an experimental, default-off disk-backed mode. It retains the
safetensors mappings and gathers rows directly from them without creating the
complete anonymous host table. The mode is useful when host capacity matters.
Its prefill path now deduplicates and sorts rows globally and drives independent
checkpoint shards concurrently; decode still needs a bounded hot-row cache
before this can become a performance default.

Enable it only together with the existing CPU service:

```bash
VLLM_PLE_CPU_OFFLOAD=1 \
VLLM_PLE_DISK_OFFLOAD=1 \
VLLM_PLE_DISK_OFFLOAD_NUM_THREADS=32 \
vllm serve RadixArk/Qwen3.8-Flash-Next-NVFP4 ...
```

Zero workers selects the bounded default of `min(32, os.cpu_count())`.

`VLLM_PLE_DISK_OFFLOAD_PROFILE=1` additionally logs lookup wall time and Linux
minor/major fault counts. It is intended for diagnosis rather than production.

## Implementation and safety properties

- The PLE worker constructs the logical embedding on `meta`; it does not
  allocate the 47.684-GiB anonymous parameter.
- The loader accepts only checkpoint-native FP8 CPU tensors whose addresses
  belong to real file mappings. Eager copies and missing shards fail closed.
- All 128 tensors are retained after loading. The small global scale is copied
  so it does not depend on a transient loader mapping.
- `MADV_RANDOM` prevents Linux mmap read-around from pulling almost the entire
  table into page cache after a sparse lookup.
- Lookup globally sorts and deduplicates logical row IDs, divides them by
  checkpoint shard, gathers independent shard segments through a persistent
  thread pool, then restores original token/head order into the pinned output.
- Disk mode without `VLLM_PLE_CPU_OFFLOAD=1` is rejected before model loading.

The feature changes no default. The normal CPU-resident PLE path is untouched.

The shard discovery/reload coverage in upstream vLLM
[#54129](https://github.com/vllm-project/vllm/pull/54129) is broader. This SM70
variant adopts its useful `np.unique` and cross-shard worker-pool strategy, but
keeps 1Cat's existing single PLE CPU process: one lookup is computed per DP
rank and fanned out to all four TP ranks. Directly porting #54129 would instead
run the host gather independently in every TP model process. `MADV_RANDOM` is
also retained here because the V100 host's normal mmap advice pulled 378.5 MiB
of a 400-MiB shard for a sparse 1,024-row lookup.

## Validation contract

The full-model measurements used:

- `RadixArk/Qwen3.8-Flash-Next-NVFP4`;
- four Tesla V100-SXM2-32GB GPUs, TP4, V2 model runner, no MTP;
- FP16 activations and KV cache, 262,144 maximum length;
- chunked prefill with 8,192 scheduled tokens and prefix caching disabled;
- current Flash-V100 QSA page4, FlashQLA GDN, and grouped NVFP4 MoE prefill;
- PyTorch 2.10.0+cu128 and the current 12-argument Flash-V100 extension.

The storage volume is ext4 on a RAID0 pair of Samsung NVMe drives. Read-only
`fio` screens measured about 47.7 MiB/s and 12.2K IOPS for 4-KiB QD1 random
reads, 186 MiB/s and 47.7K IOPS at QD32, and 6,775 MiB/s for 1-MiB sequential
reads at QD32.

### Memory and loading

A standalone real-checkpoint smoke retained all 47.684 GiB in ten file mappings
with 4.546 GiB peak RSS. Loading took 53.77 seconds from the colder run; a
subsequent cached full-model run discovered and retained the mappings in about
7 seconds. The earlier anonymous-table service used about 45--50 GiB RSS and
historically took about 126 seconds to copy and prefault the table.

The virtual mapping size is still 47.684 GiB. It is not resident memory. During
the cold random 256K run, clean file cache grew by only about 1.8 GiB because
random pages remained reclaimable under host pressure.

### Quality

The normal-prompt TP4 gate completed through 256K. Its two deterministic quality
outputs exactly matched the RAM baseline token hashes:

- arithmetic: `42<|im_end|>`;
- Chinese: `在标准大气压下，水的沸点是100摄氏度。<|im_end|>`.

The mapped-shard unit gate also compares gathered E4M3 bytes exactly and rejects
an incomplete 128-shard table. The pooled implementation additionally exercised
duplicate IDs spanning two shards. The three full-model random-input cases
produced the same output token IDs and hashes as the resident-RAM path.

## Prefill results

### Repeated prompt, warm page cache

The established benchmark repeats one short English sentence to reach each
length. Consequently it touches the same small n-gram working set after the
first chunk. Disk mmap then behaves like a reclaimable RAM mapping:

| Input | RAM PLE | Disk mmap | Throughput change |
| ---: | ---: | ---: | ---: |
| 8,192 | 7,056.26 tok/s | 7,026.03 tok/s | -0.43% |
| 32,768 | 6,603.30 tok/s | 6,332.56 tok/s | -4.10% |
| 65,536 | 6,327.43 tok/s | 6,192.32 tok/s | -2.14% |
| 131,000 | 5,927.33 tok/s | 5,764.59 tok/s | -2.75% |
| 262,143 | 5,248.63 tok/s | 5,102.62 tok/s | -2.78% |

Per-8K PLE lookup was normally 13--26 ms with zero major faults. Average GPU
utilization in the 256K request was 98.8%.

### Unique random tokens, cache dropped before every request

A second gate generated only valid non-control vocabulary IDs and had 262,141
unique trigrams in the 256K request. It synchronously evicted all ten PLE files
after engine initialization and immediately before each measured request.

An additional resident-RAM run used exactly the same prompt hashes. This
corrects the earlier, invalid comparison between random mmap input and the
highly repetitive RAM benchmark.

| Input | Resident RAM | Serial mmap | 32-worker mmap | Gain vs serial |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 2,465.38 tok/s | 2,627.33 tok/s | 4,869.10 tok/s | +85.3% |
| 32,768 | 2,412.51 tok/s | 2,171.23 tok/s | 4,771.18 tok/s | +119.8% |
| 262,143 | 2,496.81 tok/s | 2,382.35 tok/s | 4,044.35 tok/s | +69.8% |

The corresponding 256K prefill times were 104.991, 110.036, and 64.817
seconds. The pooled mmap route beats the current RAM implementation because
the latter performs an unsorted single `torch.index_select` over the resident
47.684-GiB table; it is a same-prompt control, not an optimized DRAM ceiling.

| Input | Serial mmap lookup | 32-worker lookup | Speedup | Major faults, pooled |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 1.853 s | 0.427 s | 4.34x | 135,298 |
| 32,768 | 9.346 s | 1.731 s | 5.40x | 527,484 |
| 262,143 | 57.052 s | 11.645 s | 4.90x | 3,483,396 |

At 256K, average GPU utilization rose from 48.8% to 82.3% and average board
power from 157.8 W to 218.3 W. Pooled chunk lookup started at 0.427 seconds and
settled around 0.31--0.40 seconds. The lower long-run fault count comes from
deduplication and bounded page-cache reuse; the first 8K case has effectively
the same 135K cold faults before and after, so its 4.34x lookup gain is genuine
I/O concurrency rather than a warmer cache.

## Hardware limit versus optimization headroom

The old implementation reached the hardware random-IOPS wall through a poor
submission pattern, not an unavoidable model limit. It sorted IDs, but then
visited 128 shards serially; each shard received only about 1,024 rows in an 8K
chunk, too little for PyTorch's intra-op pool to create useful queue depth.

A one-shard cold screen with the same 28,775 major faults took 4.321 seconds at
one worker and 0.173 seconds at 24 workers. The all-shard 8K screen fell from
3.442 seconds with serial shard dispatch to 0.583 seconds with 32 workers. The
full model then confirmed 1.853 to 0.427 seconds for the first cold 8K lookup
while fault count stayed constant. This is direct evidence of software
headroom above the original QD1-like path.

The volume is still bounded by random I/O: 4-KiB QD32 `fio` measured 47.7K IOPS
and 186 MiB/s, while sequential bandwidth is about 6.8 GiB/s. Larger sorted
windows remain useful (a global 256K screen took 15.864 seconds instead of the
old 57.052-second sum), but cross-shard queue depth captures most of that gain
without changing vLLM's 8K scheduler chunk.

## Next optimization sequence

1. Add two bounded 8K or 32K output buffers. The model runner already retains
   full prompt token IDs, but the current connector sends only the active 8K
   batch. Prepare window N+1 while the GPUs execute N, validate request/range
   keys before reuse, and fall back synchronously on a scheduling mismatch.
2. Pin the PLE worker and its I/O pool deliberately by NUMA topology. GPUs 4--7
   are local to NUMA node 1; compare 32 local workers with a bounded cross-node
   pool without stealing CPU from the separate GPUs 0--3 service.
3. Replace synchronous mmap faults with per-file batched asynchronous reads.
   Sort and deduplicate page offsets, coalesce adjacent pages, use fixed output
   buffers, and scatter rows only after I/O completion. Do not sequentially
   populate the complete 47.684-GiB mapping.
4. Apply global deduplication/sorting to the resident-RAM implementation too;
   its 2.50K random-input result is not a useful DRAM performance ceiling.
5. Add a bounded hot-page or hot-row cache for natural-language repetition.
   Keep its memory budget explicit so disk mode still saves tens of GiB.
6. Treat decode separately. A cold 16-row token lookup can take 2--20 ms, which
   can consume or exceed a 100-tok/s latency budget. Decode needs a rolling cache
   or lookahead before this mode can be recommended for low-latency serving.

The 32-worker run spends 11.645 seconds in PLE gather during a 64.817-second
256K prefill. Ideal lookahead would reduce the exposed total toward 53 seconds,
or about 4.9K tok/s. Reaching 6K and 8K requires totals below 43.7 and 32.8
seconds respectively, so after I/O overlap the remaining work belongs to the
GPU prefill path rather than the disk table.

## Promotion gate

Keep disk offload opt-in until all of the following hold:

- normal-prompt output tokens and hashes match the RAM path;
- cold, diverse 256K prefill keeps average GPU utilization above 90%;
- 256K disk-offload prefill is within 10% of the same-prompt RAM control;
- PLE worker RSS plus its bounded cache stays within the documented budget;
- cancellation, multiple requests, prefix reuse, and decode do not consume a
  stale prefetched buffer.
