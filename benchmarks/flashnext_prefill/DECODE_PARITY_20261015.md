# Flash-Next TP4 verifier and memory qualification

The destination is four V100-SXM2-16GB GPUs with all-pair NV2 links, CUDA
12.8, driver 570.211.01 and Torch 2.10.0+cu128. The model is
SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF IQ3_S with stock MTP4.
The total request capacity is 262144 tokens. Sampling is temperature 0.6,
top-p 0.95 and top-k 20; measured requests use seed 20261014 and at most
192 output tokens. C1, FULL M5, greedy draft sampling, prefix caching and
tool calling stay enabled. Expert placement remains 80% device / 20% host.
Authoritative QSA KV is host E4M3; compressed indexer history is on GPU.

## Acceptance metric

Verifier cost starts at the earliest target graph GPU start and ends at the
latest target state-commit completion across all four ranks. It includes
the target graph and following state work. Acceptance length and emitted
tokens/s are separate metrics. Kernel service sums and node-trace times
are attribution evidence, not accepted wall latency.

The ordinary-worker control has all per-step CUDA fences disabled. Low
overhead graph-envelope traces preserve every complete committed round.
Pipeline-end work is separately inventoried from graph and commit counts;
missing target starts are retained as conservative bounds, never omitted.

| Actual input tokens | Committed rounds | Mean ms | p50 ms | p90 ms | p99 ms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32774 | 520 | 15.894 | 14.647 | 18.675 | 31.449 |
| 65604 | 576 | 16.259 | 15.119 | 18.562 | 31.459 |
| 131124 | 680 | 16.366 | 15.593 | 17.068 | 30.765 |
| 261044 | 704 | 17.545 | 17.028 | 18.283 | 28.707 |

Control source is 760dcbcb8b. Every one of 2480 windows is exact. Mean cost
grows 10.39% across this sweep, so an all-context 16 ms claim is not yet
supported. The following changes still need matched complete-model curves
and a fresh ordinary-worker production restart before API promotion.

## Bounded resources and exact selector candidate

- HC prefill residuals belong to their TP row owners. Destination operator,
  recurrent reference and real model-adapter padding/graph checks pass.
  At 8192 rows the stage's temporary peak falls from 502.28 to 170.02 MiB.
  The optional peer-normalization bank stays disabled.
- A single prefill staging slot reuses the existing 128 MiB E4M3 budget for
  262144 tokens. Its lifetime ends after the owning attention use; side
  prefetch completion is preserved. FP16 draft capacity remains 131072.
- `qsa_host_kv_trim_warmup_cache=true` returns unused ordinary allocator
  pages once after all model, graph and auxiliary warmups. Actual host-QSA
  consumers are required; active tensors and graph pools remain owned.
  No inference-time garbage collection is added.
- `sm70_sparse.qsa_decode_partition_topk=true` reuses the normal exact
  FP32 selector over 8192-column partitions, then selects their canonical
  candidate union. Admission is M5/M10, one request, SM70 and score capacity
  32768..69632 compressed columns. Other inputs retain the qualified
  selector. Exact FP32 values, lower-index tie ordering, output guards and
  changed-input graph replay pass the destination operator suite.

All candidate flags default off in the library. No private kernel DSO,
preload or shared-venv modification is required. Native sources and the
qualified native artifact are unchanged by these changes.

## Startup host pressure

The first partition/trim candidate launch was killed by systemd-oomd during
expert loading: parent user-service memory PSI exceeded 50% for 20 seconds.
The recorded first unit peak was 40.2 GiB plus 2.4 GiB swap. Another attempt
also exceeded the pressure threshold. These were host-pressure failures,
not GPU allocation failures. The separate 5 GiB MemAvailable guard did not
fire; its threshold and the system OOM policy remain unchanged.

`sm70_gguf.expert_load_prefetch_banks=1` bounds queued pinned expert banks
during initialization. The validated range is 1..4 and the library default
remains 4. It changes neither weights nor placement nor runtime computation,
and therefore does not perturb the computation hash. The loader may also
hold the bank currently being consumed and a whole-bank unpacking scratch;
this setting is a queue bound, not a total pinned-memory quota. Tests cover
queue bounds, ordered values, configuration delivery and hash invariance.
Fresh startup and complete-model memory qualification remain required.

The zero-swap attempt finishes GPU warmup but triggers the unchanged 5 GiB
MemAvailable guard at 5067 MiB. The subsequent owned-service 2 GiB swap
budget completes startup and the paired tools, arithmetic and natural-EOS
quality gates through 260977 tokens. Its median cold 32K prefill is 5170.79
tokens/s and 6.69260 s. These results do not qualify an every-round 16 ms
claim. The allocator trim briefly leaves approximately 1 GiB driver free,
but inference grows ordinary reservations again: a measured boundary has
only 113.94 MiB free despite about 684 MiB of estimated live headroom.
Diagnostic libc trimming releases only about 170 MiB across all ranks and
initialization phases; it is not promoted as a material runtime fix.

`qsa_host_kv_allocator_headroom_mib` is a separate, default-zero resource
candidate. After all warmups, an actual host-QSA consumer may request a
CUDA allocator ceiling beyond the measured non-Torch use. The platform
returns unused ordinary pages, retains captured pools and active tensors,
and never raises an existing tighter process ceiling. Impossible retained
budgets are explicitly rejected. The ceiling is fixed for that worker
process; later native allocations or foreign GPU use can reduce the physical
margin, so sampled destination peaks remain required. It neither trims each
inference step nor changes model computation/hash. Other platforms reject
the requested allocator operation; models without a host-QSA consumer do
not apply it. CPU admission/hash tests pass; GPU graph/cache-pressure and
matched model/headroom/performance qualification are pending.

## HC prefill admission under MTP

Both allocator384 and512 candidates fail actual prefill while still leaving
physical bytes free. The cached fallback requests144/160MiB full residual
temporaries. Inspection finds that the HC row-owner admission additionally
requires FP16 reduced-precision reductions to be disabled. The established
MTP decode policy retains them, so requesting HC sequence prefill alone did
not select the measured low-memory path. The earlier whole-model result is
not evidence that row ownership executed; its stage tests explicitly set
the different reduction policy.

The repair preserves the existing process GEMM policy and MTP packed routes.
HC prefill partials communicate in FP32 for attention's single contribution
as well as MoE's separate contributions, then retain the existing FP16
combine/projection interface. Admission may therefore preserve either
qualified reduction setting; FP16 accumulation and high-level captured
prefill remain excluded. A route log distinguishes actual execution from
the requested flag. Other model/shape/topology guards remain in place.

The true-reduction FP16 transport screen fails the original48-step factor
allowance at8192 rows and is retained as failed. FP32 transport passes the
unchanged independent-reference and48-step allowances at7344/8192 on all
four ranks. Model-adapter padding, changed-input capture and compilation
checks pass4096/4097/7344/7345/8192 with the actual MTP reduction setting.
CPU admission/owner tests pass43 cases. Whole-model quality, selected route,
memory and verifier curves remain required before promoting this repair.

## Rejected screens

Warp-aggregated histograms regress random inputs. Warp-local emission
improves too little to justify a native change. An exact integer-key final
merge is slower than the existing FP32 selector. These standalone research
screens are not production providers. The partitioned FP32 candidate is
the only promoted selector experiment; model promotion remains pending.
