# GLM 5.3 Flash and Qwen3.8 Flash Next on heterogeneous GPUs

This branch carries the seven-GPU GLM GGUF/MTP/multimodal work and subsequent
GLM-specific optimizations. It also records a Qwen deployment finding on the
same source revision. The two models need separate launch policies: the GLM
service retains its accepted settings, while Qwen disables PP mixed-prefill
latency control with `--mixed-prefill-step-latency-ms 0`.

## Implementation highlights

- The heterogeneous PP path assigns model layers to GPUs with different memory
  and compute capacities. It includes PP decode batching, static activation
  transfer, packed feedback and reset guards. These paths have their own
  feature flags and regression coverage.
- GGUF model loading, native MTP draft handling, and multimodal assets are
  integrated for the GLM and Qwen deployments. MTP draft inputs are checked
  against the restored main-model embedding.
- GLM-specific sparse attention, indexer precision, tail-cache replay,
  prefill routing, and expert MMQ dispatch were refined with opt-in paths and
  shape/capability guards. The packed independent-row expert-down path is
  calibrated before dispatch, preserving a fallback for other shapes.
- Bounded tracing covers scheduler state, PP execution, cache checkpoints and
  kernel dispatch. See `docs/deployment/heterogeneous_pp7.md`,
  `docs/deployment/pp_transport_diagnostics.md`, and
  `docs/design/mixed_prefill_latency.md` for the existing interfaces.

These changes are not a universal speedup claim. In particular, GLM-specific
GGUF kernel choices do not apply to every Qwen shape, and a Qwen launch
parameter should not be copied into the GLM service without a GLM A/B run.

## Measured Qwen deployment decision

Source: `219067a4dec2f9cf7f28d24015a8d9fe4020f4bc`. Each successful
configuration served the same fixed eight-lane workload: two 214,528-token
prompts generating 512 tokens each, and six 2,048-token prompts generating
1,024 tokens each. Each round generated 7,168 tokens. After MTP shape warmup,
we ran one prime round and two warm rounds. Values below are means of the two
warm rounds. Shared decode is aggregate tokens/s during the interval when all
eight lanes were decoding, not single-request speed.

| Configuration | Warm round | Shared decode | Result |
| --- | ---: | ---: | --- |
| Historical source `ac9523a1`, TP1 x PP7 | 42.264 s | 257.800 token/s | 8/8 complete |
| Current source, TP1 x PP7, default mixed prefill | 83.947 s | 226.682 token/s | 8/8 complete |
| Current source, TP1 x PP7, Qwen override `0` | **41.083 s** | **286.397 token/s** | 8/8 complete |
| Current source, TP2 x PP3, Qwen override `0` | 50.849 s | 227.996 token/s | 8/8 complete |
| Current source, TP4 x PP1 | n/a | n/a | startup failed |

The Qwen override reduced warm-round time by **51.06%** and increased shared
decode throughput by **26.34%** relative to the current source's default
policy. TP2 took **23.77%** longer than optimized TP1 and had **20.39%** lower
shared decode throughput on the tested six-GPU group. All successful runs
completed the requested output lengths without preemption. Optimized TP1 and
TP2 each had about **44.16%** MTP draft acceptance in warm rounds.

The default PP mixed-prefill policy targeted 250 ms and repeatedly admitted
only 16 prompt tokens in mixed batches on this Qwen workload. The Qwen-only
override is a launch configuration change; it does not change shared source or
GLM defaults. A deployment can append:

```text
--mixed-prefill-step-latency-ms 0
```

The TP4 attempt has no throughput result. In eager mode, NCCL collective
`_ALLGATHER_BASE` sequence 27 saw 496,640 input elements on rank 0 and 10,240
on ranks 1-3, then timed out. Packed PLE layout and all-V100 runtime paths
also failed during startup. Rank-shape agreement must be repaired and retested
before TP4 can be recommended. This evidence does not establish that TP4 is
inherently slower or that a specific GPU caused the failure.

The comparison data is in
`benchmarks/heterogeneous_pp7/qwen_tp_comparison_20261024.json`. Output token
IDs differed between source versions; the performance comparison is for this
fixed workload, not a claim of token-for-token equivalence across versions.

## Deployment boundary

The accepted GLM service uses seven GPUs and was verified active with a 200
LAN health response after this benchmark. The prepared Qwen TP1 service is
inactive; both full-card services cannot run at once on this machine. GPU6 is
limited to 100 W and 950 MHz in the prepared Qwen unit. Model weights,
machine-local paths, credentials and systemd units are deployment assets and
are not included in this branch.

## Test machine

- Ubuntu 22.04, Linux 6.8; two AMD EPYC 7763 CPUs (256 logical CPUs),
  251 GiB RAM.
- Seven GPUs: four Tesla V100 32 GiB (SM70), two GeForce RTX 3090 24 GiB
  (SM86), and one GeForce RTX 4090 48 GiB (SM89). CUDA device IDs: V100
  `2,4,5,6`, 3090 `0,1`, 4090 `3`.
- Tensor parallelism 1, pipeline parallelism 7. `NCCL_P2P_DISABLE=1` on this
  host. GPU6 has a 100 W power cap and 950 MHz graphics clock cap for thermal
  reasons.
- Qwen PP layer split in CUDA order `2,0,1,4,5,6,3`: `8,5,5,8,8,2,12`.
  GLM PP split in order `2,0,1,6,4,5,3`: `9,4,4,5,7,7,9`.
- Qwen GGUF Q4_K_XL, 215040 max tokens, eight sequences, MTP4 with a prepared
  draft, multimodal projection. GLM GGUF IQ4_XS, 163840 max tokens, three
  sequences, MTP4 and multimodal projection.
- Models share all seven GPUs and run as mutually exclusive systemd services.

## Implementation

- Seven-stage heterogeneous PP with explicit layer partitions and
  architecture-specific native extensions.
- GGUF loading and model-specific adapters for Qwen and GLM, multimodal asset
  handling, and MTP draft support.
- PP batching/feedback and prefill/cache diagnostics, with opt-in GLM sparse
  and expert kernels.
- Qwen launch override `--mixed-prefill-step-latency-ms 0`; GLM retains its own scheduling policy.

## Validation on the merged branch

Commit `2d9737c0f7ad331cff272ede78a75a0c1649a3b5` merges upstream
`6a143197e6e5f6fa77bd483df0c5a09dcea6aa20`. There were no native-source
changes in that merge; 16 existing native extensions were reused for GPU
validation.

- 217 targeted tests passed across KV cache, GGUF TP slicing, SM70 routing,
  MRoPE, DCP, TurboQuant and FlashNext cost accounting.
- Qwen TP1×PP7: three eight-lane rounds completed 8/8, with two 214528-token
  prompts plus six 2048-token prompts. The two warm rounds averaged 43.255 s
  and 265.550 aggregate shared-decode token/s, with pooled MTP acceptance
  43.88% and no preemptions. Relative to the two pre-merge warm rounds, wall
  time was 5.29% higher and shared decode 7.28% lower. Two samples are
  insufficient to establish a stable regression.
- Qwen short MTP, three-way concurrency, and two-image API requests passed. A
  repeated isolated 3000-token prefix request did not increment cache-hit
  metrics; this assertion is shorter than two 1616-token cache pages and should
  not be interpreted as a merge regression. Long-prefix warm-round cache hits
  stayed at 423392 tokens per round before and after merging.
- GLM TP1×PP7: API startup, 29 startup-warmup cases, short MTP, three
  concurrent requests, a two-image API request, and 160000 input plus 32 output
  tokens passed. The long request took 184.516 s. Visual answer semantics were
  not assessed.

The deployed services use machine-local weights and binaries, which are not
part of the repository. The merged source was tested with these assets; the
full repository test suite and fresh native rebuild were not run.

## Service promotion on the test machine

Both mutually exclusive systemd services were switched to the validated merged
source commit `2d9737c0f`. The original service manifests and source target were
backed up locally before the change. Qwen started with seven PP stages; a
64-token text completion accepted 45 MTP draft tokens, and a two-image chat
request generated 30 tokens. GLM then started, completed its 29-case startup
warmup, accepted 50 MTP draft tokens in a 64-token text completion, and a
two-image chat request generated 32 tokens. These are API smoke checks; they do
not assess visual-answer semantics or replace the longer GPU regression above.

After promotion, `glm-pp7.service` was active, `qwen-pp7.service` inactive and
ready for switching, and the LAN proxy returned HTTP 200 for `/health` and
`/v1/models` with `glm-5.3-flash`. The two full-card services cannot run
simultaneously on the same seven GPUs.
