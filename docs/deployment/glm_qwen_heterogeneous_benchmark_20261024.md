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
