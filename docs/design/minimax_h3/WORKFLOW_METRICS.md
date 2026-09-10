# H3 workflow performance measurements

The native engine records the actual video/audio sigma sequences, selected
attention backends, useful unpadded sequence length and completed block calls.
The evaluator derives denoiser calls from the runtime intervals. LightX2V's
5/9 API points and FlashGen/FastH3's four-interval API therefore share the same
accounting without a hardcoded 49-call assumption. Normal request validation
still enforces each adapter's legal tasks and sampling settings.

Each rank reports `denoise_steps` with useful FLOPs, completed DiT calls,
executed blocks, CUDA event time and CPU enqueue time. CUDA event spans include
dependent communication, stream waits and host feeding gaps; they are not sums
of individual kernel execution times. No per-step synchronization is added.
GEMM/attention/communication service attribution still requires a separate
profiler run. The complete synchronized denoise wall time remains the throughput
denominator, using the slowest rank for every rank's numerator.

For a complete measurement, reuse a native run's `config` and `request` JSON:

```bash
.venv/bin/python -m vllm.video.benchmark \
  --contract h3-contract.json --output h3-measurements-new
```

Run this command without a profiler. It acquires the native GPU lease, uses one
persistent engine, generates one complete warmup video/audio request and three
consecutive measured requests, and saves all four results plus
`performance.json`. The output directory must be new. NVML records memory,
power, clocks, temperature, utilization and running processes alongside each
request. Source revision/file hashes, Torch/CUDA versions and actual loaded H3
kernel paths/SHA256 hashes accompany the results. Deployment and request JSON
retain model revision, checkpoint/adapter identifiers, sampling and TP settings.

The evaluator requires a complete warmup from the same engine session, exact
configuration/shape consistency, all TP ranks and all scheduled steps. Missing
blocks, inconsistent per-step/per-layer/total work, profiled runs and quality
captures are rejected. For existing result files:

```python
import json
from pathlib import Path
from vllm.video.metrics import evaluate_performance

root = Path("h3-measurements-new")
warmup = json.loads((root / "warmup/run.json").read_text())
runs = [json.loads((root / f"run-{i}/run.json").read_text()) for i in (1, 2, 3)]
report = evaluate_performance(runs, warmup=warmup)
```

Passing performance means every rank's three-run median is strictly above
80 useful TFLOP/s and complete-denoise CV is at most 5%. Peak memory (30 GiB
allocated/card budget), end-to-end latency and quality status are reported
separately. The selected shape and TP size remain in the report; passing a
shorter workload or TP1 does not complete the 243-frame TP4 or 15-second cases.
Quality must separately pass the accepted numerical and audiovisual review.

This checkpoint counts **dense uncached execution**. It records zero sparse
blocks/cache hits for those actual routes and rejects sparse/cache descriptors
until the corresponding execution counter exists. This is not VSA, TeaCache or
Cache-DiT support. Padding, ConvRot, dequantization and repeated output rows remain
excluded from useful FLOPs. Algorithmic work savings must be accounted separately
when those variants are implemented.

Column-parallel LoRA A projections have the same weights and input on every
rank. `dense_tp_lora_v2` attributes those input rows once across the TP group,
including uneven tails. Identical replicas are reported in
`redundant_denoise_flops`, `redundant_flops_by_layer` and each step's
`redundant_flops`; they never increase useful throughput. Row-parallel A consumes
distinct input shards, so its resulting partial B products remain useful work.
Legacy records without an accounting version are rejected until explicitly
audited; the measurement source/time must remain unchanged in any such audit.

## Development evidence

Integration base: `4f19ef7a20db60bb0685e599bd3f4dd156202eed` (`onecat/main`).
Owned branch: `codex/v100-h3-workflow-metrics-20260909-031302`.
Artifacts: `/data/minimax-h3/sm70-general-20260909/`.

- `metrics-regressions-v2.log`: 81 CPU acceptance/service/workflow/API checks
  pass. These include four/eight/base/DMD2 schedule semantics, TP1/2/4 and
  rejection of incomplete warmup, missing steps and inconsistent work counts.
- `metrics-block-gpu-v2.log`: real SM70 DiT block test passes, with an independent
  FLOP formula that excludes suffix padding, two measured calls,
  bitwise output preservation and hook cleanup before another request. The first
  GPU5 attempt was rejected by an existing lease and ran no GPU test; GPU0 was
  acquired after the original-weight control released it.
- Full model instrumentation validation and performance measurements are pending.
  No configuration has met the campaign's >80 TFLOP/s and full quality gates.

### Complete four-step controls and first three-run baseline

`metrics-720p-quality` validates runtime `82362a4312`, W8A16 + LightX2V4 v1.2,
TP4 GPUs0-3, internal 1280x736/124 frames for the five-second 720p sample.
All four ranks recorded four complete calls and 52 blocks/call. Full video/audio
latents and decoded RGB/PCM match frozen mainline bitwise; SSIM is 1.0 and the
audio numerical gates pass (`metrics-quality.json`). This is instrumentation
regression evidence, not independent official or human quality acceptance.

`fa-720p-three-runs` completes one full native warmup and three unprofiled
requests with the same deployment/sampling on `82362a4312`. Denoise times are
65.880184, 65.898529 and 65.965556 seconds; CV is 0.055668%. End-to-end times
are 93.173206, 95.494415 and 88.343595 seconds. Peak allocation is
19,501,498,880 bytes/card. Warmup alone also retained fresh encoder/denoise input
tensors for later independent diagnostics; measured requests contain no captures.

The original counter included identical column-A replicas. Its reported
47.524847 TFLOP/s/card is superseded by the explicit header-shape audit in
`fa-720p-three-runs/audited-counts/`. Original files, times and source fingerprints
are retained. Rank0 useful work is 3,103,284,010,387,456 FLOPs after excluding
28,533,508,276,224 replicated A FLOPs (0.9111% of the old numerator).
Audited median throughput is **47.091839–47.091855 TFLOP/s/card**, depending on
the uneven row tail. Performance remains below 80. The audit script checks all
retained per-layer counts against immutable LightX2V A/B header shapes; it is
not a new run of the revised runtime counter.

`metrics-lora-count-cpu.log`: 49 CPU tests pass, including unique logical column
adapter FLOPs across TP1/2/4 and one-row/97-row/34551-row tails. Two additional
legacy/inconsistent-redundancy rejection cases pass in
`metrics-legacy-rejection.log`. The revised counter will be exercised in the
matching FlashInfer full measurements. Sparse/cache and complete official
workflow/quality/performance coverage remain open.
