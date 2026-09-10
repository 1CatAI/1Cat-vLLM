# FlashInfer-derived GDN / HC layer-fusion campaign

## Purpose

Finish operator adaptation and evidence first, then run one consolidated
end-to-end comparison. Do not launch another full model per operator. Maintain
the same no-MTP FlashNext workload, output-quality gates and fixed-70
concurrency targets (C4/C8/C16 aggregate 238/420/728 tok/s).

Integration base: `onecat/main` at `755baae1d075ee04fa9096b23fc0225b23589a86`.
Owned branch/worktree: `codex/v100-flashinfer-gdn-conv-20260905-173007` /
`worktrees/v100-flashinfer-gdn-conv-20260905-173007`.
Related QSA prototype is Draft #513; prior batch HC work is Draft #504.
This change does not duplicate their QSA or TP-sharding implementations.
The M1-only #506/#510 work remains separate and is not overwritten.

Source pin: FlashInfer `6c14bbd5ff34210404d5d4b5f6ff3b4b2527f59f`.
The CUDA code is genuinely derived from its kernels, not a FlashInfer API
redirect to Triton. The adapters are benchmark-only until gates pass.

## GDN adaptation

Derived from `gdn_kernels/experimental/kernel/gdn_fused_decode_sm120.cu`:
gate B/A projection, width-4 causal convolution, Q/K normalization, gating,
FP32 delta-rule state update and attention output in one kernel. FP16
activations/weights and FP32 recurrent state remain unchanged. Geometry is
JIT-specialized, not identified by checkpoint name or global server settings.

The actual FlashNext TP4 shard is hidden=2560, Hq=4, Hv=12, D=128,
QKV-width=2560 and BA-width=24. Do not reuse an old Hq4/Hv8 GDN result as this
model's baseline. Reference QLA source SHA256:
`fd6389cef9f1b38df7e122e582221d74d9ae1fba377ac3bec0da047fd3d30af8`.

Load-bearing integration changes:

- Production conv PTX emits `mul.f16`, then `cvt.f32.f16`. Preserve the FP16
  product boundary and ordered FP32 accumulation, not upstream BF16/FP32
  widened multiplication. Keep the SiLU-to-FP16 materialization.
- Preserve B/A projection's FP16 materialization before FP32 sigmoid/softplus.
- Preserve in-place state aliasing and V-major `[pool,Hv,V,K]` layout,
  explicit pool strides, DS/SD conv layouts and strided QKV projection views.
- Negative padding owns no state and emits zero. Valid slots must be unique
  within a call, in range, and owned by the scheduler; this prototype is not
  a replacement for prefix-cache copy-on-write / metadata validation.
- Use cooperative launch and CUDA grid synchronization instead of relying on
  a software spin barrier's regular-launch residency assumption in a runtime
  with auxiliary streams. Cooperative graphs/stream capture are supported
  by CUDA; see [NVIDIA's CUDA 11 description](https://developer.nvidia.com/blog/cuda-11-features-revealed/).
  Runtime capability and exact occupancy are still checked before launch.

## HC adaptation

The fused combine/Gemma-norm adapter derives from FlashInfer's
`FusedAddRMSNormKernel` vector IO, shared staging and warp/block reduction.
It adds HC's injection gate and per-branch/shared affine addressing. The
materialized residual is rounded before RMS statistics, and Gemma affine is
`fma(y, w, y)`, not an unqualified ordinary RMSNorm replacement. Single-lane
shared reduction writes avoid redundant same-address writes.

FP16/FP32 residual and block types are distinct template parameters. Weight
and injection dtypes match; no FP8/int8/QPN approximation is introduced.
The initial component accepts vector-aligned contiguous matrices with
group width a multiple of 8, <=4096, and 1/2/4/8 warps for screening. No
production settings or server max-seqs/TP/prefill limits are changed.

## Existing negative evidence not to repeat

- Previous v0.6.13 standalone GDN-input GEMV replacement reached parity,
  and M1 HC down/up replacement regressed. A route hit is not a gain.
- Batch GDN projection concatenation/overlap and generic cuBLASLt retuning
  already failed their integration thresholds.
- FP16 recurrent-state compression caused large state/output error in the
  prior batch audit and is not included.
- HC projections plus communication already have an isolated #504 candidate;
  reuse its validated pieces instead of reimplementing or counting them twice.

## Test plan and progress

Environment: Python 3.12.13, Torch 2.10.0+cu128, CUDA 12.8, native SM70,
private Torch/Triton caches. No service/model engine is launched at this stage.

1. Native CPU compilation and load, current-library/geometry provenance.
2. Independent GDN oracle; dynamic state slots, live slot zero, padding,
   poisoned outputs, strided input/state layout and graph/eager equality.
3. Checkpoint-weight GDN component against actual production conv+FlashQLA;
   separate conv/state/output errors and preserve reference trajectories.
   An identical input-refresh copy is timed in both arms because production
   conv mutates its input; QKV/Z projection itself is excluded.
4. HC full combine/norm versus current Triton, FP16/FP32 residuals, B1/4/8/16.
5. Memcheck/racecheck/synccheck and longer independent recurrent histories
   before runtime admission. Micro error thresholds are screening gates,
   not proof of task-level quality non-inferiority.
6. Integrate only winners together with QSA, preserve fallback routes and
   baseline precision. Then one consolidated E2E C1/4/8/16 comparison plus
   coding/tool/schema quality checks, reporting actual routes and pure decode
   separately from prefill/TTFT. Failed operators stay off.

Native GDN and HC SM70 builds passed. Cooperative GDN builds also passed,
including Hq4/Hv12 and Hq8/Hv24 modules loaded in the same process;
per-geometry Torch namespaces prevent duplicate registration.

### GPU results, 2026-09-06

After the prior lease released GPU 0--3, the `72af224161` tests ran on locked
GPU 0: **16 passed**, including all 8 GPU cases (33.39 s, including HC build).
This covers independent GDN histories, DS/SD conv layouts, strided QKV,
padding/live slot zero, and HC shared/per-branch weights, mixed FP16/FP32,
non-power-of-two widths, poisoned outputs and graph replay.
Artifact: `.artifacts/gdn-hc-unit-gpu-v4.log`.

Checkpoint component screen: RadixArk Qwen3.8-Flash-Next-NVFP4, layer 0,
TP4 rank-0-shaped **FP16** GDN weights, FP32 state, Hq4/Hv12/D128,
synthetic changing hidden states. The target MoE quantization is not being
retested here. CUDA Graph, 9 alternating paired samples, 30 calls per graph,
identical raw-QKV refresh in both arms; no full model launch. GPU clocks use
automatic boost and were not locked, so use paired deltas, not cross-run
absolute comparisons. First screen uses 16 local steps and then 256 fully
independent state-history steps with padding and slot recycling.

| Rows | Existing BA + conv + FlashQLA, us | Fused chain, us | Latency reduction |
| --- | ---: | ---: | ---: |
| 1 | 17.237 | 11.674 | 32.28% |
| 4 | 28.604 | 15.087 | 47.26% |
| 8 | 33.041 | 22.801 | 30.99% |
| 16 | 57.344 | 47.343 | 17.44% |

All four independent-history screens pass the unchanged operator gates.
Worst output relative L2 across those histories is 2.458e-4 and worst FP32
state relative L2 is 2.636e-5; conv state updates remain exact. These errors
are not zero and this is not a task-quality admission. The M1 reference
includes separate BA projection whereas production M1 can fuse QKVZ/BA;
**do not count the M1 component delta as a production gain**.
Artifact: `.artifacts/gdn-screen-v1.log`.

Commands inside the owned GPU-lock/environment launcher (the wrapper sets
the pinned QLA binary, FlashInfer headers and task-private caches):

```bash
.venv/bin/python -m pytest -q -x --confcutdir=flashinfer-sm70/tests \
  flashinfer-sm70/tests/test_layer_fusion.py
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_flashinfer_gdn_conv \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 --steps 16
.venv/bin/python -m benchmarks.kernels.benchmark_sm70_flashinfer_hc_norm
```

Reference QLA binary SHA256:
`3982305151798be22a1dabd0140feb085f787e1e76da01f58d8256de66050975`.
GDN candidate binary SHA256:
`45cfe0090f43792ac8f4a5a21b9475e988c0f6872c874faf2e5da967db978d66`.
The 8-row-warp cooperative kernel has 122 registers/thread (M1: 110),
zero stack/local memory. These are static resources, not measured occupancy.

HC shared-staging version passed all numerical screens but **lost every
timed shape**. FP16 residual results (best of 1/2/4/8 warps):

| Rows | Existing Triton, us | Best FlashInfer-derived HC, us |
| --- | ---: | ---: |
| 1 | 3.164 | 4.321 |
| 4 | 3.052 | 3.942 |
| 8 | 2.918 | 3.717 |
| 16 | 3.144 | 3.953 |

The FP32-residual arm also regressed. Retain the existing production HC;
do not promote this version on the basis of a FlashInfer label.
Artifact: `.artifacts/hc-norm-screen-v1.log`. HC v1 binary SHA256:
`ef905abb8feb41a5887fc64dc45f11dbea97039c60095a1e17b93fbe143d079b`.

### Follow-up candidates and remaining gate

- GDN four-row-per-warp variant: B8/16 independent-history screens pass;
  paired baseline/candidate medians 36.420/25.054 and 57.344/46.353 us.
  Registers fall to 89 (M1: 78), still no spills. This is not a same-run
  eight-versus-four comparison; do not change the default based on the small
  cross-run difference. Artifact: `.artifacts/gdn-screen-r4-v1.log`.
- HC register-staging variant: retain materialized FP16/FP32 residuals in
  registers over the reduction, avoiding the shared-value write/reload.
  Local D2560, 4/8-warp specialization; all other geometries retain the
  general component. Native build passes, but GPU correctness/speed is pending.
  Artifact: `.artifacts/hc-norm-build-v2.log`; binary SHA256:
  `6087639c2f615ce04775000d556325589f391993c88653a5e3baad04bff347ff`.
- Updated CPU suite: **8 passed, 12 GPU cases skipped** with GPU hidden.
  The four additional register-HC cases have not run on GPU. Previous v1 GPU
  results must not be relabeled as v2 validation.
- The targeted GDN memcheck attempt did **not** launch: the new QUASAR E4M3
  task acquired the paper GPU 0--3 lease between component jobs. Exit 75 and
  an empty sanitizer log are not a pass. Wait for release; never preempt it.
- No runtime integration, new E2E throughput, or model-quality pass yet.
  Preserve existing projection/M1 and HC paths until complete-chain gates
  pass; avoid recomputing BA if using the new fused-input GDN boundary.

Local artifacts: `.artifacts/gdn-build-v1.log`, `gdn-build-v2.log`,
`gdn-build-cooperative.log`, `gdn-build-multi-geometry.log`,
`hc-norm-build-v1.log`. GPU test logs may exist
but be empty when lock acquisition timed out; file presence is not a result.
No model speed or output-quality result is claimed yet. No owned service.

AI-assisted work (Codex); human review and DCO sign-off required before merge.
