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

Current result: initial GDN and HC native SM70 builds passed. Updated
cooperative GDN build, GPU correctness, speed and sanitizer results must be
recorded before any performance claim. GPU 0--3 are reserved by another task;
honor the paper and 1Cat shared locks, even between its GPU launches.

Local artifacts: `.artifacts/gdn-build-v1.log`, `gdn-build-v2.log`,
`gdn-build-cooperative.log`, `hc-norm-build-v1.log`. GPU test logs may exist
but be empty when lock acquisition timed out; file presence is not a result.
No model speed or output-quality result is claimed yet. No owned service.

AI-assisted work (Codex); human review and DCO sign-off required before merge.
