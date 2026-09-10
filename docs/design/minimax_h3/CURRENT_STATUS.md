# H3 retained FA delivery status

Current delivery concentrates on already working H3 dense workflows and the
retained `FLASH_ATTN_V100` implementation. This is already the native H3
default backend. Slower experimental replacements are excluded. Further
Attention prototype work and new workflow expansion are paused at the user's
request; no GPU experiments remain queued. Native CLI/HTTP APIs remain the
frontend entry point.

## Formal four-step result

TP4, V100 SXM2 32GB, single request, 1280x736 internal canvas, 120 requested
frames aligned to 124, 24 FPS, LightX2V v1.2 four-step W8A16. Both backend
runs use the same pageable host/shared VAE policy and native peer reduction.
Each has one complete request warmup and three unprofiled measurements.

| Metric | Retained FA | Retained FI |
| --- | ---: | ---: |
| Complete denoise median | 58.293218 s | 60.969633 s |
| Denoise divided by four updates | 14.573304 s | 15.242408 s |
| Minimum-card median useful TFLOP/s | 53.235745 | 50.898828 |
| Denoise CV | 0.076959% | 0.069457% |
| Complete request median | 91.071940 s | 89.525699 s |

FA reduces denoise by 4.389750%. The complete request includes other stages
and is slower in this measurement; an overall FA request speedup is not
established. The tracked live-allocation upper bound including IPC is
20,475,227,136 bytes/card. All >80 gates remain incomplete.

## Retained coverage

These are complete native output-preservation controls. Except for the
explicit formal result above, the times below are captured cold diagnostics.
They are not repeated performance acceptance or independent official-model
quality acceptance.

| Workflow | Weights | Denoise | Numerical control |
| --- | --- | ---: | --- |
| Base H3, no adapter, 49 updates | Original floating | 649.973 s | Passed |
| LightX2V four-step v1.2, peer reduction | Original floating | 59.324 s | Passed |
| Six official FL2V four/eight-step adapters | W8A16 | Four: 58.293-61.263 s; eight: 120.478-121.542 s | Passed |
| Ref2VA four-step, image/video/audio, peer reduction | W8A16 | 181.261 s | Passed |
| Ref2VA eight-step, image/video/audio | W8A16 | 366.800 s | Passed |
| Light4 v1.2 first / last / both keyframes | W8A16 | 66.419 / 64.852 / 69.615 s | Passed |
| FlashGen four-step | Original floating | 59.488 s | Passed |
| FastH3 Dense data-free | Original floating | 56.224 s | Passed |

The no-adapter original-weight control decreases denoise from 725.819 to
649.973 seconds (10.449687%) and complete request from 835.356 to 700.121
seconds. Final video/audio latents, all pre-encoding RGB frames and PCM are
bitwise equal. Combined host, projection and residual changes contribute;
this is not an isolated Attention comparison.

Original floating and W8A16 bases share FP16 projection, scale restoration,
LoRA addition and Attention interfaces. Adapter increments retain unrotated
inputs and enter row reduction before the collective. Residual sharding and
native peer reduction are explicit options, with memory-budget fallback.
No quantization-only or adapter-free restriction selects shared Attention.

## Reproducing the measured configuration

Use the native `H3Config` with the model and matching adapter identifiers
from the recorded request contract. The measured TP4 primary configuration
sets `attention_backend="FLASH_ATTN_V100"`, `attention_query_tile=128`,
`fp16_weight_layout="column"`, `residual_sequence_parallel=True`,
`residual_reduction="peer"`, `residual_reduction_memory_gib=4`,
`host_weight_pin_memory=False`, and `share_host_vae_weights=True`.
The mixed Ref4 control uses an explicit 8 GiB communication budget.
Keep each adapter's official sigma, flow-shift and task contract.

This records explicit measured options, not a universal default or AUTO
promotion. Peer reduction has complete controls for the primary four-step,
original four-step/no-adapter and mixed Ref4 cases. Other rows retain their
measured configurations; the latest peer option is not claimed validated for
every shape/adapter combination. TP1/TP2 capacity checks use layer offload
and ordinary reduction, with no primary-shape >80 claim.

FastH3 VSA remains outside qualified delivery because its full numerical
diagnostic fails. TeaCache and Cache-DiT/SCM have request-lifecycle and
small-shape GPU evidence; primary-shape official quality and performance are
incomplete. The 243-frame and 15-second runs establish generation/memory
compatibility only. Human audiovisual review and independent official-model
controls remain pending. The user authorized source integration through
PR #583 on 2026-09-10; this does not close those acceptance gates. See
[VSA_QUALITY_SPEED.md](VSA_QUALITY_SPEED.md) for the subsequent VSA stage:
the native path reaches 30.990756 seconds but fails independent quality;
the acceptance-only exact FP32 path passes primary numerical checks at
60.353224 seconds and fails the speed target. Neither enters automatic selection.

Evidence: [CAMPAIGN_RESULTS.md](CAMPAIGN_RESULTS.md),
[FA_DEVELOPMENT.md](FA_DEVELOPMENT.md), and artifact root
`/data/minimax-h3/sm70-general-20260909/` with
`peer-api-720p-three-runs/performance.json`,
`peer-api-fi-720p-three-runs/performance.json`,
`peer-api-breadth-summary.json`, `original-base-summary.json`, and full
request/source/binary contracts. No new timing is inferred from the scope
change or documentation update.
