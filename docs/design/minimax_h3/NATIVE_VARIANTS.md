# Native H3 variant integration

This branch extends the prepared execution and workflow accounting stack
(#571 / #578) with the pinned official FastH3 VSA algorithm. Performance above
80 useful TFLOP/s/card, independent official full-sampling quality and human
audiovisual review remain incomplete.

## Sparse execution contract

Select `attention_backend="FASTVIDEO_VSA"` with an explicit official FastH3
VSA adapter and `vsa_topk=64` in `H3Config`. The equivalent native CLI options
are `--attention-backend FASTVIDEO_VSA --fastvideo-vsa-topk 64`. Existing video
HTTP APIs consume the configured engine; no additional frontend is required.

The loader validates the artifact's declared inventory and injects all 50 main
DiT compression projections before normal TP loading and CPU staging. The
token refiner remains dense, as in the release. Sparse artifacts cannot run on
a dense backend, dense artifacts cannot silently switch to VSA, and FastH3
still requires original floating weights, T2VA, four intervals and its own
per-modality shifted DMD2 schedule. Fused adapters remain immutable per engine.

The geometry and learned compression follow `fastvideo_vsa.py` from Omni
`7be014bce6374f06c95b703763bdbac4c6198f31`. Prefix segments form independent
64-token chunks; target video forms 4x4x4 tiles. Prefix queries select every
valid key block, while video queries select all prefix blocks plus top-k video
blocks. Edge tiles retain their actual lengths. The learned compressed output
is added using the same pooled FP32 scores and explicit gate; no sigmoid or
replacement with a dense attention call is introduced.

The shared `sm70_sparse_attention.block_sparse_attention` interface accepts
pre-tiled FP16 operands, a boolean block map and int32 valid block lengths.
A warp compacts each map row in ascending block order. The SM70 kernel executes
only those blocks and masks all nonterminal padding holes. Its CUDA extension
is included in the SM70 wheel targets and also supports the existing CUTLASS
source-development workflow. No external FastVideo kernel package is required.

## Accounting and isolation

Per-layer and per-step records retain actual selected block/token-pair counts,
pooled compression FLOPs and avoided attention FLOPs. The numerator includes
selected valid pairs, actual compression and the gate projections. Padding
and unselected pairs do not inflate throughput. The acceptance evaluator checks
the declared geometry against every layer and step, keeps algorithm savings
separate, and still requires a full warmup and three unprofiled requests.

Geometry caches contain immutable indices only, keyed by shape, prefix and
device. Scores, pooled activations and gate values are created for each call.
This implementation does not enable TeaCache or Cache-DiT; those request-state
policies and AUTO selection remain separate campaign work.

## Validation record

Python 3.12.13, Torch 2.10.0+cu128, CUDA toolkit 12.8.93, V100 SXM2 32GB.
Evidence root: `/data/minimax-h3/sm70-general-20260909/`.

- `sparse-gpu-v1.log`: 8 operator checks pass against an independent masked
  FP32 reference, with NaN-poisoned padding holes, different batch/head maps,
  dense-equivalent arithmetic and invalid-input rejection.
- `vsa-geometry-cpu-v1.log`: 13 geometry, top-k and learned-gate checks pass.
- `vsa-integration-cpu-v1.log`: 169 existing API/workflow/adapter checks pass,
  one GPU case skipped.
- `vsa-fusion-cpu-v1.log`: 33 checks pass, one GPU case skipped; this includes
  all 50 gate assignments, TP1/TP2/TP4 native loaders and host snapshots.
- `vsa-dense-and-accounting-gpu.log`: 37 checks pass, including the rebuilt
  dense specialization and a real sparse DiT block with independent FLOP
  arithmetic, bitwise hook transparency and hook cleanup.
- `vsa-accounting-cpu-v1.log`: 71 checks pass, two GPU cases deselected. The
  evaluator rejects invented padding, block counts, compression and savings.

Full native FlashGen and FastH3 Dense five-second GPU generation completed on
the preceding combined source `5195f31b8d`, with original weights and native
pageable host masters. Basic media checks pass; those cold captured requests
are not speed or independent quality acceptance.

Full native VSA generation on source `61c57e36ae` also completed with the
official data-free adapter (SHA256
`42dc502a2078f166c396a1fa75f29728d1844363652d345d5ef3e2b444ed6470`).
This TP4 cold captured request used the same 1280x736, five-second, seed-42
prompt. Complete denoising took 37.387443 seconds; the slowest rank per step
took 12.211990, 8.383499, 8.374612 and 8.412264 seconds. Peak allocated memory
was 20,987,568,640 bytes/card. All four ranks passed the native sparse-work
validator, and basic video/audio checks passed. Results and raw captures are
in `/home/ymzx/h3-sm70-artifacts-20260909/runs/fasth3-vsa-720p-native/`;
`fasth3-vsa-native-summary.json` records the compact audit.

Actual useful throughput was 45.2679–45.2773 TFLOP/s/card. Approximately
1.484e15 skipped attention FLOPs/card are reported separately, not credited
to throughput. This single cold request is not the warmup-plus-three-run
performance gate. Independent official full-sampling comparison and human
review remain pending; a sampled frame contains several ducks despite the
prompt specifying one, so basic media checks do not establish prompt fidelity.
No complete workflow is yet qualified for AUTO or the >80 target.
