# SM70 Tile Runtime Exploration

This document tracks the TileRT-inspired path for Qwen3.6-27B-AWQ TP2
decode on V100/SM70. The goal is not to clone TileRT's closed runtime, but to
reproduce the execution model idea in the narrowest place where this tree can
measure it safely.

## Public Signals

TileRT's public README and blog describe three ideas that matter for our
decode path:

- Move from `graph -> operator -> kernel` to a persistent engine kernel.
- Decompose operators into tile-level tasks so compute, IO, and communication
  can progress together.
- Use warp/block specialization and heterogeneous workers so different GPU
  execution resources keep making progress instead of repeatedly draining at
  operator boundaries.

Relevant public references:

- TileRT README: https://github.com/tile-ai/TileRT
- TileRT execution-model blog: https://www.tilert.ai/blog/speed-as-the-next-scaling-law.html
- TileRT 1000 TPS blog: https://www.tilert.ai/blog/breaking-1000-tps.html
- DistFuse tile-wise GEMM/all-reduce paper: https://mcanini.github.io/papers/distfuse.hotinfra24.pdf
- TokenWeave paper: https://arxiv.org/html/2505.11329v2
- CUTLASS-native Distributed GEMM write-up: https://blog.shi-labs.com/distributed-gemm-88be6a481e2b

The TileRT repository currently ships Python wrappers and binary backends, not
the core engine source. The actionable implementation detail therefore comes
mostly from the execution-model description plus related open papers.

## Current Local Evidence

Today's Nsight Systems trace for 27B-AWQ TP2 decode shows that the steady
decode replay is still a single serial CUDA graph stream per rank. Per token,
device 0 spends about:

- `~19.95 ms` wall time.
- `~11.67 ms` in GEMM kernels.
- `~1.57 ms` in all-reduce kernels.
- `~0.78 ms/token` in all-reduce after `gemm_grid_1x20x7`.
- `~0.76 ms/token` in all-reduce after `gemm_grid_1x40x12`.

The custom all-reduce small-message block-limit tuning is real but small:

- 5120 fp16 all-reduce microbench improved from about `12.96 us` to `10.59 us`.
- End-to-end decode improved from `56.168 tok/s` to `56.450 tok/s` with the
  default auto heuristic.

The chunked all-reduce diagnostic is the stronger negative result:

- 1 chunk for 5120 fp16: `~13.21 us`.
- 2 chunks: `~19.97 us`.
- 4 chunks: `~37.37 us`.
- 8 chunks: `~66.55 us`.
- 16 chunks: `~127.68 us`.

So the next step cannot be "split all-reduce into many launched kernels".
Tile-level overlap has to be inside one fused/chunk-aware engine, or through a
small number of long-lived worker kernels with device-side flags.

## First Target

The first TileRT-style target is the row-parallel AWQ GEMM followed by TP2
all-reduce, especially the MLP down projection.

Current execution:

```text
TurboMind AWQ GEMM -> global store -> custom all-reduce launch -> global load
```

Target execution:

```text
GEMM tile produces N-slice
  -> publish tile-ready flag
  -> peer/device worker reduces that tile while later GEMM tiles continue
  -> final output lands in the same tensor layout expected by vLLM
```

This is a smaller version of TileRT's model:

- The host still launches through vLLM/CUDA graph initially.
- We do not build a whole-model persistent engine in phase 1.
- We only collapse the hottest GEMM/communication boundary.
- The experiment stays TurboMind-only and TP2-only.

## Why This Boundary

The existing TurboMind GEMM epilogue already has:

- CTA/tile awareness.
- Split-K partial storage.
- Per-tile barrier logic.

The existing SM70 custom all-reduce already has:

- IPC-opened peer pointers.
- Peer-visible signal buffers.
- CUDA graph registration support.

The missing bridge is a tile descriptor/flag protocol between these two
systems. That bridge is much smaller than writing a persistent whole-model
runtime first.

## Engineering Plan

1. Add a tile-runtime candidate ranker.

   Verification: rank 27B-AWQ TP2 nsys traces by all-reduce time after GEMM,
   GEMM grid shape, CTA count, and estimated overlap ceiling.

2. Add a local tile-ready microkernel.

   This should not touch model execution. It should simulate:
   producer stores N tiles and sets flags, consumer waits and reduces/copies
   peer-visible tiles. This validates flag ordering, tile size, and busy-wait
   overhead on V100.

3. Add an experimental TurboMind epilogue mode.

   Gate it behind an env var. The GEMM still writes the normal output, but
   also publishes a tile-ready flag per output tile. A separate single
   persistent/long-running reduce worker consumes tiles. This proves overlap
   before risking correctness in the main path.

4. Collapse to a fused row-parallel AWQ op.

   Add a TP2-only op that receives TurboMind AWQ tensors plus the custom AR
   communicator metadata and returns the reduced output. Start with M=1 and
   output hidden size 5120. Validate bitwise/token parity against the current
   graph path.

5. Generalize only after measured gain.

   If the fused path recovers less than all-reduce time or hurts GEMM
   occupancy, stop and inspect scheduler/register/shared-memory effects before
   expanding to attention o_proj or other shapes.

## Acceptance Gates

The first source-path implementation is acceptable only if all are true:

- Same token hash as current fast selector path.
- 27B-AWQ TP2 decode benchmark uses the same model, GPUs, graph policy,
  input/output length, sampling, and TurboMind route.
- Nsight Systems shows the GEMM/all-reduce boundary is no longer strictly
  serialized for the target shape.
- End-to-end decode improvement is larger than the current all-reduce
  heuristic noise floor; target first-stage win is at least `3%`.
- If the first target cannot hit `3%`, the trace must explain whether the
  failed limit is GEMM occupancy loss, peer flag overhead, or insufficient
  overlap window.

## Risks

- V100 lacks Hopper-era TMA, PDL, and NVSHMEM-like hardware assistance used by
  newer distributed GEMM work. The design must rely on P2P memory and explicit
  flags.
- A fused all-reduce epilogue can increase registers/shared memory and reduce
  GEMM occupancy. TokenWeave specifically calls out this tradeoff for fused
  communication kernels.
- Small communication tiles can be inefficient. Our own chunked-AR diagnostic
  already shows this. Tile size must be coarse enough to avoid kernel/flag
  overhead dominating.
- Split-K and tile-ready signaling interact. For split-K GEMMs, only the final
  split can publish a complete output tile unless the reduce path also handles
  split partials.

## Current Direction

For 27B-AWQ TP2, the next implementation target should be:

```text
M=1, N=5120 output, TP2, fp16 output,
TurboMind AWQ GEMM epilogue tile-ready flags,
single fused/chunk-aware TP2 reduce worker,
CUDA graph compatible.
```

Do not expand to Marlin, FP8, MoE, or a whole-model persistent engine until
this boundary either shows a real decode win or gives a clear negative result.

## Phase 0 Skeleton

Implemented first local skeleton:

- C++ op: `_C_custom_ar.tile_runtime_all_reduce`.
- Python wrapper: `CustomAllreduce.tile_runtime_all_reduce`.
- Benchmark: `benchmarks/benchmark_sm70_tile_runtime.py`.

This is deliberately not the final TileRT-style engine. It is a single-kernel
persistent CTA pool that executes tile tasks:

```text
CTA worker gets tile
  -> copies local input tile to peer-visible staging buffer
  -> publishes tile-ready flag through existing custom-AR signal memory
  -> waits for peer tile-ready flags
  -> reduces peer staging tiles into output
```

The point of this phase is to prove the substrate:

- TP2 P2P pointers can be reused from the existing custom all-reduce
  communicator.
- Tile-ready flags work inside a CUDA graph replay.
- A single kernel can execute a pool of tile tasks without adding one kernel
  launch per tile.
- Output matches `torch.distributed.all_reduce` exactly.

Validation artifact:

- `bench_results/nsys_27b_awq_tp2_20260620/tile_runtime_proto_5120_fp16_tp2_engineblocks.json`

Key 5120-fp16 TP2 results on GPU0/1:

- `tile_numel=5120`, `engine_blocks=1/2/4/8/0`: about `10.7-10.8 us`,
  `max_abs=0`.
- `tile_numel=2560`, enough blocks: about `12.35-12.45 us`, `max_abs=0`.
- `tile_numel=1024`, `engine_blocks=8/0`: about `12.07-12.13 us`,
  `max_abs=0`.
- `tile_numel=512`, `engine_blocks=0`: about `12.22 us`, `max_abs=0`; with
  only one worker block it degrades to about `65.5 us`.

Interpretation:

- Coarse tiles are still required. Fine-grained tiles are only viable if there
  are enough resident worker CTAs, and even then flag/scheduler overhead is
  visible.
- The single-tile case matches the optimized custom all-reduce latency class,
  so the prototype did not introduce a large unavoidable overhead.
- This skeleton is now suitable as the substrate for the next phase: replace
  the synthetic producer copy with TurboMind AWQ tile production.

## Phase 1 Role-Specialized Engine

Implemented a second standalone prototype:

- C++ op: `_C_custom_ar.tile_runtime_all_reduce_engine`.
- Python wrapper: `CustomAllreduce.tile_runtime_all_reduce_engine`.
- Benchmark mode: `benchmarks/benchmark_sm70_tile_runtime.py --modes engine`.

This version moves one step closer to the public TileRT execution model by
separating tile workers by role inside a single CUDA graph-compatible kernel:

```text
producer CTA
  -> writes local tile to peer-visible staging buffer
  -> publishes tile-ready flag

reducer CTA
  -> waits for peer tile-ready flags
  -> reduces peer staging tiles into output
  -> advances the per-tile replay flag
```

This is still not a model-level persistent runtime. It is a narrow device-side
engine substrate that keeps the producer/reducer protocol explicit so the next
step can replace the synthetic producer copy with TurboMind AWQ tile production.

Validation artifacts:

- `bench_results/nsys_27b_awq_tp2_20260620/tile_runtime_engine_phase1_5120_fp16_tp2_smoke.json`
- `bench_results/nsys_27b_awq_tp2_20260620/tile_runtime_engine_phase1_5120_fp16_tp2_rolesweep.json`
- `bench_results/nsys_27b_awq_tp2_20260620/tile_runtime_engine_phase1_5120_fp16_tp2_inline_baseline_1000.json`

Key 5120-fp16 TP2 results on GPU0/1:

- Inline baseline, `tile_numel=5120`: about `11.41 us`, `max_abs=0`.
- Role engine, `tile_numel=5120`: best observed about `12.27 us`,
  `max_abs=0`.
- Role engine, `tile_numel=2560`: best observed about `12.82 us`,
  `max_abs=0`.
- Role engine with fine tiles is much slower when too few producer/reducer CTAs
  are available; for example `tile_numel=512`, `producer_blocks=1`,
  `reducer_blocks=1` is about `39.5 us`.

Interpretation:

- The role engine should not replace the current all-reduce microkernel by
  itself. Pure staging plus reduce pays extra CTA and flag overhead.
- The useful result is correctness and graph compatibility for role-specialized
  producer/reducer workers. This is the shape needed to test true GEMM/comm
  overlap.
- The next implementation should connect the TurboMind AWQ epilogue to this
  protocol for the M=1, TP2, N=5120 row-parallel output path, publishing
  coarse tile-ready flags only after complete output tiles are available.

## Phase 1.5 Dense MLP Gate-Up Fused-SiLU

Implemented a narrow, default-off dense MLP fast path:

- Env gate: `VLLM_SM70_AWQ_MLP_ENGINE=1`.
- Weight loading: AWQ `gate_up_proj` can prepare a single interleaved
  TurboMind layout through `awq_sm70_prepare(..., interleave_gated_silu=True)`.
- Runtime: Qwen2/Qwen3 dense MLP calls `forward_fused_silu_and_mul()` for
  M=1/TP2 only, so `gate_up_proj GEMM + SiluAndMul` becomes one TurboMind GEMM
  with `gated_silu=True`.
- The MLP `down_proj` and TP2 all-reduce remain on the existing path.

Validation artifacts:

- `bench_results/sm70_awq_mlp_engine_20260621/baseline_i512_o16.json`
- `bench_results/sm70_awq_mlp_engine_20260621/candidate_i512_o16.json`
- `bench_results/sm70_awq_mlp_engine_20260621/baseline_i512_o128_r3.json`
- `bench_results/sm70_awq_mlp_engine_20260621/candidate_i512_o128_r3.json`

Correctness:

- Synthetic AWQ op check: normal layout `GEMM + silu*mul` vs interleaved
  `gated_silu` epilogue produced `max_abs=0`; interleaved normal output
  restored to the original order also produced `max_abs=0`.
- End-to-end Qwen3.6-27B-AWQ TP2 token hashes matched baseline:
  - 16-token smoke: `3d326cfe4c2bcd0aae49b11ed684dc2baff445be00e7eff6472cb014eb269cdb`.
  - 128-token repeat-3: `25acc1257b65019cd7398d36105134378675d9de82bc7d30c6b49d78b1f7b755`.

Performance, 27B-AWQ TP2 on GPU0/1, `input_len=512`, `output_len=128`,
`repeat=3`, Flash-V100 compile graph:

- Baseline: `17.7997 ms` TPOT, `56.1807 tok/s` steady decode.
- Candidate: `17.7242 ms` TPOT, `56.4201 tok/s` steady decode.
- Delta: about `0.42%` TPOT improvement.

Interpretation:

- This phase is correct and gives a reusable MLP fast-path entry point, but it
  is far below the `3%` first-stage acceptance target.
- The result confirms that eliminating only the separate `SiluAndMul` launch
  and full gate/up intermediate is too small. The large remaining opportunity
  is still the row-parallel `down_proj` AWQ GEMM plus TP2 all-reduce boundary.
- Keep this path experimental/default-off until a later down-proj tile/reduce
  overlap phase shows a material end-to-end win.

## Phase 2 MLP Down-Projection Tile-Ready Reduce

Implemented the first source-path TurboMind down-projection experiment:

- Env gate: `VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP=1`.
- Scope: Qwen3.6-27B-AWQ, TP2, M=1 decode, dense MLP `down_proj` only,
  fp16 output, TurboMind backend only.
- TurboMind GEMM epilogue now supports `kTileAllReduce` and publishes
  peer-visible ready flags for complete output N tiles.
- The custom all-reduce communicator exposes the existing CUDA-graph
  rank-data registration path so TurboMind can reuse IPC-opened peer staging
  pointers instead of adding a second IPC system.
- Python dispatch wires `RowParallelLinear.down_proj` to a single opaque custom
  op under CUDA graph capture. The op falls back to normal GEMM plus all-reduce
  outside the supported M=1/TP2 path.

Three variants were tested:

1. `same-stream wait-reduce`: GEMM publishes tile flags, then the existing
   reduce worker consumes the staging buffer on the same stream.
2. `side-stream wait-reduce`: a reducer worker starts on a side stream and
   waits for tile flags while GEMM runs on the graph stream.
3. `epilogue-fused reduce`: the producing GEMM CTA publishes its tile flag,
   waits for the peer tile, then writes the reduced final output itself.
4. `tail-worker reduce`: every producing CTA stores its staging tile and
   increments one completion counter; only the first N-tile CTA remains alive
   as a single tail worker, waits for both ranks to finish the whole row, then
   reduces the complete fp16 output with `half2`.

Validation artifacts:

- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/samestream_publishfix_i512_o4.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/epilogue_fused_epochfix_i512_o4.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/epilogue_fused_localfrag_i512_o4.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/tailworker_i512_o4.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/baseline_i512_o32_r2.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/fused_localfrag_i512_o32_r2.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/baseline_current_i512_o32_r2.json`
- `bench_results/sm70_awq_mlp_down_tile_overlap_20260621/tailworker_i512_o32_r2.json`

Correctness:

- The first graph-capture route initially missed the new op because
  `torch.compile` traced the M>1 profiling branch and skipped the M=1 decode
  branch. The fix was to dispatch the opaque custom op for prepared down-proj
  layers and let the runtime op handle M=1 support/fallback.
- The first tile-ready epilogue deadlocked because the TurboMind CTA N tile and
  logical reducer tile were not treated independently. The fix publishes all
  logical reducer flags fully covered by a CTA.
- The first epilogue-fused reduce returned a wrong token hash because it did
  not advance the per-tile replay epoch. Updating `self_sg->_flag[tile]` after
  reduce restored token parity.
- Current fused path matches the normal path on the 32-token comparison:
  both repeats produced token hash
  `1ccba72c62311edee3c604b4d41cdeddad72adb96fd6666a1af609c59708fd68`.
- Tail-worker reduce also matches the normal path:
  - short smoke, `output_len=4`: token hash
    `e3b3ebc930b09ed8921c4e20f0c8dd6fb93f0bd51c382cdc33a485050ef60939`;
  - stable comparison, `output_len=32`: both repeats produced token hash
    `1ccba72c62311edee3c604b4d41cdeddad72adb96fd6666a1af609c59708fd68`.

Performance, 27B-AWQ TP2 on GPU0/1, `input_len=512`, `output_len=32`,
`warmup=1`, `repeat=2`, Flash-V100 compile graph:

- Normal custom-all-reduce path:
  `17.7176 ms` mean TPOT, `56.4411 tok/s` steady decode.
- Epilogue-fused tile reduce:
  `18.0920 ms` mean TPOT, `55.2730 tok/s` steady decode.
- Delta: about `2.1%` slower.

Same-day current-binary A/B after adding the tail-worker reducer, same shape:

- Normal custom-all-reduce path:
  `17.5106 ms` mean TPOT, `57.1084 tok/s` steady decode.
- Tail-worker reduce, with the benchmark command recording
  `VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_REDUCER_BLOCKS=1`:
  `17.9204 ms` mean TPOT, `55.8023 tok/s` steady decode.
- Delta: about `2.34%` slower than the current normal path, but about `0.95%`
  faster than the earlier per-tile epilogue-fused reduce.

Interpretation:

- Blocking the Tensor Core GEMM CTA on peer tile readiness is the wrong
  TileRT-style shape for V100. It is correct, graph-compatible, and avoids a
  separate reduce launch, but it extends the GEMM kernel's critical path and
  loses more than the launch/all-reduce boundary can recover.
- The tail-worker version reduces the number of wait/reduce points, but every
  producer CTA still pays a system fence plus atomic completion update, and a
  single CTA then serializes the final row reduce. It improves the previous
  per-tile fused shape but still does not beat the normal custom all-reduce
  path.
- The next viable route is a real lightweight tile scheduler/worker that can
  wait on flags without occupying the GEMM CTA and without turning the final
  reduction into a single-CTA tail. The previous side-stream worker variant
  hung in real decode, so the worker must be redesigned rather than simply
  increasing reducer blocks.
- Do not promote the current epilogue-fused or tail-worker reduce to the
  default path. Keep them as default-off diagnostic substrates for Nsight and
  future worker design.
- A later attempt to make `reducer_blocks == 1` an explicit tail-worker switch
  was reverted: the full-model smoke reached graph capture, registered zero
  custom-allreduce CUDA graph addresses, then the worker exited. Keep the
  current route selection unchanged until the worker design is replaced.

### Single-Kernel Reducer CTA Prototype

Implemented one narrower TileRT-style single-kernel prototype after the
tail-worker result:

- Env gate:
  `VLLM_SM70_AWQ_MLP_DOWN_TILE_OVERLAP_KERNEL_REDUCER_BLOCKS`, default `0`.
- Scope remains M=1 decode, TP2, fp16 output, dense `down_proj` only.
- Producer CTAs are still the TurboMind AWQ GEMM CTAs. They store rank-local
  staging output, publish per-tile ready flags, and exit without waiting.
- Extra reducer CTAs are appended as a second z-plane in the same
  `gemm_kernel` launch. Reducer CTAs wait for both ranks' ready flags and
  write `half2` rank0+rank1 into the final output.
- The path is intentionally default-off. It is a scheduler substrate, not an
  accepted fast path.

Validation:

- Built `_C` directly with
  `cmake --build build/temp.linux-x86_64-cpython-312 --target _C -j 8`;
  full `setup.py build_ext --inplace` still fails on the unrelated
  `_deep_gemm_C` target in this tree.
- Import/schema check passed with the new op argument.
- Short smoke with `kernel_reducer_blocks=4` matched the previous short hash:
  `e3b3ebc930b09ed8921c4e20f0c8dd6fb93f0bd51c382cdc33a485050ef60939`.
- Stable 32-token comparison matched the same steady hash as the baseline:
  `1ccba72c62311edee3c604b4d41cdeddad72adb96fd6666a1af609c59708fd68`.

Performance, same GPU0/1, `input_len=512`, `output_len=32`, `warmup=1`,
`repeat=2`. The current rebuilt tree required
`VLLM_FLASH_V100_ALLOW_TRITON_FALLBACK=1` for this Flash-V100 run because the
direct no-fallback attention path failed before the MLP path.

- Tail-worker control under the same compile/fallback setup:
  `17.8614 ms` mean TPOT, `55.9866 tok/s` steady decode.
- Single-kernel reducer CTA path with `kernel_reducer_blocks=1`:
  `23.7277 ms` mean TPOT, `42.1449 tok/s` steady decode.
- Single-kernel reducer CTA path with `kernel_reducer_blocks=4`:
  `19.5130 ms` mean TPOT, `51.2478 tok/s` steady decode.
- Delta: `kernel_reducer_blocks=4` is about `9.25%` slower than the same-run
  tail-worker control; `kernel_reducer_blocks=1` is about `32.84%` slower.

Interpretation:

- This is the important TileRT lesson from the prototype: "same kernel" is not
  equivalent to a tile runtime.
- Appending a reducer z-plane creates extra CTAs that compete with the GEMM
  CTAs for SM scheduling. The producers no longer wait, but the reducer work is
  now resident in the same kernel and can delay Tensor Core work before enough
  useful tiles exist to reduce.
- Reducing the active reducer count to one makes the result much worse, so the
  other failure mode is also real: too few reducer CTAs serialize the final row
  consumption into a long tail.
- The remaining path should be a real persistent tile scheduler, not more
  reducer blocks. A useful next version needs fixed CTA roles or a device-side
  work queue so compute CTAs keep the SMs full while a small number of
  communication/reduce CTAs consume only ready tiles.
- A practical first scheduler should use a flattened grid, not a mostly-idle
  extra z-plane. Target shape: reserve one or two reducer CTAs per GPU only
  after enough producer progress, keep producer CTAs dominant, and make reducer
  work steal cycles only when Tensor Core producer CTAs would otherwise be
  stalled.

Nsight note:

- Attempted `nsys profile --capture-range=cudaProfilerApi` around the measured
  repeat with `profiler_config={"profiler":"cuda"}`. Under this multiprocess
  vLLM startup, the worker processes were terminated during checkpoint loading,
  so no valid `.nsys-rep` was produced. Timeline capture should be retried with
  the already-working project profiling wrapper or a smaller direct op harness,
  not by repeatedly wrapping full model startup in this exact command.
