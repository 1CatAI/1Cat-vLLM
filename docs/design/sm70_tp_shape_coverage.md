# SM70 attention and quantized projection coverage across TP sizes

## Scope

Extend the existing V100 acceleration to the local tensor layouts produced by
TP1, TP2 and TP4. Dispatch must follow dtype, head grouping, matrix dimensions,
alignment and supported arithmetic, rather than a tensor-parallel-size allowlist.
Preserve FP32 accumulation where required by the accepted E4M3/75T contract.

Integration: `onecat/main`. Base: `b711d5304525dfc0cca6bc8a0bb005f33fe1bbf8`.
Owned branch/worktree: `codex/v100-tp-generalize-20260921-021932` /
`worktrees/v100-tp-generalize-20260921-021932`.

## Acceptance and worklog

- Correct E4M3 XQA partition selection for multiple KV heads at 32K through 256K.
- Admit GQA6/D256 prefill, batched decode and grouped FP32 verification across
  multiple local KV heads; retain the existing single-head implementation.
- Replace explicit TP4 quantized-projection gates with native layout capabilities
  and extend the measured projection configurations for the TP1/TP2 layouts.
- Validate arithmetic against independent references, CUDA Graph replay and
  route selection; separate prefill, pure decode and end-to-end serving results.
- Keep controls and candidate settings matched. Do not benchmark with eager mode.
- Promote defaults only after the affected quality and performance checks pass.

2026-09-21: Source audit found TP1/TP2 E4M3 C1 planning selects partition 1024 at
32K+, while native XQA rejects large partitions when Hkv>1. The 75T prefill,
E4M3 batched XQA and grouped FP32 gates require Hq=6/Hkv=1; NVFP4 QPN2 and FP8
QPN8/prefill additionally have explicit TP4 gates. Source policy checks are
not GPU performance or quality measurements.

GPU use authorized by the user: stop both existing services and use GPU 0–7.
Stopped-service launch records and raw validation artifacts are retained in
`/home/ymzx/1cat-build/tp-generalize-20260921/artifacts`. No unrelated service
code, model files, or canonical checkout changes belong to this task.

Implementation and validation: pending.
