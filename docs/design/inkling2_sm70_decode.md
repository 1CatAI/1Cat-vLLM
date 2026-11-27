# Inkling2 on SM70: single-card decode speed (stacked on the GGUF route)

Follow-up to `docs/design/inkling2_gguf_sm70.md`. Per decode step on one V100 (eager GPU
time, one sequence), #1598's model takes 23.5 ms and a standalone single-card port 14.4 ms
on the same weights. The profile attributes the gap to three components:

| component | #1598 kernels | #1598 | standalone | gap | layout-dependent? |
| --- | --- | --- | --- | --- | --- |
| router: logits and top-k | `_k_group_kernel` (2.77 ms), `_select_kernel` (0.54 ms) | 3.31 ms | 0.65 ms | **2.66 ms** | no |
| expert GEMMs and permutation | TurboMind NVFP4 stage, FP16 GEMV (layers 20–23), sort / permute / unpermute | ~7.7 ms | ~4.0 ms | **~3.7 ms** | yes |
| attention and short convs | `_rel_attention`, `_attn_prologue`, `_rel_decode_combine`, `_sconv` | ~3.6 ms | ~1.4 ms | **~2.2 ms** | no |
| loop controllers | `_pair_rms_norm`, `_blend_rms_norm`, two small batched matmuls | ~0.6 ms + matmuls | ~0.8 ms | ≈ 0 | no |

**Correction.** An earlier version of this profile (also in
`docs/design/inkling2_gguf_sm70.md`) attributed `_k_group_kernel` to the loop
controllers. It is the router: `select_experts` computes the `[128, 2304]` router logits
as K-group partials (`k_group_partials`). On SM70 that kernel pads one decode row to a
16-row mma tile, about 58 µs per call, for 48 calls per step.

## Routed experts: what reuse is possible

- **Where #1598's decode time goes.** At decode, #1598 runs the general SM70 NVFP4 MoE
  stage (`nvfp4_moe_dense_stage_sm70_out`) on TurboMind banks:
  - TurboMind GEMM, ~5.7 ms per step;
  - a sort / permute / select / unpermute pipeline, ~1.85 ms per step.
- **The shared codec's decode modes are model-specific.** `Fp4MoECodec` has
  small-batch modes (`qpn`, `direct_reduce`, `batch_reduce`, `fused_qpn`), but their kernels
  (`csrc/sm70_turbomind/ops/mxfp4_qpn_m1_sm70.cu`) are specialised to Qwen3.8 TP4 shapes:
  - 512 experts;
  - top-10 routing;
  - fixed output shapes `{512, 160, 320}` / `{512, 10, 2560}`.

  Calling `nvfp4_moe_qpn_m1_sm70_out` with Inkling2's shapes (128 experts, top-8, H 2304,
  I 1152) fails with "expected 10..160 routes in 10-route token groups and 512 experts".
  Generalising those kernels is native work in shared code, outside this PR series.
- **The standalone port's decode kernels need a different layout.** They are Triton dp4a
  GEMVs with SiTU fused, one program per (token, expert slot), and need no permutation.
  But they read NVFP4 in the GGML / official layout, while #1598 converts experts at load
  into TurboMind's swizzled tensor-core layout (`nvfp4_sm70_prepare`) and drops the
  original. Keeping both would duplicate ~11.5 GB of experts on one card.

The candidate approaches for the experts are therefore:
1. a Triton decode GEMV that reads TurboMind's layout directly (needs the swizzle
   documented and tested);
2. keeping experts in the official layout, with a graph-safe prefill path over it.

They are investigated separately, after the layout-independent parts.

## Order of work

1. **Router:** an SM70 GEMV branch in `k_group_partials` for up to 8 rows. It produces
   the same fp32 `[G, T, N]` K-group partials, so the in-order group sum, `select_experts`
   and the reference grouping are unchanged; only the summation order inside a 1024-wide
   group differs. Larger batches keep the tile kernel.
2. **Decode attention and short convs:** first, SM70 launch configs for #1598's existing
   prologue, short-conv and decode-attention kernels (done, below); new fused kernels
   only if the remaining gap justifies them.
3. **Experts:** after the layout question above is settled.

Each step replaces a #1598 decode kernel only where it is measured faster at TP=1 and TP=2,
behind #1598's existing call sites, with tests against #1598's current kernels as the
reference.

## Results

### 1. Router GEMV

`k_group_partials` takes an SM70 GEMV path for at most 8 rows. Q8 GGUF, TP=1, one V100,
CUDA graphs:

| decode tok/s (short / 4k / 16k) | before | after | change |
| --- | --- | --- | --- |
| this branch | 44.2 / 41.4 / 38.4 | 51.3 / 47.7 / 43.7 | +16 / +15 / +14 % |

Prefill is unchanged (~772 tok/s at 16k).

Numerics, measured by running the official checkpoint against #1598 with
`tools/inkling2/gguf_parity.py` (6 prompts, 24 tokens):
- greedy tokens are identical;
- chosen-token log-probs differ by 0.001 on average, 0.058 at most;
- the largest top-5 differences are on improbable alternatives (p 6e-7 → 4e-6 and
  8e-5 → 7e-4), consistent with rare routing near-ties resolving differently.

Kernel test `tests/models/inkling2/test_sm70_router_gemv.py`:
- the partials match the tile kernel within fp32 rounding;
- batches above 8 still take the tile kernel, bit for bit.

### 2. Launch configs: prologue, short conv, decode attention

Same kernels as #1598, with SM70 decode launch parameters chosen by sweeps (eager per-kernel
profile at TP=1, ms per decode step):

| Kernel | Change | Before | After |
| --- | --- | --- | --- |
| `_attn_prologue_kernel` | 8 warps for decode (no fused q/r tokens) | 16.8 µs | 9.6 µs |
| `_sconv_kernel` | 128-channel decode tile | 9.5 µs | 5.7 µs |
| `_rel_attention_kernel` | 32-key blocks for 16-row decode tiles | 1.66 ms | 0.91 ms |
| `_rel_attention_kernel` at 7k context | same | 4.0 ms | 3.4 ms |

Per-kernel figures are per launch for the prologue and short conv, and per step (48
launches) for attention. Rejected in the attention sweep:
- 2 warps: 4–8× slower;
- 1 pipeline stage: no change;
- fewer local splits: slower at 4k.

Prefill launches and other architectures are unchanged.

Q8 GGUF, TP=1, one V100, CUDA graphs, decode tok/s (short / 4k / 16k):

| Stage | tok/s |
| --- | --- |
| #1598 | 44.2 / 41.4 / 38.4 |
| + router GEMV | 51.3 / 47.7 / 43.7 |
| + prologue and short-conv configs | 52.8 / 49.0 / 44.8 |
| + 32-key decode attention blocks | 55.8 / 51.5 / 46.9 |

That is +26 / +24 / +22 % over #1598. Greedy output matches the official checkpoint on all
parity prompts at every stage.

### TP=2

Q4 GGUF, TP=2 on both V100s, 262k context, CUDA graphs, served with the #1599 parsers.
Decode tok/s (short / 4k / 16k), median of 3 runs:

| Tree | tok/s |
| --- | --- |
| PR 1 (#1598 kernels) | 47.2 / 44.8 / 42.9 |
| this branch | 58.1 / 54.3 / 51.7 |

That is +23 / +21 / +21 %. Prefill (~1,310 tok/s at 16k), weight memory (11.43 GiB per
GPU) and KV capacity (562,555 tokens) are unchanged.

### Quality

**Coding benchmarks.** Q4 GGUF, TP=2, temperature 1.0, up to 65,536 tokens, single
samples:

| | HumanEval / HE+ | LiveCodeBench v6 (60) |
| --- | --- | --- |
| PR 1 (#1598 kernels) | 98.2 / 93.3 | 85.0 |
| this branch, run 1 | 98.8 / 95.1 | 76.7 |
| this branch, run 2 | | 83.3 |

Earlier runs of other configurations scored 80.0–90.0 on the same 60 problems. The two
runs of this branch average 80.0. Three problems that four earlier runs all solved failed
in both runs of this branch, so the numerics were measured directly.

**Decode-path teacher forcing** (`forced_decode.py`, kept with the benchmark harness):
- Method:
  - sample traces on PR 1 for four LiveCodeBench problems: the three above, plus a hard
    problem every run solved, as a control; 16,728 tokens in all;
  - replay the traces through each tree with a logits processor that forces the trace's
    tokens;
  - record the model's raw top-20 log-probs at every step.
- Conditions: decode kernels, CUDA graphs, batch 4, Q4 GGUF.
- Determinism: replays are bit-identical run to run, and identical to the sampling run.

| Against PR 1 at TP=1 | mean \|Δ log p\| (chosen token) | KL (base top-20) | argmax flips |
| --- | --- | --- | --- |
| + router GEMV | 0.0182 | 0.00099 | 0.90 % |
| + launch configs | 0.0186 | 0.00104 | 0.93 % |
| + 32-key attention blocks (this branch) | 0.0190 | 0.00101 | 1.00 % |
| PR 1 itself at TP=2 | 0.0212 | 0.00134 | 1.08 % |

- Almost all of the difference comes from the router GEMV's summation order. The launch
  configs and attention blocks add nothing measurable.
- The difference is uniform across problems: the control problem shows the same size as
  the three that failed.
- It is smaller than the difference between #1598 at TP=1 and at TP=2, the same kernels
  with a different reduction split.
- The benchmark differences are therefore consistent with sampled trajectories diverging
  under a perturbation no larger than the one TP=2 already introduces.

### Test fix: FP8 KV attention test read past its scale buffers

`test_rel_attention_fp8_kv` (from #1598) failed intermittently in full-file runs:
- 2 of 6 runs at the #1598 head, 1 of 6 on this branch before the fix;
- non-finite output in the prefill case;
- never when run alone.

Cause: the test passed per-head `(H,)` q/r scales, while the fused clip reads
per-element `(HQ, D)` / `(HQ, D_rel)` scales, as the model allocates them.
compute-sanitizer memcheck reported 1879 invalid reads in `_l2_clip`, so results depended
on whatever memory followed the 96-byte buffers.

The test now uses full-shape scales with the same values, and `attention_prologue` asserts
the shapes. The model path was already in bounds. After the fix, 4 of 4 full-file runs
pass with identical numbers, and memcheck reports 0 errors.
