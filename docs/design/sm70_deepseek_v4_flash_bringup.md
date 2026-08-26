# DeepSeek V4 Flash on SM70

## Scope

Bring `deepseek-ai/DeepSeek-V4-Flash` up on eight Tesla V100-SXM2-32GB
GPUs without regressing the existing 1Cat V100 fast paths. The checkpoint
uses MXFP4 experts and FP8 dense weights. The first milestone is a correct,
measurable route; kernel tuning starts only after correctness is established.

## Source Baseline

- Integration base: `onecat/main`
- Base SHA: `3ec0c68c6596d6ab31fbdee9fa676254a52c2b7d`
- Worktree: `worktrees/v100-deepseek-v4-sm70-bringup-20260801-143050`
- Branch: `agent/v100-deepseek-v4-sm70-bringup-20260801-143050`
- Official reference release: vLLM `v0.26.0` (2026-07-27)
- Official reference main SHA at final audit:
  `dc818c198d3ff50a16f38eba567da006478239c8` (2026-08-01)
- Model repository SHA:
  `60d8d70770c6776ff598c94bb586a859a38244f1`

The 1Cat base diverges from official vLLM near `v0.21.1rc0`. Official
`v0.26.0` is 1,815 commits ahead while 1Cat has 24 divergent commits. A
read-only merge-tree audit found broad conflicts in CUDA build logic,
TurboMind, quantization, CUDA graphs, MTP, scheduling, and model code.
Therefore this migration backports a dependency-closed DeepSeek V4 subset;
it does not merge the full upstream release.

## Checkpoint Contract

The model repository metadata and safetensors headers were inspected directly
at the pinned model SHA. The release route is locked to these dimensions:

- 43 transformer layers, hidden size 4096, and maximum context 1,048,576.
- 64 attention heads with 512-dimensional Q/K heads, including 64 RoPE
  dimensions; eight O-projection groups with rank 1024.
- 256 routed experts, top-k 6, expert intermediate size 2048, and one shared
  expert. Routed experts use packed E2M1 MXFP4 values with UE8M0 scales.
- The Lightning Indexer has 64 heads of dimension 128 and selects top-k 512.
- One MTP layer is present in the checkpoint.
- Total indexed checkpoint size is 159,609,485,896 bytes. Packed expert
  safetensor shapes agree with TP8 local intermediate size 256 after sharding.

## Target Hardware

- Eight Tesla V100-SXM2-32GB GPUs (SM70)
- GPUs 0-3 and 4-7 form separate four-GPU NVLink/NUMA islands
- Cross-island links include `SYS` paths
- Primary deployment candidates: TP8 and PP2 x TP4
- Remote CUDA driver: 580.173.02
- Remote Python environment target: Python 3.12

The local machine currently cannot launch CUDA work because the loaded NVIDIA
kernel module and userspace NVML versions differ. Local validation is limited
to static and CPU tests until reboot; GPU correctness and performance run on
the remote V100 host after the PR is merged and freshly cloned.

## Upstream Reuse

Reuse these official implementations and semantics:

- DeepSeek V4 model/config, C4/C128 compressor, indexer, MTP, and correctness
  fixes from vLLM `v0.26.0`.
- Generic Triton qnorm + GPT-J RoPE + FP8 KV insertion from the official XPU
  fallback, adapted from BF16 to an FP16 SM70 compute path.
- Generic Triton sparse MLA fallback and FP8 KV dequant/gather from the
  official XPU implementation, adapted for SM70 HMMA and the existing 1Cat
  metadata interface.
- Official DeepSeek TileLang algorithms for sparse attention and mHC. The
  official kernels already disable TMA and warp specialization; SM70 still
  requires FP16 dtype and launch-shape adaptation.
- Existing 1Cat TurboMind SM70 kernels for dense FP8/MXFP4. Expert MXFP4 must
  use a TurboMind implementation; Marlin and full-weight dequantization are
  not accepted production routes.
- An SM70 prefill workspace-area factor of 8 replaces the upstream factor of
  32. At the checkpoint's 1M context this lowers the profile reservation from
  about 8.3 GiB to 2.1 GiB while preserving request packing within that area.

### Upstream Delta Decisions

| Official change | Decision for SM70 branch |
| --- | --- |
| `4673ca1d78` MTP projection prefixes | Backported; required for FP8 quantization and weight-name matching. |
| `e18fe932ca` actual-width prefill chunk plan | Backported into shared metadata and the SM70 gather path. |
| `f70caef48b` cached token-to-request mapping | Backported; compressor, SWA, and sparse metadata builders share one mapping per batch. |
| `37e370fe93` skip empty C128 compression | Backported outside full CUDA graphs. |
| `b0cb1da1bd` skip short-context indexer | Backported; all candidates are selected exactly when compressed length is at most top-k. |
| `b2f9e4caa4` adaptive C128 top-k width | Backported with 128-slot alignment and packed graph-stable views. |
| `837eae6458` remove redundant combined-index fill | Backported through reusable workspace outputs on SM70 prefill. |
| `904fae8be1` allocate MTP PP buffer only when needed | Backported; avoids the large no-MTP residual buffer. |
| `442c421e79` first-layer mHC broadcast | Backported without materializing repeated embeddings. |
| `74d3b799e1` block-M mHC row-local reduction | Current FP16 kernel already has row-local FP32 accumulators; added a 1024-row carry-over regression test. |
| `df71917cf1` eager-break scratch reuse | Deferred until the V100 correctness gate. Official buffers assume BF16/CUTEDSL outputs; the SM70 FP16/software-FP8 route needs a separate lifetime audit before sharing scratch across auxiliary streams. |
| `04adc8843b` per-spec KV reshape dtype | Not copied verbatim: this branch mutates V4 FP8 to `fp8_ds_mla` before cache allocation and its runner uses a different reshape API. The backend's 584-byte shape has a static regression test; validate physical cache binding on V100. |
| `ba18929079` bucketed MTP completeness guard | Not applicable; this base has no bucketed-update validation module. |
| Post-v0.26 Quark/ROCm/Marlin MXFP4 changes | Not applicable to the exact-SM70 TurboMind packed-expert route. |
| FlashMLA/FlashInfer/CUTLASS V4 kernels | Not applicable to SM70; only algorithm and metadata fixes are reused. |

## Known SM70 Blockers

1. Sparse attention selects NVIDIA FlashMLA, which supports SM90/SM100 only.
2. The fused CUDA qnorm/RoPE/KV-insert op intentionally errors on SM70.
3. Main MLA currently requires the FlashMLA backend and FP8 DS MLA cache.
4. The O projection uses FP8 einsum kernels unavailable on SM70.
5. TileLang 0.1.9 does not compile its packed BF16 fallback on SM70. CUDA uses
   TileLang/TVM-FFI 0.1.10, and V100 model execution must explicitly use FP16.
6. MXFP4 expert MoE has no registered TurboMind SM70 route.
7. Official upstream fixes after the current 1Cat import must be audited for
   TP, MTP, prefix-cache, packed-KV, graph, OOM, and indexer correctness.

## Implementation Slices

| Slice | Owner | Route |
| --- | --- | --- |
| Sparse attention and KV cache | Main agent | SM70 FP16 Triton/TileLang |
| mHC | Sagan | SM70 FP16 TileLang |
| MXFP4 expert MoE | Hume | SM70 TurboMind |
| Model dispatch and integration | Main agent | NVIDIA SM70 capability gate |

## Acceptance Gates

### Route

- Model registry recognizes `DeepseekV4ForCausalLM` and the MTP class.
- Startup logs explicitly report SM70 sparse attention, FP16 mHC, dense FP8
  TurboMind, and MXFP4 expert TurboMind.
- No FlashMLA, Marlin, eager-only, or full-weight dequant fallback is active.
- TP8 and PP2 x TP4 both initialize with the intended GPU topology.

### Correctness

- Q RMSNorm/RoPE and KV RoPE match a PyTorch FP32 reference within FP16
  tolerances.
- FP8 DS MLA cache pack/dequant matches the existing BF16 path and preserves
  slot mapping, prefix-cache, and exact max-length boundaries.
- C4, C128, and SWA sparse attention match a dense indexed reference.
- mHC, MXFP4 MoE, grouped O projection, TP collectives, and logits pass
  component checks before full-model generation.
- Greedy and official sampling prompts produce coherent output; MTP quality is
  compared against no-MTP output and acceptance statistics are reported.

### Performance

- Report TTFT/prefill separately from pure decode TPOT.
- Measure 1K, 4K, 16K, 64K, 128K, and 256K contexts where feasible.
- Compare TP8 with PP2 x TP4 using identical model, cache, graph, prompt,
  sampling, input length, and output length.
- Profile each major route with Nsight Systems before kernel-level NCU work.

### CUDA Graph

- C128 metadata keeps an adaptive logical width but uses the full backing
  buffer's row stride on SM70. This prevents rows after row zero from moving
  between FULL-graph capture and replay.
- SM70 sparse decode reads each row's `topk_lens` on device and uses it as the
  dynamic loop bound. Context growth therefore does not depend on a captured
  Python `EXTRA_WIDTH`, and short contexts do not scan the 1M-context capacity.
- Remote tests must replay one captured graph across C128 widths 128, 256, 512,
  1024, 2048, and 8192 with MTP M=5 and batch size at least two.

## Rejected or Deferred Paths

| Path | Decision | Reason |
| --- | --- | --- |
| Merge all of vLLM `v0.26.0` | Rejected | Conflict surface is too broad and risks existing V100 work. |
| FlashMLA on V100 | Rejected | Kernel requires SM90/SM100 features. |
| Marlin MXFP4/FP8 | Rejected | Project production route is TurboMind. |
| Persistent full dequantization | Bring-up only, not release | Excess memory and bandwidth; cannot fit the target efficiently. |
| Tune before route/correctness gates | Rejected | Performance evidence is invalid until the intended route is proven. |
| Full C4 index-key gather as final decode route | Deferred optimization | It is functional but creates an FP16 `[tokens, compressed_seq, 128]` workspace and leaves substantial long-context work. Replace it only after the component quality gate. |

## Experiment Log

| Date | Change or test | Result | Decision |
| --- | --- | --- | --- |
| 2026-08-01 | Audit vLLM `v0.26.0` and latest main | Initial V4 support exists, but CUDA fast paths remain SM90/SM100-centric. Official XPU Triton fallback is reusable on SM70. | Backport the dependency-closed V4 subsystem and add explicit SM70 routes. |
| 2026-08-01 | Refresh official main from `39f55ffdaa` to `dc818c198d` | Three new commits affect GPT-OSS, AMD ownership, and registry typos; none touch the audited V4 dependency closure. | Keep the scoped backport unchanged. |
| 2026-08-01 | Audit remote topology | Two four-GPU NVLink islands with cross-island `SYS` links. | Validate both TP8 and PP2 x TP4. |
| 2026-08-01 | Targeted CPU/static tests | Combined V4 route, metadata, software FP8, mHC, TP8/TP4 MXFP4, indexer, 584-byte layout, and graph-stride suite: 34 passed, 57 CUDA-dependent tests skipped. | Local executable gates pass; GPU tests remain required. |
| 2026-08-01 | Driver-free Triton AOT to SM70 with model `index_topk=512` | 17 kernels, including MTP RMSNorm and SWA-only decode, emitted SM70 cubins; no PTX contained native E4M3 instructions. | Software FP8 and all new Triton routes are instruction-compatible with V100. |
| 2026-08-01 | AOT resource audit | Sparse prefill uses 49,152 B shared; max-width C128 decode uses 51,712 B shared. | Functionality compiles, but max-context occupancy is a remote performance gate. |
| 2026-08-01 | Standalone C++/CUDA compile | TurboMind source compiles for `sm_70` with the repository constexpr flags; Torch bindings pass C++17 syntax checking. | New MXFP4 operator and registration are build-valid locally. |
| 2026-08-01 | Independent graph-contract review | Found C128 active width was also changing physical row stride, so FULL-graph replay could read another row's old indices. | Fixed only on SM70 with a stable row stride plus a device-side dynamic length loop; preserve upstream packed layout on other GPUs. |
| 2026-08-01 | Local CUDA execution | CUDA initialization fails with error 804 because of the local driver/userspace mismatch. | Do not claim runtime correctness or speed from local AOT; run real GPU gates remotely. |
| 2026-08-02 | Remote full-core SM70 build | The 80-core build completed all main vLLM and Flash-V100 extensions; `_C`, `_moe_C`, stable libtorch, and Flash-V100 import successfully. | Reuse the built extensions for component gates instead of rebuilding the full tree. |
| 2026-08-02 | TileLang dependency isolation on V100 | TileLang 0.1.9 fails while compiling an unused BF16 `fma2` fallback. Exact TileLang/TVM-FFI 0.1.10 passes 7 focused and 43 general SM70 FP16 mHC tests; 8 ROCm-only cases skip. | Pin CUDA to 0.1.10; preserve FP16 inputs rather than silently casting BF16. |
| 2026-08-02 | Remote V4 component gates | SM70 route tests pass 9/9, TurboMind MXFP4 MoE passes 12/12, and compressed indexer slot mapping passes with the local V4 config. The DeepGEMM MegaMoE staging test is not applicable because that runtime requires SM100. | Proceed to full-model TP8 and PP2 x TP4 route and quality gates. |
| 2026-08-02 | Full-model TP8 startup | Two full-model-only interface gaps were exposed and fixed: MXFP4 post-load read bias from the obsolete `layer.moe`, and FP8 warmup passed grouped-BMM tensors without the per-group slice used by runtime. Focused tests pass 13/13 and 2/2; the real V100 grouped warmup passes M=1 and M=4. | Keep the full warmup enabled; do not bypass either failure. |
| 2026-08-02 | TP8 route gate | Non-eager FULL decode graph starts with FP16 mHC, packed-FP8 sparse MLA/KV, FP8 dense and grouped-BMM TurboMind, MXFP4 MoE TurboMind with 256 local experts, and the SM70 LM-head layout. Model loading uses 19.92 GiB per rank; no Marlin route is active. | Route gate passes for TP8; PYNCCL is used across the two four-GPU topology islands. |
| 2026-08-02 | TP8 no-MTP 1K/256 baseline | Official sampling (`temperature=1`, `top_p=1`), exact 1024-token input, and three 256-token natural outputs give mean TTFT 1.761 s and steady TPOT 134.143 ms (7.455 tok/s). Per-run interval p50/p90/p99 are approximately 133.85/136.19/136.67 ms. All three outputs are coherent and finish by the 256-token length limit. | Use this unprofiled result as the absolute baseline; use Nsight graph-node traces only for composition. |
| 2026-08-02 | Same-service TP8 output-drift localization | The exact 1024-token ID prompt (`sha256=97381166...f611`) produced dense request-to-request logits drift (`max_abs=6.2959`, `mean_abs=0.9480`, 129,189/129,280 entries changed), while all TP ranks agreed within each request. Layer-0 `hidden/qr/kv/q` were bitwise stable. The gathered KV differed in 1,649 values across 31 tokens, exclusively in RoPE dimensions 448-511; affected values alternated between rotated and unrotated KV. | Root cause is the SM70 qnorm/RoPE kernel copying all 512 KV dimensions and then writing RoPE over 448-511 from different Triton lanes without synchronization. This is a write race, not sampling or quantization noise. |
| 2026-08-02 | Remove overlapping SM70 KV stores | Restrict the initial KV copy to NoPE dimensions 0-447, leaving RoPE dimensions 448-511 with a single writer. An exact-SM70 regression with 1,024 tokens, eight heads, and four physical block ranges is bitwise repeatable and matches the FP32 RoPE reference after the FP16-to-BF16 cache round trip. | Keep the disjoint-store implementation; never restore an overlapping whole-row copy as a micro-optimization. |
| 2026-08-02 | Post-fix TP8 no-MTP quality gate | Two explicit-token-ID requests have identical top-20 logprobs; all captured tensors across 43 attention layers are bitwise equal. Two 256-token greedy continuations have the same text hash (`63254475...a52`). Official-template sampling is coherent, and a 1,529-token HTML response stops naturally with balanced tags, no missing-angle tag lines, and JavaScript passing `node --check`. Artifacts: `/home/fudanwl/v100-worktrees/runs/dsv4-quality-rope-fix-fullmodel-20260802/`. | The no-MTP TP8 request-to-request corruption is fixed. Prefix-cache, MTP, PP2 x TP4, and long-context quality remain independent release gates. |
