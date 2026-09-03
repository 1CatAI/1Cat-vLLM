# SM70 Flash-V100 paper ablation campaign

## Scope

This campaign reconstructs submission-grade ablations for the Flash-V100
long-context D256 GQA prefill paper. It does not change the production route in
this revision. The first step replays frozen operator artifacts to validate
component effects and identify reproducibility gaps before rebuilding all
variants from one source base.

## Ownership and source

- Canonical repository: `/home/ymzx/桌面/1cat-vllm/vllm`
- Integration line: `onecat/main`
- Base SHA: `45a58ab6749096248dc15b1263bdf5faf51f5c70`
- Branch: `agent/v100-flash-v100-paper-ablation-20260903-133941`
- Worktree: `/home/ymzx/桌面/1cat-vllm/worktrees/v100-flash-v100-paper-ablation-20260903-133941`
- Raw artifacts: `/data/minimax-h3/task-cache/flash-v100-paper-ablation-20260903`
- GPUs used: physical 0–3 only
- GPUs excluded: physical 4–7, occupied by an unrelated TP4 service

## Frozen pre-run contract

- Operator shape: Q8000/KV128000/Hq6/Hkv1/D256, causal, FP16
- Useful work: 6.094872576 TFLOP
- Driver: 580.173.02
- PyTorch/CUDA: 2.10.0+cu128 / 12.8 runtime
- Hardware: Tesla V100-SXM2-32GB
- GPU clocks: observed, not locked; the current account cannot change
  application clocks
- Timing order: paired symmetric or ABBA
- Quality: max-abs and relative-L2 operator comparison; prior whole-model
  acceptance status remains authoritative for production

## Results

### Calibrated half2 exponential

Twenty-iteration ABBA replays on four separate V100s show a per-GPU paired
latency reduction of 1.55%, 1.99%, 1.96%, and 1.23% (median about 1.76%).
The candidate relative L2 is 0.00676666 versus 0.00675797 for the control.
Absolute candidate throughput ranges from 79.96 to 84.32 useful TFLOP/s because
GPU0 sustains about 1286 MHz while GPUs1–3 sustain about 1350 MHz. Cross-card
absolute TFLOP/s is therefore not pooled. This route remains rejected for
production because the earlier whole-model long-context gate failed.

### Direct FP32 cross-block state

On GPU0, with the matching environment cuBLAS runtime, paired means are:

| Variant | Mean latency | Relative L2 |
|---|---:|---:|
| FP16-score control | 99.924 ms | 0.00626665 |
| fixed-probability unnormalized PV | 90.497 ms | 0.00626582 |
| direct FP32 cross-block state | 89.306 ms | 0.00626582 |

Unnormalized PV reduces latency by 9.43%; direct FP32 state removes another
1.32%, for 10.63% overall. This validates the direction of the historical
60.741-to-69.479-TFLOP/s stage.

### Repaired batched triangular tail

On GPU1, the repaired 1000-token batched tail reduces the paired endpoint mean
from 76.879 to 75.802 ms (1.40%) and the tail phase from 4.943 to 3.861 ms
(21.9%). Relative L2 changes from 0.00671603 to 0.00680668. A pre-fix binary
also reproduces a superficially faster result with relative L2 8.383 and is
explicitly rejected.

### Continuous prefix/tail overlap

On GPU2, continuous overlap reduces the paired endpoint mean from 74.833 to
74.009 ms (1.10%) while relative L2 remains exactly 0.00680430.

### Closed distributed-row variant

On GPU3, the M128 distributed-row candidate is about 0.20% slower (73.740 vs
73.593 ms), despite improving relative L2 from 0.00675797 to 0.00631071. It is
not promoted.

## Reproducibility finding

The frozen campaign contains binaries with different dynamic-library
requirements. Early direct-FP32 artifacts reproduce only with the environment
CUDA-12.8 cuBLAS. Later stage37 artifacts reproduce with system cuBLAS and an
explicitly preloaded CUDA-12.8 `libcudart` for the PyTorch extension. The wrong
combination causes 2–3x latency changes without a source change. The artifact
directory therefore records executable and runtime-library SHA-256 hashes in
addition to source and hardware metadata.

## Acceptance boundary

These measurements are pre-runs, not submission-final numbers. Clocks were not
locked, and the tested binaries were frozen at different historical stages.
Submission evidence requires rebuilding all variants from one source base,
recording loaded-library hashes and route hits, and applying operator,
captured-activation, logit/token, and long-generation gates.

## Next action

1. Rebuild the exact BM32, Split-D, dense-gather, and SplitKV3 lineage from a
   single base.
2. Repeat the hybrid component rows from a single base with the same ABBA
   runner.
3. Promote only quality-accepted variants to the 256K full-model benchmark.
4. Report prefill, TTFT, steady decode, memory, power, and useful/executed FLOP
   separately.
