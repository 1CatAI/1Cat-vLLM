# DFlash2 QPN2 and TurboMind shared NVFP4 weights

## Scope and activation

`VLLM_SM70_NVFP4_QPN2_SHARED_WEIGHT=1` removes the separate QPN2 code
buffer from compatible SM70 TP4 NVFP4 projections. It requires a rebuild
providing `nvfp4_qpn2_prepare_scales_sm70` and
`nvfp4_qpn2_tm_dispatch_sm70_out`. Older binaries retain the existing
separate layouts and log the missing capability. The switch defaults to
zero pending broader model quality and concurrency performance acceptance.

The existing QPN2 model and shape gates still apply. The target is
Qwen3.8-27B-QUASAR-NVFP4 with DFlash2 q7. This does not enable QPN2 on
additional models or quantization schemes. Set the switch before loading;
changing it on a loaded model does not reclaim or restore weight buffers.

## Storage and dispatch

TurboMind keeps its non-interleaved SM70 HMMA884 B/Pack1 codes and FP16
scales. QPN2 stores only its packed E4M3 scales, including zero padding to
N=32 alignment. It reads the TurboMind code tensor directly. No QPN2 code
buffer, code alias buffer, or transient full QPN2 repack is registered.

Both formats preserve each E2M1 nibble. Within K=8 their physical order is
`[0, 2, 4, 6, 1, 3, 5, 7]`. For QPN lane `l`, let
`c = ((l >> 2) & 3) * 8 + (l & 3) + ((l & 16) ? 4 : 0)`.
The two words for K/16 group `g` in N/32 tile `t` are TurboMind words
`(t * (K / 8) + 2 * g) * 32 + c` and that index plus 32.
The common reader implements this address change for ordinary/gated QPN2
and the prefill dequantizer. Arithmetic, accumulation and activation order
are preserved.

Scales remain separate because TurboMind merges the global scale into FP16
while QPN2 retains E4M3 plus a separate global scale. Recovering the original
E4M3 data from rounded FP16 would change the numerical contract.

The new opaque C++ dispatcher keeps dynamic M selection outside Dynamo:

| Live M, with default prefill threshold | Route |
| --- | --- |
| 1–32 | QPN2 reading TurboMind codes |
| 33–1023 | Existing TurboMind GEMM |
| 1024 and above | Existing bounded FP16 prefill, reading TurboMind codes |

With QPN2 prefill disabled, the shared operator receives a zero threshold
and all M above 32 use TurboMind. Python still handles empty inputs and
crops padded outputs before bias. TurboMind warmup and state ownership are
unchanged.

For all 256 supported QUASAR TP4 target projections, the removed codes
total 2.835693 GiB per rank (11.342773 GiB across four ranks). The remaining
QPN2 scale allocation is approximately 0.354 GiB per rank. This is a tensor
storage calculation; allocator reservations and automatically sized KV
cache can conceal the reduction in NVML usage. Hold KV bytes fixed when
comparing total memory.

## Validation recorded on 2026-09-08

Integration base: `e5d63c51f0fcc1ddf75d229e3df06bf52df206f5`.
Torch 2.10.0+cu128, CUDA 12.8, V100-SXM2-32GB, FP16 activations;
`VLLM_SM70_NVFP4_QPN2_M16_NATIVE=1`. Actual checkpoint shards for TP ranks
0–3 were tested sequentially on one V100. These tests do not measure TP
communication or whole-model throughput.

- Nine CPU tests pass, covering shared loading without code preparation,
  old-binary fallback, original routes, output cropping, and prefill off.
- All 24 real projection shards match the old CUDA converter's code bits
  and packed scale bits, including GDN N=4120 padded to 4128.
- All 280 cases at M=1/8/16/32/33/64/135/1019/1024/4096 match FP16 output
  bits in eager execution and CUDA Graph replay after changing inputs.
  Gate/up is tested both as a linear projection and with fused SiLU.

Rank-0 graph timing brackets the shared candidate with controls compiled
in the same invocation. M8 shared/control ratios range from 0.982 to 1.044;
M16 ordinary projections are 1.091–1.123; gated M32 is 1.150. Large-prefill
ratios are 0.989–1.020. These are operator measurements under unlocked
clocks, not evidence of a model speedup. Two separated 32-bit loads cost
more than the previous 64-bit load for some shapes; retaining an explicit
opt-in exposes this memory/performance tradeoff.

A bounded experiment replaced the shared reader's two streaming loads with
read-only cached loads. All 28 rank-0 cases remained bitwise equal. Although
the K=1536 output projections improved, QKV and MLP regressed (gated M8
ratio 1.131, MLP down M16 ratio 1.167). The global replacement was rejected;
the implementation retains streaming loads. Selective cache policies are
outside this change.

The first sidecar build omitted `ENABLE_SM70_TURBOMIND`, hiding declarations
in `ops.h`; defining it fixes the build. The initial benchmark omitted the
stable activation library, so gated TurboMind dispatch failed to resolve
`silu_and_mul`; loading the matching stable library fixes the harness.
Neither failure was a shared-reader numerical discrepancy.

## Reproduce the focused operator check

Use an isolated worktree and owned GPU locks. The sidecar adds the two new
operators to an existing build and registers controls privately. Do not
load it into a build that already registers these operators.

```bash
export CUDA_HOME=/path/to/cuda-12.8
export TORCH_CUDA_ARCH_LIST=7.0
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch_extensions"
CUDA_VISIBLE_DEVICES='' .venv/bin/python - <<'PY'
from torch.utils.cpp_extension import load
load(
    name="nvfp4_qpn2_shared",
    sources=[
        "csrc/sm70_turbomind/ops/nvfp4_qpn2_sm70.cu",
        "csrc/sm70_turbomind/ops/nvfp4_qpn4_sm70.cu",
        "benchmarks/kernels/sm70_nvfp4_shared_sidecar.cpp",
    ],
    extra_cflags=["-O3", "-fvisibility=hidden", "-DENABLE_SM70_TURBOMIND"],
    extra_cuda_cflags=[
        "-O3", "--use_fast_math", "-lineinfo", "-Xcompiler=-fvisibility=hidden"
    ],
    extra_ldflags=["-Wl,-Bsymbolic"],
    is_python_module=False,
)
PY
CUDA_VISIBLE_DEVICES=0 VLLM_SM70_NVFP4_QPN2_M16_NATIVE=1 \
  .venv/bin/python benchmarks/kernels/benchmark_sm70_nvfp4_shared_weight.py \
  --model /path/to/Qwen3.8-27B-QUASAR-NVFP4 \
  --core-library /path/to/lib/_C.abi3.so \
  --shared-library "$TORCH_EXTENSIONS_DIR/nvfp4_qpn2_shared/nvfp4_qpn2_shared.so" \
  --json-out /path/to/task-artifacts/operator-results.json
```

The matching `_C_stable_libtorch.abi3.so` must accompany the core library.
The JSON records library hashes, the source base and diff hash, shard
shapes, bitwise comparisons, and both control timings.
