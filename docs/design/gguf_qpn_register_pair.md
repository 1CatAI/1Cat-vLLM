# Exact IQ3_S in the NVFP4 register-pair skeleton

This research screen separates weight decoding from execution structure. It reuses the native NVFP4 M8 register-pair kernel and replaces its reader with signed-u4 IQ3_S values plus unchanged FP16 group coefficients. The existing DMV11 IQ3_S projection is the control. Production storage and dispatch are unchanged.

## Purpose and budget

The decoded values `v = 2q - 15` are exact. A 320-byte packet holds 32 rows of K16 codes (256 bytes) and FP16 coefficients (64 bytes); coefficients from a K32 group are duplicated for its two K16 packets. Each paired real TP4 shard therefore reads 27,852,800 bytes instead of 19,496,960 bytes, about 43% more. This prototype tests whether the NVFP4 register-pair skeleton can compensate for that increase. The minimum weight-read time itself increases from 21.71 to 31.02 us at the 877-MHz theoretical memory bandwidth. A larger memory budget cannot be justified without a measured chain gain.

FP32 accumulation is retained. The skeleton uses eight contiguous warp partitions rather than the original four strided partitions, so output reassociation is measured explicitly. The GGUF epilogue retains FP16 gate/up rounding followed by FP32 SiLU/product and final FP16 output rounding. It does not introduce the NVFP4 path's intermediate FP16 SiLU round.

## Test plan

Base `c4f6245f841466782752a8c3283e4727565cf17a`; four V100-SXM2-32GB ranks, full NV2, CUDA 12.8 and Torch 2.10.0+cu128. The unchanged normal runtime is `1.5.2.dev0+g2b00cc8a38.cu128`; projections use one task-owned research JIT module. No private binary is required by serving.

Use real IQ3_S gate/up shards from layers 6, 23, 24 and 51, M=8, K=5120, N=4352 per projection. Check GPU-decoded weight bits against exact FP16 coefficients times signed values, compare projection output against official GGUF dequantization and the unchanged control at three input amplitudes, and change inputs across twenty graph replays. Four independent pairs exceed L2. Time symmetric ABBA after five seconds of sustained graph warmup and retain 50-ms clock samples. Power limit is 185 W and is not changed.

## Test result

All four ranks passed weight bit checks and graph error checks. Maximum relative L2 error was 0.000629 against official dequantization, 0.0000791 against the control, and 0.0000690 across changed-input replay comparisons. Outputs are not bitwise equal because of FP32 reduction reassociation. The candidate used 80 registers/thread with zero stack/local spills.

| Rank | Four-pair control us | Register-pair candidate us |
|---|---:|---:|
| 0 | 176.932 | 173.111 |
| 1 | 163.730 | 168.548 |
| 2 | 163.350 | 167.912 |
| 3 | 158.480 | 166.520 |

The candidate does not consistently improve all shards. Only eight pure IQ3_S gate/up layers are proven by this format, and the rank-0 gain alone amounts to less than 0.01 ms per round when weighted by that coverage. That is an isolated estimate, not model evidence. The increased read volume and additional stored bytes are not justified by this result.

## Decision

Defer this expansion/skeleton combination. Do not integrate it into serving or spend model-level A/B time on it. Reusing the NVFP4 skeleton alone is insufficient to establish a throughput gain for this packed source. The next diagnostic should collect counters for the hot projection inside the complete model: isolated-kernel counters and cold-L2 samples do not reproduce all preceding-kernel, cache and scheduling effects.

Raw compact records are in [data/gguf_qpn_register_pair_20261008.json](data/gguf_qpn_register_pair_20261008.json). Generated CUDA, the research binary, clocks and lock records remain outside Git. No complete-round, acceptance or C4 gain is claimed.

The copied paired execution comes from `nvfp4_qpn2_sm70.cu`, which retains the v100-skinny MIT attribution in `csrc/sm70_turbomind/ops/LICENSE.v100-skinny`. The existing GGUF reader/control remains under its Apache-2.0 notice.

These are historical measurements retained from [the original experiment](https://github.com/1CatAI/1Cat-vLLM/pull/1068). The prototype remains available in that branch and is not added to production. The stored generated-CUDA fingerprint does not match the current prototype generator, so these records do not validate a fresh build or current-main GPU execution. No GPU measurements were repeated when preserving this record.
