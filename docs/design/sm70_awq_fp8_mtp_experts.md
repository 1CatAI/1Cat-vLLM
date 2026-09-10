# SM70 FP8-resident MTP experts with AWQ or NVFP4 targets

Qwen4Exp MTP experts can retain FP8 storage independently of the AWQ or NVFP4 target. The implementation accepts both unquantized MTP experts for opt-in conversion and serialized block-FP8 experts with their original scales. The target model, draft attention/router, embeddings, output head and KV precision retain their existing configuration.

The checkpoint route backports [vLLM #55513](https://github.com/vllm-project/vllm/pull/55513): ModelOpt `FP8_PB_WO` / `FP8_BLOCK_SCALES` dispatch and AMD/NVIDIA MTP quantization-metadata remapping. This PR adds SM70 storage and TP alignment to that upstream loading support. It does not use the resident-FP16 fallback from [1Cat #553](https://github.com/1CatAI/1Cat-vLLM/pull/553).

**Validation status:** the online AWQ results below are complete. Checkpoint-native CPU loading and real-weight V100 kernel checks have passed; complete-model checkpoint-native AWQ and NVFP4 comparison runs are still in progress. The online results do not establish NVFP4 or native-checkpoint end-to-end acceptance.

## Configuration

Add `"mtp_expert_quantization": "fp8"` to an existing MTP speculative configuration, for example:

```json
{"method": "mtp", "num_speculative_tokens": 3, "mtp_expert_quantization": "fp8"}
```

This flag is for unquantized MTP experts in AWQ or ModelOpt checkpoints. Serialized FP8 experts use the checkpoint-native route automatically on SM70. An independent FP8 MTP checkpoint can be selected using the existing speculative `model` and `quantization` fields:

```json
{"method": "mtp", "num_speculative_tokens": 3, "model": "/path/to/fp8-mtp-checkpoint", "quantization": "modelopt_mixed"}
```

The SM70 adaptation requires FP16 execution, the `Qwen4ExpMTP` draft architecture, ordinary tensor parallelism, pipeline-parallel size 1 and standard rejection sampling. Expert parallelism and synthetic acceptance are rejected. Checkpoint-native weights must use E4M3 with 128x128 block scales and separate per-expert gate/up/down tensors. Excluded unquantized experts retain their original method unless online conversion is explicitly requested. Other GPU architectures retain upstream dispatch.

## Storage and kernel adaptation

The ordinary loader first loads and shards FP16 expert matrices. Each shard is quantized per output row to E4M3, with scales rounded to the FP16 precision consumed by the existing SM70 weight-only FP8 kernel. Packing removes the unpacked weights; execution retains packed FP8 bytes and FP16 scales.

For the tested TP4 model, the expert intermediate width is 160. Both gate/up halves and the down-projection input are zero-padded to 256 before quantization. Unpadded K=160 produced incorrect native GEMM results; merely accepting the layout in the packer is insufficient. The padding is therefore a correctness requirement, with a storage cost.

The online route temporarily needs FP16 weights and conversion buffers. The checkpoint-native route allocates FP8 expert parameters directly and retains the original quantized bytes. At TP4, logical slices begin at offsets 0/32/64/96 within their original 128-wide blocks. Leading zero padding preserves these coordinates in both gate/up rows and down columns, keeping the original scales correctly associated without requantization. Each rank uses a physical width of 256; ordinary vLLM TP loading handles the expanded checkpoint tensors.

The reused SM70 kernel stores scales as FP16. Source scales that become infinite or underflow to zero are rejected. The original BF16 scales in the real-weight probe are exactly representable in FP16; FP32 source scales may round to kernel precision. No reduction in loading peak or increase in speed is claimed.

## Answer-quality contract

The reference is the same AWQ or NVFP4 target, not an unquantized target. Only draft probabilities change. With standard rejection sampling, a proposal drawn from the actual draft distribution q is accepted with probability min(1, p/q); rejection uses the normalized positive part of p-q. This preserves the target distribution p in exact arithmetic regardless of draft quality. Greedy verification uses the target argmax. This change does not modify either verification algorithm or the proposal-probability handoff.

Reduced draft acceptance is allowed. A different random sample with the same seed does not imply a different output distribution. Conversely, finite logprobs or a few correct answers alone do not establish distribution preservation. Finite-precision kernels and batching can also change greedy text, so literal equality is reported separately from answer correctness.

## Validation

Tests use the native SM70 runtime from commit `752f86495f`, with the changed Python modules overlaid from this branch based on `fe67339ddf`. This is not a clean rebuild of all native extensions from the branch. The reused FP8 MoE implementation matches the branch base. The separate output-head-sharing fix is absent from both comparison arms.

- CPU: `tests/models/qwen4_exp/test_mtp_fp8_experts.py`: 12 passed. Covers TP4 shapes, scale rounding, finite/zero rows, input preservation, scoped dispatch, padding equivalence and unsupported configuration guards.
- V100: `tests/models/qwen4_exp/test_mtp_fp8_experts_gpu.py`: 4 passed, M=1/2/8/64, E=16, H=2560, unpadded I=160, top-4 routing. Compares the actual method against explicit reconstruction of its quantized experts and verifies identical CUDA Graph replay. The final committed test and quantizer were rerun successfully (4 passed).
- Additional V100 probe: padded W13 [512,2560] and W2 [2560,256], GEMM M=1/2/4/8/16/64 and CUDA Graph passed.
- Existing rejection-sampler test functions executed against the native runtime: adversarial stochastic draft distributions at speculative lengths 1 and 3, 200,000 trials each; corresponding greedy verification and calibrated nucleus checks passed. These isolate sampler behavior and are not an end-to-end benchmark.

Full-model results from matched testing are recorded below. No throughput improvement or broad benchmark-quality guarantee is inferred from the smoke suite.

### Checkpoint-native validation in progress

- CPU: 47 tests passed across the online and checkpoint suites. The checkpoint suite covers original byte/scale preservation at TP1/2/4/8, invalid scales, both ModelOpt block-FP8 names, excluded layers, metadata remapping in both backends, complete weight/scale streams and normal TP loading under AWQ/ModelOpt NVFP4/mixed config names.
- V100: the expanded GPU suite passed 20 tests: four online cases and sixteen checkpoint cases covering TP4 offsets at M=1/2/8/64. All compare reconstructed reference weights and exact CUDA Graph replay.
- Two original experts from `nvidia/Qwen3.8-Flash-Next-NVFP4` revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` passed the real loader/packer/kernel in four separate V100 processes at M=1/2/8/64. Maximum absolute error versus FP16 reconstruction was 0.0008544921875, relative L2 below 0.00082; graph replay was exact. The original BF16 scales were exactly representable in the kernel's FP16 scale format.
- A compact draft containing unchanged source MTP, embedding and head tensors loaded alongside the AWQ target and completed 16 greedy plus 16 stochastic requests. The same-source FP16 expert control and NVFP4 target comparisons are pending. Whole-GPU memory for this first native AWQ run is recorded separately; no paired native-checkpoint saving is established yet.

The native full-model run preceded two additional input guards (excluded ModelOpt layers and out-of-range kernel scales), while the 20 GPU tests use the current code. Neither guard changes numerical execution for the validated checkpoint. No native-checkpoint throughput or loading-peak claim is made.

### Runtime weight inspection

All four TP ranks reported the same values after packing. The original FP16 `w13_weight` and `w2_weight` attributes were absent; a validation-only observer asserted this without changing the quantizer.

| Per-rank expert payload | MiB |
| --- | ---: |
| Original FP16 W13 + W2 | 1200 |
| Packed FP8 W13 + W2 | 960 |
| FP16 scales for the packed weights | 15 |
| Packed weights plus scales | 975 |
| Weight-and-scale saving versus original FP16 | 225 |

Pointer tables, layout metadata and execution buffers are excluded from this payload table. Full GPU usage is measured separately and must not be equated with weight compression alone.

### Full-model paired run

The same Qwen3.8 Flash-Next Uncensored AWQ-g32 checkpoint was run on four V100 32-GiB GPUs, TP4, FP16 activations/KV, MTP3, concurrency 2, CUDA Graphs and configured context length 262144. KV capacity was fixed to 5 GiB per rank, so freed memory could not be consumed by automatic KV sizing. The actual longest test prompt was approximately 7700 tokens; this is not a full-256K-context validation.

| Measurement | FP16 draft experts | FP8 draft experts |
| --- | ---: | ---: |
| Idle whole-GPU usage after greedy requests, every rank (MiB) | 30992 | 29962 |
| Idle whole-GPU usage after stochastic requests, every rank (MiB) | 30992 | 29962 |
| Complete greedy responses matching the baseline | reference | 16/16 |
| Stochastic/top-p requests with finite token logprobs | 16/16 | 16/16 |

The observed whole-GPU reduction is 1030 MiB per rank. Only 225 MiB is the weight-and-scale payload reduction above; the rest has not been individually attributed across backend workspaces, allocator reservation and other runtime overhead. These are post-request idle snapshots, not loading peaks, and are specific to this configuration.

Greedy prompts covered arithmetic, probability, Python/JavaScript/SQL, JSON, Chinese/English, translation, summarization and long-key retrieval. Two initially truncated responses were rerun in both arms with a 1024-token limit until natural stop. The final comparison includes the full response choices and token usage, not just matching prefixes. An earlier FP8 run differed from the baseline in two explanatory passages; rerunning those two prompts on the FP16 baseline reproduced the FP8 passages exactly. This establishes an existing reproducibility caveat rather than universal bitwise invariance.

Stochastic requests used temperature 0.8, top-p 0.9, seeds 0 through 15, concurrency 2 and output caps 1/7/64/256. They exercised rejection and subsequent requests with changing lengths without request errors, non-finite returned logprobs or a stuck engine. Different same-seed text is expected when the proposal distribution changes; this smoke suite alone is not a statistical proof of the full-model output distribution. The preservation argument additionally depends on the unchanged target, actual proposal probabilities and standard rejection algorithm described above.

Acceptance counters for the greedy campaign were 1106/1449 (76.33%) for FP16 and 1109/1446 (76.69%) for FP8. This small workload does not establish an acceptance-rate or throughput advantage.
