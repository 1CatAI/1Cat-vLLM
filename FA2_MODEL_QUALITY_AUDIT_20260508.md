# FA2 Model Quality Audit

Date: 2026-05-08

## Scope

Target version: `1Cat-vLLM-0.0.3`

Model:
- `/home/ymzx/models/Qwen3.5-35B-A3B-AWQ`
- TP: 4
- dtype: float16
- GPUs: V100-class `Tesla PG503-216`, GPUs `1,2,3,4`
- chat template: Qwen3.5 official tokenizer template with `enable_thinking=False`
- `disable_custom_all_reduce=True`
- `max_num_batched_tokens=8192`

The audit uses objective, locally graded tasks. It is intended to detect model-level quality loss from the FA2 backend, not strict token parity.

## Artifacts

- Audit script: `tools/fa2_model_quality_audit.py`
- 32k FA2/Triton result: `model_quality_audit_32k_20260508/combined.json`
- 32k logs: `model_quality_audit_32k_20260508/*.log`
- 256k FA2 result: `model_quality_audit_256k_fa2_20260508/combined.json`
- 256k logs: `model_quality_audit_256k_fa2_20260508/*.log`

## 32k Backend Comparison

Run settings:
- backends: `FLASH_ATTN_V100`, `TRITON_ATTN`
- `max_model_len=32768`
- `max_num_seqs=4`
- short objective tasks enabled
- needle lengths: `8192`, `32768`
- needle depths: `0.10`, `0.50`, `0.90`
- generation: `temperature=0`, `top_p=1.0`, `max_tokens=64`, `logprobs=5`

Summary:

| Backend | Passed | Pass rate | Non-finite logprob cases | Median latency |
|---|---:|---:|---:|---:|
| FA2 | 10 / 11 | 90.9% | 0 | 0.805 s |
| Triton | 10 / 11 | 90.9% | 0 | 1.369 s |

Backend comparison:
- FA2-unique failures: none
- Triton-unique failures: none
- Cases where both passed but token output diverged: none

The only failed case was `arith_integer_4391`. Both FA2 and Triton returned `4401`; the expected answer was `4391`. Because both backends produced the same wrong output, this is a model/task failure, not an FA2-specific quality regression.

## 32k Case Results

| Case | Category | FA2 | Triton | Notes |
|---|---|---:|---:|---|
| `arith_integer_4391` | math | fail | fail | both output `4401` |
| `arith_integer_55` | code reasoning | pass | pass | exact `55` |
| `json_contract` | format | pass | pass | valid expected JSON |
| `zh_instruction_keyword` | instruction | pass | pass | exact Chinese phrase |
| `ignore_distractor_code` | instruction | pass | pass | exact `SAFE-V100-42` |
| `needle_len8192_depth0.10` | long needle | pass | pass | exact code |
| `needle_len8192_depth0.50` | long needle | pass | pass | exact code |
| `needle_len8192_depth0.90` | long needle | pass | pass | exact code |
| `needle_len32768_depth0.10` | long needle | pass | pass | exact code |
| `needle_len32768_depth0.50` | long needle | pass | pass | exact code |
| `needle_len32768_depth0.90` | long needle | pass | pass | exact code |

## 256k FA2 Long-Context Audit

Run settings:
- backend: `FLASH_ATTN_V100`
- `max_model_len=262144`
- `max_num_seqs=1`
- needle target: `258000`
- needle depths: `0.10`, `0.50`, `0.90`
- generation: `temperature=0`, `top_p=1.0`, `max_tokens=64`, `logprobs=5`

Summary:

| Backend | Passed | Pass rate | Non-finite logprob cases | Median latency |
|---|---:|---:|---:|---:|
| FA2 | 3 / 3 | 100% | 0 | 116.260 s |

Case results:

| Case | Prompt tokens | Result | Latency | Output |
|---|---:|---:|---:|---|
| `needle_len258000_depth0.10` | 257983 | pass | 116.701 s | `QA258000-10-BAJ3DTQ1` |
| `needle_len258000_depth0.50` | 257979 | pass | 114.999 s | `QA258000-50-TVUNDT4H` |
| `needle_len258000_depth0.90` | 257985 | pass | 116.260 s | `QA258000-90-31H4XF07` |

The log confirmed FA2 model paths:
- `FLASH_ATTN_V100 decode path active (paged KV kernel, CUDA-graph safe)`
- `FLASH_ATTN_V100 prefill path active (no prefix/chunked context)`
- `FLASH_ATTN_V100 prefill path active (prefix/chunked via paged-KV gather)`

## Quality Conclusion

No FA2-specific end-to-end quality regression was found in this audit.

Observed facts:
- FA2 matched Triton exactly on all 32k cases where the task passed.
- FA2 had no unique task failures versus Triton.
- The only 32k failure was shared by both backends and therefore is not attributable to FA2.
- FA2 passed all 256k long-context retrieval cases at early, middle, and late needle depths.
- No non-finite generated logprobs were observed.

This does not prove that every open-ended answer is semantically identical. It does show that, on objective instruction-following and long-context retrieval tasks, the current FA2 backend did not cause measurable model-level quality loss or obvious degradation.

## Remaining Risk

Open-ended reasoning and writing quality were not judged by a separate model or human evaluator in this run. Earlier strict token-exact tests showed FA2/Triton can diverge on ambiguous prompts, but this audit found no case where that divergence became an objective task failure.
