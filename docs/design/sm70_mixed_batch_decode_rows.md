# Resident decode rows inside mixed prefill batches

A chunked-prefill step can carry resident requests next to the prefill chunk:
plain decodes and DFlash2/MTP verification spans. Two layers of the SM70
Qwen3.8 stack used to drop those rows off their fast routes whenever a prefill
chunk was present, so a long prompt arriving at a busy service made every
resident request pay a much larger per-step cost than a pure decode step.

## Attention: grouped FP32 route for E4M3 verification rows

Mixed batches already pull the small-query rows out of the paged-prefill loop
and run them on a decode operator (`VLLM_FLASH_V100_PREFILL_PREFIX_DECODE_ROWS`,
on by default). For a DFlash2 E4M3 target that decode operator was the scalar
paged decoder: XQA is excluded for selector targets because it keeps half
precision partials, and the dedicated grouped FP32 operator only accepted a
batch that consisted of nothing but verification rows. A verification span of
eight tokens therefore read the whole KV eight times, and the cost grew with
the context.

The rows are now gathered into request-major groups of eight and run on the
same grouped E4M3 FP32 operator a uniform verification batch uses:

* one group per request, or per eight-token slice of a longer span; every row
  keeps its own causal length, so slicing is exact;
* groups are padded to eight rows with length zero, which the operator defines
  as zero output, so padding never reads the cache;
* at most sixteen groups go to one launch;
* layouts the operator does not admit fall back to XQA or the scalar decoder
  exactly as before, and nothing is written before admission succeeds.

Row lengths used to be derived from the host copy of `seq_lens`. Under async
speculative decoding that copy may only be an upper bound, which would have
moved the causal boundary of every verification row. They are now derived on
the device from the authoritative `seq_lens`.

The host side of the layout (row selection, token indices, group table) is
built once per step and cached on the step's metadata, instead of once per
attention layer.

## GDN: packed verification beside prefill chunks

The packed DFlash2 GDN verifier and the mixed-QKV verifier required a batch of
speculative requests only. With a prefill chunk present, verification fell back
to the generic gather/rearrange/recurrence sequence. The verifier only touches
the speculative requests, whose tokens and state slots are described by the
`spec_*` metadata, so the restriction is lifted: operands are gathered by
`spec_token_indx`, the verifier fills a scratch buffer, and the result is merged
back by token index like every other mixed-batch output.

## Graphs

A step with more than 32 tokens is outside the captured shapes and runs
without CUDA graphs. That is a property of mixed steps and not a gating bug:
the per-step host cost is unchanged here, and the two changes above remove
device work from the eager step.

## Measurements

Qwen3.8-27B NVFP4 with DFlash2 (E4M3 KV, page 1648, TP4 on four V100), one
GPU for the operator timings.

Attention rows of a mixed batch (four resident verification spans of eight
tokens next to a 2048-token chunk), per layer:

| context | scalar (before) | grouped FP32 (now) | uniform verify batch |
| ---: | ---: | ---: | ---: |
| 32768 | 6.01 ms | 1.10 ms | 1.37 ms |
| 131072 | 22.70 ms | 4.08 ms | 6.93 ms |

The grouped route is 5.5x cheaper at both lengths. The uniform column is the
same rows without a prefill chunk, run eagerly through the existing
verification entry point; it is listed for scale only.

End to end, two resident decoders and two 32K prompts admitted together,
three repetitions each. Resident throughput while the prompts prefill
(sum of both residents, tokens per second):

| prefill chunk per request | before | after |
| ---: | ---: | ---: |
| 8192 (default) | 5.5-8.6 | 5.9-10.9 |
| 1024 | 9.0-10.7 | 12.2-19.5 |

The default-chunk difference is inside the run-to-run spread. A step there
is dominated by the prefill chunk itself (about 2.9 s, the longest resident
stall is unchanged at 2.87 s), so shaving tens of milliseconds of attention
off the resident rows cannot show. With smaller chunks the resident rows are a
larger share of each step and the change is visible (+56% on the mean), but
the residents still only advance once per mixed step. Prefill throughput and
the four-prompt wave are unchanged (3.29K and 3.34K tokens/s at 32K, last
first token 39.7 s and 39.5 s), and the pure-decode window is not slower.

What these fixes do not do: a resident request still advances one speculative
step per mixed step, so its rate while a long prompt prefills is set by the
step time. Keeping that near pure-decode speed needs decode-only steps to be
interleaved with prefill chunks, which is a scheduling decision and is not
part of this change.
