# SM70 release profile and acceleration status

The SM70 wheel includes `serve_qwen38_27b_nvfp4_v100.sh` and the packaged
`qwen38_27b_nvfp4_dflash2` profile. The launcher reads the profile from its
installed Python environment. Extra command-line options override its values.

```bash
python -m pip install ./1cat_vllm-1.5.1-cp312-cp312-linux_x86_64.whl
serve_qwen38_27b_nvfp4_v100.sh /path/to/target
serve_qwen38_27b_nvfp4_v100.sh /path/to/target --draft /path/to/downloaded-draft
python -m vllm.sm70_profiles show qwen38_27b_nvfp4_dflash2 --json
```

This recipe targets four peer-connected V100-SXM2 16GB or 32GB GPUs, TP4, CUDA
12.8, Torch 2.10.0+cu128, FP16 target/draft compute, FLASH_ATTN_V100, a
262144-token context, a 4096-token prefill budget, four sequences, 0.86 memory utilization,
2048-token KV blocks and 2048-token mamba blocks. Model weights are downloaded
separately; pip installs the declared Python and CUDA runtime dependencies.
No source overlay or private native extension is required.

The first launch downloads approximately 3.85 GB of draft weights unless
`--draft` selects a local checkpoint directory. This option keeps the profile's
sampling and acceleration settings and removes the remote revision argument.

The release owner selected E4M3 KV for the profile and the main qualification
path. It admits the grouped, long-context and scalar-tail capabilities.
Performance and output-quality qualification remain release gates.
An explicit `--kv-cache-dtype fp8_e5m2` override reports `kv_dtype` for these
E4M3 paths and is not the profile's accelerated long-context baseline.

The draft uses a fixed Hugging Face revision. The profile also records the
corresponding ModelScope commit and configuration/weight SHA256 values: commit
IDs belong to different providers, while the two checkpoint payloads match.

At configuration completion, before graph capture, the engine logs configured
SM70 route capabilities and final switch values. Disabled expected paths emit
warnings with a reason. `GET /v1/sm70/acceleration` returns the same report and
uses the server's existing API-key authentication. A configured capability is
not proof that a particular request executed a kernel; retain worker route
logs and the graceful-shutdown route summary for performance qualification.
Worker shutdown explicitly flushes route counters, because multiprocessing
workers can exit without invoking Python's `atexit` hooks. Counters cover Python
dispatch and graph capture; CUDA graph replay does not increment them.

For a selected, qualified profile, set
`VLLM_SM70_REQUIRE_PROFILE_ACCELERATION=1` to refuse startup when an expected
path is disabled. It remains off by default. On other platforms paths are
reported as `not_applicable`.

The 1.5.1 release requires compilation caching to be enabled by default, without
user environment variables. Removing the two existing SM70 cache opt-outs is
tracked in [#621](https://github.com/1CatAI/1Cat-vLLM/pull/621), separately from
this profile and status implementation. The final wheel must pass cold/warm
output-quality checks before that default is qualified for release.

The standard cache location is `~/.cache/vllm`, or `$XDG_CACHE_HOME/vllm` when
configured. Compilation artifacts are reused for matching configuration,
compiler, environment and source hashes. Each process still loads model weights
and captures its CUDA graphs; enabling the cache does not eliminate all startup
work. An explicit `VLLM_DISABLE_COMPILE_CACHE=1` remains a troubleshooting opt-out.
Compilation caching and AOT FX-graph serialization are separate controls. The
tested E4M3 release contract reuses compiled subgraphs while rebuilding its FX
graph, preserving CUDA graphs and avoiding the failed AOT-reload output gate.
The corresponding default-policy fix and #621 must be included in the final
wheel; the profile/status change alone does not implement those defaults.

The report's `expected_acceleration` lists the paths required by this recipe.
Use that list when computing enabled/total counts. Flash-Next FP16 MoE decode
is a different model contract and is `not_applicable` to this dense 27B recipe.
`compile_cache_disabled` describes cache state; it does not mean the user
disabled decode acceleration. The unmodified main baseline automatically
selects that state. It does not satisfy the release's default-cache requirement.

| Override | Expected capability affected |
| --- | --- |
| `--kv-cache-dtype fp8_e5m2` | E4M3 grouped FP32, long-context and scalar tail: `kv_dtype` |
| `--max-num-batched-tokens 6000` | Q8000 prefill: `budget<8000` |
| `--block-size 16` | Compact scalar tail: `page_size` |
| DFlash2 with five speculative tokens | Qualified verifier defaults: `num_speculative_tokens=5≠7` |
| No speculative config | DFlash2 verifier: `method=None≠dflash` |
| `--enforce-eager` | Compile/graph capability: `user_override` |

The 4096-token budget keeps the long-prefill attention on the FP32-accumulated
79T architecture through its native Q4096 specialization once a prompt's KV
reaches 8192 tokens; the first chunks keep the exact Split-D kernel. Budgets of
8192 select the Q8000/Q8192 specializations as before. Both budgets reserve the
E4M3 prefix bridge for the full context during memory profiling. Measured on
4x V100-SXM2-16GB, the profile leaves 4.07 GiB of KV per GPU against 3.33 GiB
for 8192-token chunks at 0.92 memory utilization; time to first token is within
1% up to 32K, +1.6% at 128K and +3.9% at 250K tokens, and decode rounds are
unchanged.
Prefix caching runs in mamba `align` mode, which can resume a cached prefix
only where a GDN state was saved: at multiples of `mamba_block_size`. With
8192-token blocks a conversation shorter than 8K never hit the cache and longer
ones recomputed up to 8K tokens every turn. 2048-token blocks (the KV page size,
and the smallest valid value) keep the KV capacity unchanged (425,473 tokens on
4x16GB) and decode rounds identical, cost about 1.5% of uncached 32K-128K
prefill time, and in an eight-turn agent conversation (shared 2.6K system
prompt, ~1.3K new tokens per turn) raise the prefix-cache hit rate from 28% to
68%. Four such conversations at once finished in 36.4 s instead of 57.7 s, with
the median time to first token falling from 2.34 s to 1.34 s.

The profile also enables partial prefix-cache hits
(`--partial-prefix-cache`). The chat API passes two prompt positions to the
engine: the end of the last message (before the generation prompt) and the end
of the system prompt, each rounded down to the 64-token GDN chunk. When a
prefill step reaches one of them, the partially filled KV pages, the sliding
window blocks of the draft and the recurrent states at that position are copied
into cache-owned blocks. A later request that matches the full blocks before it
and the tail inside the block resumes there from private copies. Reasoning
templates render a finished assistant turn differently from the prompt that
started it, so the end of the last message is also the hit point in thinking
mode; thinking requests no longer cache their output blocks, which the next
turn cannot reuse. In an eight-turn agent conversation this raises the hit to
99-100% of the reusable prefix and leaves 30-60 recomputed tokens per turn
beyond the new ones (previously about 1,000); with four concurrent
conversations the median time to first token fell from 1.07 s to 0.71 s.
Entries are bounded to a quarter of the KV blocks and released first when
running requests need blocks.
Responses report the reused tokens in `usage.prompt_tokens_details.cached_tokens`
(`--enable-prompt-tokens-details`), so clients can see their cache hits.

The profile disables the mixed-prefill latency target
(`--mixed-prefill-step-latency-ms 0`), so a prompt that arrives while other
requests decode is prefilled in full 4096-token chunks. On SM70 every mixed
step pays a fixed cost of about 130 ms, mostly launch-bound recurrent layers.
With the 250 ms target, chunks shrank to about 100 tokens: a 4K prompt then
took about 35 mixed steps at under 1000 prefill tokens/s, and the resident
requests advanced one round per step throughout. Full chunks finish the same
prompt in about one second at full prefill speed and return the batch to pure
decode. With prefix caching, most new prefills are the uncached tail of a
conversation, so the decode pause stays short. Pass a positive value to bound
mixed steps for latency-sensitive services.

The 0.86 default keeps about 690 MiB of device headroom on 16GB cards: one
250K request followed by four concurrent 64K requests completed with a 15.1 GiB
peak and 4.07 GiB of KV per GPU. At 0.92 that mix ran out of memory, and at
0.88 the peak came within 160 MiB of the allocator limit.

Some operators select their default inside the quantization layer rather than
in the global environment table. In particular, NVFP4 QPN2 decode and bounded
prefill are automatic for the compatible DFlash2 contract even when no
`VLLM_SM70_NVFP4_QPN2*` variables were exported. Larger live shapes retain
TurboMind. QPN4's target-only/single-sequence contract is a separate route;
enabling every experimental switch is not a supported serving recipe.

`benchmarks/benchmark_sm70_openai_stream.py` counts returned token IDs when
reporting TPOT. DFlash2 can emit multiple tokens in one response chunk, so chunk
intervals are reported separately. Token latency describes availability at the
HTTP client: co-emitted tokens share an arrival time, and its percentiles do not
measure individual GPU rounds. The client also supports 32K and longer input
construction and an explicit `--top-k` sampling option.
