# Flash-Next GGUF MTP4 baseline

The first workload combines Flash-Next GSQ-RCO IQ3_S with the original MTP
checkpoint loaded through the existing FP16 path. The independent draft
safetensors loader must not inherit the target GGUF quantization method.
The target embedding and Q6_K head are shared with the drafter.

| Setting | Initial comparison |
|---|---|
| GPUs | Four V100 SXM2 32 GiB, 300 W, direct NVLink ring |
| Tensor parallelism | 4, including experts |
| CUDA / Torch | CUDA 12.8 / Torch 2.10.0+cu128 |
| Target / draft storage | IQ3_S GGUF / original BF16 checkpoint cast to FP16 |
| Activation / KV / recurrent state | FP16 / FP16 / FP32 |
| Speculation | MTP4, greedy proposals, standard rejection sampling |
| Decode graphs | FULL, confirmed by each worker |
| Context capacity / input / output | 8,704 / 8,192 / 256 tokens |
| Prefill batch budget / sequence slots | 512 / 4 |
| Memory utilization / prefix caching | 0.90 / disabled |
| Natural prompts | Greedy, EOS enabled, separate from timing |
| Timing prompts | Fixed length, greedy, EOS ignored, C1 and C4 |
| Collective comparison | Ring disabled versus automatic admission |

Shorter context capacity makes the initial startup cheaper. It is not the
final 256K capacity measurement. Loading, compilation, graph capture and
warmup are excluded from the decode intervals. The engine timestamps measure
complete emitted-round intervals; they do not isolate target GPU forward
time. Speculative metric deltas report mean acceptance length and per-position
acceptance rates. CUDA-event and Nsight diagnostics must retain their separate
measurement scopes when target, draft, sampling and overlap are attributed.

The clean installed baseline package uses source `d5490b095d` and wheel SHA256
`0fbe03b1448a3db4b3de44d593a050ec3a4b7d5bea0894202b23b2c90f888553`.
Its source-built native core is unchanged from the formal ring package:
`0bb550c1cde2a901f08224ef01b70f7abe52134c4996dd30c7cb7b6b20e8f933`.
The adapter, draft configuration, loader and collective Python modules match
the installed package byte for byte. No private extension override is used.

The first process loaded both models at 18.99 GiB per rank and captured the
graphs, then blocked during V2 execution warmup: FULL hybrid PLE skips the
remote request but the embedding still waited on its semaphore. Reuse the
local/cascade fix from #821 (`cc936d801e`); cascade rows retain their wait.
Both regression cases pass from source and the installed wheel. The corrected
source is `3769638467`, with wheel SHA256
`2ac9d1da0f4355a9447c274793947223ea423ac217de4fb600eb69b5a64c19c5`;
the native core remains unchanged. The failed process supplies no latency
or acceptance result.

## Initial measured C1

The ring-disabled 8K-input greedy cohort records 57 trimmed steady intervals:

| Measurement | Result |
|---|---:|
| Complete engine round, mean | 46.822 ms |
| Pure decode | 71.566 tok/s |
| Mean acceptance length | 3.427 tokens |
| Draft rounds / proposed / accepted tokens | 75 / 300 / 182 |
| Position 1 / 2 / 3 / 4 acceptance | 0.800 / 0.640 / 0.560 / 0.427 |

All four natural prompts stop normally: Paris, arithmetic, Chinese translation
and a complete Chinese explanation. This measures the complete engine
interval, not the target-only CUDA-event phase. The larger-capacity and
multi-prompt acceptance comparison is still pending.

At batch budget 512, four 8K prompts prefill in succession and requests finish
before a steady four-request decode cohort forms. That case has zero valid
C4 intervals and supplies no C4 speed evidence. The C4 smoke therefore uses
128 input tokens per request in both arms, retaining the same capacity,
sequence slots, graph policy and model. C1 continues to use 8K input.

The Flash-Next ring comparison remains pending. The 27B C1 improvement is a
smoke check and is not a Flash-Next
latency estimate. Ring merging requires matched Flash-Next C1/C4, output
identity and natural termination. Follow-up optimization starts with the
actual trace and packed PLE pinned-UVA reading.
