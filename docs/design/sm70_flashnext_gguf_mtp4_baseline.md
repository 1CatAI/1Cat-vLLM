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

## Initial ring comparison

Both C1 arms use the same installed package, model and 8K-input workload.
Four natural prompts produce identical complete token lists and stop at EOS.

| C1 measurement | Ring disabled | Ring automatic |
|---|---:|---:|
| Complete engine round, mean | 46.822 ms | 41.499 ms |
| Complete engine round, p50 / p99 | 46.303 / 61.817 ms | 41.483 / 41.788 ms |
| Pure decode | 71.566 tok/s | 75.212 tok/s |
| Mean acceptance length | 3.427 | 3.036 |
| Draft rounds / proposed / accepted | 75 / 300 / 182 | 84 / 336 / 171 |
| Position 1 / 2 / 3 / 4 acceptance | .800 / .640 / .560 / .427 | .643 / .536 / .452 / .405 |
| Trimmed timing intervals | 57 | 66 |

The observed round-mean reduction is 5.323 ms, and pure decode improves
5.095%. Acceptance differs in the synthetic timing request, so this is an
endpoint observation rather than an isolated communication saving. Complete
synthetic token lists were not retained in this initial harness; output identity
is established for the separate natural prompts. The per-round cost model
must use actual trace call counts and account for the changed acceptance.
Acceptance counters cover the full measured request, while latency and pure
decode use the trimmed steady window. In that exact timing window, emitted
tokens per request-round are 3.351 without ring and 3.121 with ring; dividing
these by their respective round means reproduces the measured pure decode.

The automatic-ring C4 smoke has 37 steady four-request intervals: 66.413 ms
per engine round, 240.918 aggregate tok/s and 3.861 mean acceptance length.
The matched ring-disabled C4 run records 66.615 ms and 253.171 aggregate
tok/s over 37 steady intervals, with 3.920 mean acceptance length. Thus the
observed engine round changes by -0.202 ms while pure decode decreases 4.840%.
The trimmed cohorts emit 624 and 592 tokens respectively: 4.216 and 4.000
emitted tokens per request-round in the timing window. All natural token
lists still match and stop normally, but this C4 point does not establish
non-regression. Retain it as a negative result and compare three repeats with
1024 output tokens, saving complete output lists before deciding admission.
The 27B result remains a separate smoke check. Ring merging requires a
passing Flash-Next C4 comparison and natural output checks.
Follow-up optimization starts with the actual trace and packed PLE pinned-UVA
reading.

## Longer C4 comparison

The follow-up retains 128 input tokens, 8,704 capacity and batch budget 512,
with 1,024 output tokens and three measured repeats per arm. Both use the
normal package from `3720df1225`, wheel SHA256
`def49fcd6c7ce3aad1150d898a65a4e461eba70c9b4147e72adcf3a2626599e9`;
the native core remains unchanged. The only runner change adds existing,
disabled-by-default diagnostic markers. Loading and warmup are excluded.

| C4 measurement | Ring disabled | Ring automatic |
|---|---:|---:|
| Pure decode, repeats 1 / 2 / 3 | 228.918 / 229.421 / 233.225 tok/s | 260.337 / 263.335 / 263.158 tok/s |
| Pure decode, total tokens / total seconds | 230.505 tok/s | 262.269 tok/s |
| Complete engine round, weighted mean | 67.673 ms | 65.624 ms |
| Full-cohort mean acceptance length | 3.285 | 3.830 |
| Steady emitted tokens per request-round | 3.900 | 4.303 |
| Steady intervals per repeat | 192 | 199 |
| Steady emitted tokens per repeat | 2995 | 3425 |

The observed weighted round reduction is 2.049 ms and pure decode improves
13.780%. Each arm has identical complete 1,024-token lists and acceptance
counters across its three repeats. All four separate natural prompts have
identical token lists between arms and finish at EOS. This longer C4 smoke
shows no throughput regression under its contract; it does not erase the
negative 256-output-token point or establish all-workload non-regression.

The forced-length timing lists differ between arms. Their first different
zero-based positions are 34, 471, 495 and 669 for the four streams, repeated
identically in all three comparisons. Differences include EOS and whitespace;
the causal numerical attribution is not established. Therefore the throughput
change includes a changed acceptance trajectory and must not be reported as
isolated collective speedup. Ring merge output identity is scoped to the four
natural EOS-enabled prompts. Distribution and long-output model quality
remain separate follow-ups. Trace attribution next measures actual call counts,
waiting and overlap against the unprofiled C1 baseline.
