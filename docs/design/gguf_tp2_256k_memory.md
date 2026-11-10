# TP2 GGUF 27B with DFlash2 at 256K

Measured on 2026-10-08: two V100-SXM2-16GB GPUs can serve
Qwen3.8-27B-GSQ-RCO-IQ3_S with Qwen3.8-27B-DFlash2-Q8_0 at a total length
of 262144 tokens. Target KV uses E4M3; draft KV and activations use FP16;
GDN state and accumulation retain FP32. CUDA Graph remains enabled.

## Contract

- Hardware: four V100-SXM2-16GB GPUs, pairwise NV2; only GPUs 0 and 1 are used.
- Driver 580.178.04, CUDA 12.8, Torch 2.10.0+cu128, Python 3.12.3.
- SM clocks and memory usage sampled once per second; nominal SM 1530 MHz,
  memory 877 MHz. Clocks are sampled rather than fixed.
- TP2, maximum length 262144, maximum sequences 1, prefill chunk 512,
  GPU memory utilization 0.88, prefix caching disabled, language model only.
- DFlash2: seven speculative tokens, draft TP2, probabilistic draft sampling,
  FP16 draft KV, Flash-V100, draft-window split enabled.
- GGUF projection planes enabled with `all` scope and signed IQ2 nibbles;
  allreduce/RMS fusion enabled. FP16/BF16 reduced-precision reductions and
  FP16 accumulation are disabled.
- Target file: 11771546784 bytes, SHA256
  `64b53b64c7aa39f20a7e54bd80582fe595b1d745624ee8a72e92508c0326d810`.
- Draft file: 2056414816 bytes, SHA256
  `c18e800daedc59ca68fd13b6a856d795746af6d399a9279ac6a277d1d422f87e`.

## Why the earlier layout did not fit

The previous 32K setting was a configured limit. A 256K startup with the same
weight precision, DFlash2 width, 1024-token chunk and memory utilization 0.88
failed admission: 5.33 GiB of cache was required per rank but only 3.71 GiB
was available.

The target cache page initially spans 1648 E4M3 tokens. An FP16 draft page
with the same token count is twice as large. Halving the draft to 824 tokens
fails Flash-V100's 16-token alignment requirement, so page unification doubled
the target and recurrent-state pages instead:

| TP2 layout | Target tokens/page | Draft tokens/page | Physical bytes/page |
| --- | ---: | ---: | ---: |
| Previous | 3296 | 1648 | 3375104 |
| Rounded compact | 1664 | 832 | 1703936 |

Round the common physical page up to the least common multiple of each
attention format's 16-token byte stride. Recurrent checkpoints keep their
logical block grid, state shapes, dtype and speculative width. Existing
exact-divisor layouts and unsupported backends retain their previous behavior.
This saves approximately 0.6 GiB of minimum TP2 admission memory.

With only this page fix, utilization 0.95 and chunk 512 passed capacity
admission with 4.99 GiB available, then failed during CUDA Graph capture.
Increasing utilization alone therefore did not establish a working configuration.

Keeping the IQ2_S embedding packed frees another 0.994 GiB per rank without
changing decoded values. Utilization 0.88 and chunk 512 then provide a 4.88 GiB
cache pool. Logged cache specs imply approximately 4.68 GiB is required for
one full-length request, including the eight GDN state pages and bounded
draft window.

## Memory accounting

File size is a lower bound rather than the loaded-memory measurement. The two
files total 12.878 GiB; equal TP2 partitioning would be 6.439 GiB per rank.
Some learned draft tables and its context projection are replicated, and the
resident projection formats have their own scales and padding.

| Measurement | GiB per TP2 rank |
| --- | ---: |
| Previous target registered storage | 7.232 |
| Packed target registered storage | 6.238 |
| Draft exclusive registered storage | 1.399 |
| Combined registered storage | 7.637 |
| Target and draft position tables included above | 0.188 |
| Weight/quantization storage excluding position tables | 7.449 |
| Previous loading allocation increment | 9.44 |
| Packed loading allocation increment | 8.45 |
| Cache pool | 4.88 |
| Observed total GPU peak | 15.524 |

Shared target embedding and output-head tensors are counted once. The TP2
output head has one raw Q4_K representation. The loading increment also covers
resident work allocations; approximately 0.81 GiB above registered model
storage remains outside that tensor inventory. A shared 178257920-byte FP16
dequant/GEMM workspace is directly identified. The residual should not be
treated as irreducible weight storage.

Both cards peak at 15897 MiB during the long-context sweep. CUDA reports
260046848 bytes free after the boundary request; the tested configuration is
one full-length request. Concurrent short requests and concurrent full-length
requests have different state and KV requirements.

## Model results

The retrieval fixture places the same reference label at three positions in
repeated engineering notes. Retrieval uses temperature zero, normal EOS and
at most 64 output tokens. Only the exact-boundary stress case ignores EOS to
force all 64 output tokens and reach the configured total-length limit.

| Input tokens | Output tokens | Result | Request wall time |
| --- | ---: | --- | ---: |
| 131072 | 6 | `Saffron42`, normal EOS | 185.44 s |
| 261120 | 6 | `Saffron42`, normal EOS | 581.51 s |
| 262080 | 64 | Total length 262144, length termination | 586.97 s |

All three complete without OOM. DFlash2 executes during generation; its
speculation counters are present in the retained run logs. These wall times
include prefill and short generation and are not steady decode latency.
Long prefill remains slow; capacity success does not establish a prefill
performance improvement.

The long sweep uses complete-wheel source `5d1ff8bd63`. Final source
`99de68fe3c` adds the dense tied-embedding guard; this checkpoint has separate
input and output matrices. Its fresh complete wheel repeats 56 CPU checks,
four GPU row/graph checks and TP2 1K/8K plus natural-EOS model checks with the
same memory configuration. All 16 native artifact hashes match the previously
qualified native source `753ae2bca8`. Final wheel SHA256:
`a649213c5c48cc1a983d691c9352020cbb70ae454f9d6b4b8b8f00f8c99c7fa7`.
