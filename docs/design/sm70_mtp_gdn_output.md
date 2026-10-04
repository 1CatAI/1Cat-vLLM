# Direct MTP GDN output

Pure speculative verification no longer needs to copy a temporary recurrent
output into its destination. The mixed-QKV operator accepts a validated
contiguous caller buffer; measured SM70 M5/M20 pure batches use it. Mixed
batches and other shapes retain the existing output merge. The kernel,
FP32 accumulation, state schedule and output rounding are unchanged.

The ordinary package `1.5.2.dev744+g58fb205cd` passes 18 GPU checks with six
unmeasured strided fixtures skipped. Graph replays compare FP16 output and
convolution state plus every FP32 SSM snapshot bitwise against the original
split-QKV recurrence, including changing state selectors and output canaries.
All 213 installed dependencies are compatible.

`benchmark_sm70_mtp_gdn_output.py --out result.json` captures 36 distinct
state banks per graph on V100, with FP16 input and FP32 state. Median timings
use five samples of 100 graph replays following warmup.

| Tokens | Allocate and copy, µs | Direct output, µs | Saving, µs |
| ---: | ---: | ---: | ---: |
| 5 | 550.902 | 470.876 | 80.026 |
| 20 | 1081.477 | 1021.399 | 60.078 |

These are recurrence-chain measurements, not model round gains. No whole-model
restart is required for this local copy screen.

Whole wheel SHA256:
`432e081f3c7823d02df1b09ae63729031358a58e4da5e8fa792a34a29c4cfef3`.
Core SHA256:
`6274c6e4e17e14d18e5e1f7e96c432e3410ef33947955afca7f81456505b43d5`.
