# Replicated HC weight lookahead screen

The M5 replicated HC up counter sample reports 51.6% L1TEX issue-stall
contribution, memory throughput 412 GB/s and approximately four active warps
per SM. This motivates screening one-group weight lookahead while retaining
the original FP32 MMA sequence, FP16 gate boundary and branch reduction.

The ordinary package `1.5.2.dev744+g5f170b534` contains the full core and passes
all 213 dependency checks. Eight real Flash-Next HC pairs at layers
0/12/24/47 cover both attention and FFN. M5 and M20 use identical input seeds,
five samples of 100 graph replays and three changing-input snapshots. All
FP32 partials, FP16 low-rank values, output and injection compare bitwise.

| Tokens | Original chain µs | Lookahead chain µs | Decision |
| ---: | ---: | ---: | --- |
| 5 | 202.076 | 202.588 | No material improvement |
| 20 | 263.598 | 303.094 | Reject regression |

The candidate is reverted. These sequential-process operator screens do not
isolate clock drift, but provide no evidence to admit lookahead. No model run
was restarted for this experiment. Counters and instrumented duration remain
separate from the unprofiled graph timings.

Candidate wheel SHA256:
`96f3dbfbf626f436068efbef0d945bfaf4741ff4b03d469acc6ff49b9f02dbf8`.
Candidate core SHA256:
`49f6eeb27d411f02ce2ebc0ca0c31837ae45c6a57145e0024df9e7163ef87645`.
