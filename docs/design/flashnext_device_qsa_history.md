# Direct device QSA history on SM70

The device-history reference stores target K/V as per-vector E4M3 bytes with
FP32 scales, and draft K/V as FP16. Previously both passed through protected
hot-page resolution and miss staging before attention. Direct device reads
remove that ownership, resolution and staging dependency when the authoritative
history is already on the GPU. Host history keeps the protected reader.

The native reader groups six query heads per local KV head and decodes
per-vector E4M3 scales into the same FP16 K/V operands as the protected
reader. QK uses Volta mma884 with FP32 accumulation; softmax, PV and the
split merge retain FP32. Attention is rounded to FP16 before the FP32
sigmoid gate. Retaining FP32 probabilities changes the protected reader's
FP16 PV rounding boundary; model error is measured below.

The two attention kernels cover target M1..20, H6, D256 and selection widths
up to 4096. A per-device, per-width workspace is allocated before graph
capture. Native split states contain both maxima and denominators; M20 at
width 2051 needs 7920 floats. Unsupported shapes and host history retain
the protected reader. Speculative draft attention retains its existing path.
Target and draft caches are bound under the target worker's configuration.
Each QSA layer therefore retains its construction-time history policy and
draft identity; binding and subsequent reconfiguration cannot admit a draft
owner through the target capability. A draft reports
`speculative_draft_unqualified` even when direct device reads are requested.
`sm70_qsa_device_history` controls the capability, enabled by default for
admitted target history. Startup and runtime reports explain fallbacks.
The extension is built and installed by normal CMake/wheel registration.
The arithmetic-preserving Triton variant remains a numerical reference.

## Same-loaded-model qualification

All arms use one set of loaded Flash-Next IQ3_S weights, FP16 MTP4, TP4,
full CUDA graphs, target device E4M3 history, FP16 draft history and disk
PLE. Hardware is a full-NV2 V100-SXM2-32GB mesh at 1530 MHz, CUDA 12.8,
Torch 2.10.0+cu128. The draft graphs are unchanged. A recaptured control
reproduces all 64 aligned teacher logits and eight natural continuations
bit-exactly, ruling out recapture as the numerical difference.

The native target path matches 63/64 teacher top-1 decisions. Mean/max KL
is 0.000522681/0.009480884. Eight 600-token prompt clusters have a paired
acceptance difference of -0.0251 percentage points, with prompt-bootstrap
95% CI [-1.7922, +2.0874]. No clear decrease is detected; the interval does
not establish exact equivalence. The continuations diverge numerically.
The deterministic I8192/O256 C1 probe retains every token and 4.885714
emitted tokens/round. C4 emits 2.326271 tokens/request/round versus 2.324786.

Unobserved same-process ABBA means are C1 19.352215 to 18.760776 ms/round
and C4 38.960747 to 38.410213. Control endpoints drift by 0.724 ms in C1,
so these are provisional endpoint estimates. A separate request-internal
ABBA/BAAB diagnostic measures target GPU savings of 0.431/0.656 ms at C1
and 1.335/0.135 ms at C4 on rank 0. Within each complete 32-round block,
16 control and 16 candidate rounds are compared on the same GPU clock;
other ranks agree. Draft and exposed gaps change only slightly. GPU-event
envelopes are not substituted for ordinary acceptance ms/round.

These results supersede the earlier separate-process quality comparison
below; they do not change the historical 17.401789-ms baseline or establish
the 12-ms objective. Final production-dispatch checks are recorded separately.

## Installed production-dispatch validation

The final comparison uses runtime source `eca6e737038e6136b08a787ecd67ab0a2f97ecdb`
and benchmark revision `eb552a12a849934196860c8252550191ab007182`. The normally
installed wheel is `1.5.2.dev1434+geca6e7370.cu128`, with SHA256
`dd24b1c100b484653149081cb26725077ebd4d3b7292b987daefeb0424156861`.
All 18 native modules have recorded build provenance and installation hashes;
no diagnostic replacement library is loaded. The integrated suite passes
137 GPU cases: 22 direct-history, 16 scorer, 98 host-history, and one four-rank
HCX graph test. Fourteen CPU cache-policy cases also pass.

The hardware and history placement match the same-loaded-model setup above.
Maximum length is 9216, the prefill budget is 512, maximum sequences are four,
and GPU memory utilization is 0.95. The benchmark asserts target/draft owner
identity after binding and rejects explicit native admission for every draft
owner. It changes production capability state before target graph capture;
the original draft graphs are reused throughout.

Both control recapture and restoring the original graphs reproduce all 64
teacher logits, eight natural outputs and eight acceptance counters exactly.
With both target QSA optimizations enabled, teacher mean/max KL is
0.000451698/0.004927856 and top-1 agrees at 62/64 positions. Maximum absolute
logit difference is 1.278320 and maximum relative L2 difference is 0.133410.
All eight 600-token continuations differ, so this is not a bit-exact path.
All three short Chinese/English completion checks reach a normal stop in
every arm and produce coherent answers.

Mean draft acceptance is 44.7423% for control and 46.1643% with both changes.
The paired eight-prompt difference is +1.4219 percentage points, with 95%
prompt-bootstrap interval [-0.7449, +3.5926]. No clear decrease is observed;
the interval does not prove equivalence or an acceptance improvement.

Uninstrumented acceptance probes use I8192/O256 for C1 and I128/O600 for
C4. Each entry is the mean of two control and two candidate cohorts in ABBA
order, after warming both arms:

| Target change | C1 control/candidate ms | C4 control/candidate ms |
| --- | ---: | ---: |
| Native device-history reader | 19.7791 / 19.0889 | 39.4380 / 38.9911 |
| Native reader and shared-key scorer | 19.8134 / 18.7204 | 41.1333 / 39.2898 |

Native-only C1 saves 0.6902 ms; both changes save 1.0930 ms. Their C1 control
endpoint drifts are 0.0967 and -0.0470 ms. All C1 IDs agree, at 4.885714
tokens/round. C4 emits 2.334764 tokens/request/round for control and 2.326271
for the candidate, with different continuations. C4 control endpoints drift
0.8998 ms for native-only and 4.2775 ms for both changes; their apparent mean
savings are not treated as established C4 gains. Repeated C4 measurements
are tracked with the [shared-key integration](https://github.com/1CatAI/1Cat-vLLM/pull/1127).
This device-E4M3/disk-PLE comparison does not replace the historical
17.401789-ms resident-FP16 baseline or establish the 12-ms objective.

## Initial FP32-probability experiment

The initial native experiment used Volta mma884 for FP16 QK operands with
FP32 accumulation, then retained FP32 probabilities and scalar PV FMAs. It
changed the protected reader's FP16 probability boundary despite using the
same decoded K/V values. The measurements below describe that experiment,
not the arithmetic-preserving revision.

V100-SXM2-32GB, CUDA 12.8, Torch 2.10.0+cu128, one card, same-process graph
ABBA. Twelve independent target histories and one shared draft history use
causal selections, 512 selected page4 groups and an open tail. Long-context
queries share 350 selected pages per request; these are synthetic selectors,
not replayed model activations. Query standard deviation is one during timing.
Each of four draft calls reuses the same FP16 history. Workload M5/M1/M1/M1
and M20/M4/M4/M4 matches C1/C4 attention row counts.

| Chain | Protected reader us | Direct reader us | Difference us |
| --- | ---: | ---: | ---: |
| Target twelve M5 calls, context 8448 | 841.168 | 489.296 | -351.872 |
| Target twelve M20 calls, context 728 | 1287.296 | 953.984 | -333.312 |
| Target M5 plus four draft calls | 1080.464 | 565.024 | -515.440 |
| Target M20 plus four draft calls | 1537.904 | 1090.304 | -447.600 |

The complete attention chain drops from 64 to 32 kernels. These are isolated
chain measurements, not endpoint ms/round. Each candidate passes an FP32
attention oracle on exactly decoded FP16 K/V, two changed-query graph replays,
invalid-request and negative-position masks, and an isolated query-scale-eight
case. Maximum absolute initial errors are 3.052e-5 for M5 and 6.104e-5 for
M20. The protected reader's FP16 probability boundary produces larger oracle
error in these samples. Model teacher-forcing, acceptance and same-wheel C1/C4
measurements remain required before promotion.

The clean wheel passes nine GPU cases, including all E4M3 encodings, rewritten
history and aliased pages. Its 16 previous native modules retain their exact
hashes. On a full TP4 NV2 V100 mesh, the capability-disabled model control
measures C1 19.809 ms/round at I8192/O256 and C4 43.113 ms/round at I128/O600.
The C1 probe emits 4.886 tokens/round. Sixty-four repeated, aligned
teacher-forcing positions have identical logits and top-1 decisions within
that engine. The capability-enabled initial experiment measures C1 19.103
and C4 42.244 ms/round, saving 0.706 and 0.869 ms respectively. C1 emits the
same 256 token IDs and 4.886 tokens/round.

The initial experiment fails the model qualification gate. Across 64 aligned
teacher positions, mean KL is 0.000859, maximum KL 0.011318 and top-1 agrees
at 63 positions. Repeated candidate captures are bit-exact at all 64 positions.
Eight 600-token prompt clusters have mean acceptance 43.445% versus 44.990%
for the control. The paired difference is -1.544 percentage points, with a
95% prompt-bootstrap interval [-4.348, +0.675]. An interval spanning zero
does not establish unchanged acceptance. Natural outputs diverge after 3–164
tokens, and none of the four C4 streams is identical. These speed results are
not admitted as a qualified improvement.

The next revision loads device history inside the existing attention kernel,
without protected page resolution or miss staging. It uses the same compact
page4 count, split assignment, online softmax and FP16 PV boundary. New
isolated tests compare directly with the protected reader in addition to the
FP32 oracle, including long and short contexts, E4M3 and FP16 histories and
changed-input graph replay. The arithmetic-preserving revision passes 21
normally installed-wheel GPU cases, including strided queries and gates.
Direct/protected outputs match bit-exactly. A matched, synthetic-selector
chain with 12 target and four draft calls measures M5 1077.568→885.920 us and
M20 1527.008→1380.992 us, with 64→32 kernels. All 16 calls and changed-query
replays match the control exactly; both retain the same FP32-oracle error.
These 0.192/0.146 ms reductions are below the structural optimization budget,
so this revision has no new model timing or acceptance claim. Model tests of
the independent packed PLE result change keep direct QSA disabled.

A probability-decomposition variant replaces scalar PV with three tensor-core
products and an FP32 residual. Independent power-of-two scaling preserves
probability bits, including values not representable by a normal half. CPU
reconstruction is exact on one million random values and exponent boundaries;
isolated GPU numerical checks pass. It is slower: M5 target chain
825.952→686.336 us versus about 489 us for scalar PV, and M20
1286.000→1723.952 us. It is rejected and has no production dispatch.

## Sources and applicability

[FlashInfer split-KV](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
shows how KV splits fill otherwise idle SMs for small query batches; the
native path retains enough splits for the 80-SM V100 without host scheduling.
[HiSparse](https://arxiv.org/html/2608.07009v1) emphasizes that resolution and
placement costs lie directly before attention. Here the authoritative history
is already device-resident, so resolution can be removed entirely; the host
placement experiment remains separate.
[BitDecoding](https://arxiv.org/html/2503.18773v1) separates quantized layouts,
cooperative decoding and tensor-core computation. Its tested SM80/89/90
mechanisms are not copied onto SM70: this implementation uses ordinary loads,
shared memory and mma884, with no TMA, cp.async or WGMMA.
