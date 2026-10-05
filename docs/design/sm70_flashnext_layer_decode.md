# Flash-Next whole-layer decode fusion screen

This campaign targets TP4 SM70, single-token decode and the M5 MTP4 verifier.
The acceptance targets are at most 600 kernels per step, no single-CTA kernels,
and ordinary pure decode at most 7.5 ms/token before the final 6 ms target.
These targets have not been achieved by this screen.

## Establishing a comparable endpoint

The integration base is `615710ae5106beeeab87950599fff28977e92962`.
The historical approximately 97.7 tok/s source reproduction at `b2042fb24b78`
used 8192 input tokens, 513 forced output tokens, FP16 KV, native FP32 recurrent
state and pinned-UVA PLE decode. The recent endpoint uses disk mmap PLE.
PR #831 records a source-complete mapped-result disk baseline of 10.98609 ms.
PR #885 requalification at `fd65333826e9` records 11.95157 ms for its reference
reader and 11.87902 ms for the default native reader. Earlier qualification of
that reader recorded 11.01475/10.86912 ms. The later pair is retained separately;
the earlier pair does not establish current-main speed.

The normal `fd65333826e9` package and this integration base have identical CUDA
sources and NVFP4 model implementation. Their only source differences are four
GGUF Python files. Its native artifact is therefore a source-matched NVFP4
control. A fresh normal native build from this owned tree is in progress.
Historical pinned-UVA timing cannot by itself identify a regression commit in
the disk route; same-contract comparisons must locate the later disk slowdown.

The maintained regression entry records source, native hashes, resolved worker
routes, full timing token IDs and natural health outputs. It fixes FP16 dense
and KV, FP32 recurrent state, TP4, 262144 startup capacity, 94% memory budget,
8192 input / 513 output, one request, CUDA graphs, no prefix cache and no MTP.
Pure decode is separate from first-token and prefill time. EOS suppression is
confined to the fixed-length timing requests.

```bash
.venv/bin/python -m benchmarks.benchmark_sm70_flashnext_regression \
  --model /path/to/model --source-sha SOURCE_SHA --output baseline.json
```

## Research implementations

No experimental runtime dispatch or new environment switch is installed.
The screen DSOs are research artifacts; they are not source-complete production
performance evidence. An admitted implementation must move into the ordinary
package build and pass a clean-artifact model gate.

| Segment | Screen | Synchronization and reduction |
| --- | --- | --- |
| HC | combine/norm/down/push plus up/mix, two kernels | Per-row tagged producer packets; no grid barrier; fixed FP32 sums |
| Router and experts | One router CTA inside W13, intermediate-tile W13/W2, one kernel | Cooperative resident launch without a grid barrier; independent producer flags and ordered sums |
| Shared expert | gate/up, SiLU, W2 and shared gate, one kernel | Ten resident intermediate-tile CTAs, release/acquire flags, ordered partial sums |
| GDN | Full value-head recurrence and gated RMSNorm, one kernel | One CTA owns every value column and all sequential M5 states |
| QSA | Selection, index expansion, sparse attention, merge and output gate, one kernel | Resident selector/core/merge CTAs and release/acquire flags |

HC covers real checkpoint weight banks, both M1 and M5, and changing width
allocations. Its M1 control uses the current TP4 operator. Its M5 control uses
the existing replicated FP32 MMA verifier implementation. An additional HC
screen incorporates the preceding core all-reduce into down. Shared expert
uses real checkpoint weights and includes
the small gate; its following all-reduce is excluded. GDN includes the complete
post-convolution core and output norm; projections, convolution and production
state-slot metadata are excluded. M1 compares FlashQLA plus norm; M5 compares
the actual mixed-QKV verifier core plus norm. Independent FP32 arithmetic checks
all M5 recurrent states. QSA excludes score construction, projection, rotary
and KV writes; it admits only the screened 8K geometry. Router/expert compute
excludes shared experts and the following TP reduction. These exclusions are
recorded in each result.

Prebuild on CPU while waiting for a complete GPU lease:

```bash
CUDA_VISIBLE_DEVICES= TORCH_CUDA_ARCH_LIST=7.0 \
  .venv/bin/python -m benchmarks.kernels.benchmark_sm70_hc_norm_push_screen \
  --build-only --output hc-build.json
```

Each screen records graph measurements and retained CUDA profiler node counts,
including grid geometry when the profiler exports it. Missing geometry is not
reported as zero single-CTA kernels. Keep kernels with insufficient whole
segment benefit out of model execution. A microbenchmark delta cannot be
subtracted from endpoint TPOT or credited as an end-to-end result.

## Quality and promotion

For this campaign, record teacher-forcing KL mean/p99/maximum over matching
prefixes and the full valid vocabulary, top-1 agreement and natural termination.
Maximum absolute logit difference is diagnostic and does not veto admission.
FP32 association differences do not require bitwise investigation. Moderate
precision changes require the same distribution and output-health gate.
This campaign contract supersedes the historical raw-logit maximum veto for
these changes; old evidence and thresholds remain historical records.

Compilation and Python syntax/lint checks have passed. GPU measurements,
matched endpoint qualification and model distribution gates are pending. No
speed improvement, kernel-count acceptance or default production admission is
claimed yet.
