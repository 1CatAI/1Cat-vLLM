# DFlash split windows with E4M3 hybrid caches

An SM70 hybrid target using E4M3 KV chooses 1648-token pages. Attention cache
specifications refresh the shared block size after loading, so FP16 draft
attention can also receive 1648-token pages. The split-window implementation
already indexes the live page size and K/V strides, but the public adapter only
admits 832/1024/2048. This sends the draft back to general paged attention when
the target changes cache dtype.

Admit 1648 pages for the existing single-request Q8/H8/KV2/D128 FP16 split
window. The per-engine draft-window policy disables the new hybrid page routes.
The native concurrent implementation retains its 1024/2048 page ABI. Unsupported
query counts, dtypes, layouts or unavailable split imports retain the fallback.
There is no new kernel, weight conversion, approximation or environment variable.
Probability, PV and partial reduction remain FP32.

Focused tests extend the FP64 window oracle, page indirection, zero lengths,
live graph lengths, public dispatch without the native symbol and concurrent
fallback. The installed source-complete CUDA 12.8 / Torch 2.10+cu128 wheel passes 37 focused
GPU/graph/dispatch checks. Its sixteen native libraries are byte-identical to the
qualified artifact. Cold graph ABBA on V100-SXM2-32GB at 1290/877 MHz measures
117.235 to 56.440 us at 1091 tokens and 208.073 to 61.038 us at 8192, with sixteen
interleaved KV banks and external events. Both policies run in one normal wheel.
The FP64 window oracle, repeated graphs and zero lengths pass. Artifact and numerical details are in
[data/sm70_dflash_page1648_20261007.json](data/sm70_dflash_page1648_20261007.json).

The matched TP4 comparison uses the same sixteen prompts (eight each at 1K and
8K), 600 output tokens, 262144 maximum length, max sequences 4, temperature 0.7,
top-p 0.9, top-k 20 and seed 123. Target KV is E4M3; draft KV is FP16 and SSM state
FP32. Both normal wheels retain identical native libraries. Timing excludes
prefill and the first 20 rounds and uses complete engine-output intervals, with
no device timers or profiler during generation. All timed clock samples remain
1290/877 MHz on all four ranks.

| Input | Before admission,ms/round | After admission,ms/round | Saving,ms | Output tokens/round | ms/output token |
| --- | ---: | ---: | ---: | ---: | ---: |
|1K|16.3051|15.8691|0.4359|2.9286|5.4253|
|8K|17.0675|16.2916|0.7759|2.9682|5.5044|

Quality compares only equal input prefixes and positions: 126 of 128 rows remain,
mean KL 4.13e-5, max 6.03e-4 and top-1 agreement 100%. Both natural prompts stop
normally with reasonable outputs; all four concurrent smoke requests complete.
The C4 smoke is a health check, without a matched C4 performance claim. Paired
prompt saving intervals are 0.420–0.452ms and 0.750–0.802ms; these do not quantify
temporal run-to-run variability. This measures draft attention admission; it is
not a claim that E4M3 alone accounts for the full saving.
