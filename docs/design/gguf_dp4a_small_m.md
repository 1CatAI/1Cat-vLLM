# Shared GGUF dp4a path for small batches

For M at most20, quantize each activation row once to standard Q8_1 blocks:
32 signed int8 values, FP16 scale and FP16 sum. Integer-codebook readers feed
SM70 dp4a directly, with original weight and activation scales applied after
the integer dot. FP32 accumulation and the final FP16 projection boundary
remain unchanged. Activations incur the explicitly authorized int8 rounding.

The common header under TurboMind is intended for both dense and routed GGUF
projections. The first reader is IQ3_S. A routed gate/up launch reads routing
IDs directly and forms SiLU times up without activation gathering. A debug
mode exposes the separate projection boundary for the official-weight/Q8_1
oracle. It is not selected by model dispatch until calibrated and qualified.

IQ3_XXS and IQ2_S, dense formats and fused down reduction follow only after
the IQ3_S M5 expert benchmark reaches450GB/s. Every bandwidth report must give
actual weight storage and distinguish unique active-expert bytes from repeated
route reads. Gate/up-only and activation-quantization-inclusive timings are
reported separately, with kernel counts. Whole-model gains require separate
unprofiled C1/C4 measurements.

References: [llama.cpp integer dot](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh),
[Q8_1 quantization](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/quantize.cu),
[Strix rows/signs/prefetch](https://github.com/halo-box/strix-llama.cpp/pull/106),
[shared-expert fusion](https://github.com/ggml-org/llama.cpp/pull/29184).
The original MIT licenses are packaged with the GGUF implementation. Strix
wave tuning targets AMD and needs independent Volta measurements.

## First IQ3_S measurements

The initial whole-wheel implementation uses layer17's original IQ3_S rows,
512 experts, TP4 local N160 and K2560. Random top10 routing touches49 unique
experts at M5 and168 at M20. Activation inputs are synthetic FP16 normal
samples with standard deviation0.125. Measurements use CUDA Graph replay,
eight alternating candidate/control epochs and100 replays per epoch.
The native control is the current original-block FP32 grouped vector dot;
its activation gather and separate SiLU are prepared outside the timed region.
The dp4a candidate includes direct routing and the FP16 SiLU/multiply boundary.

| M | Native gate/up | dp4a fused | Q8_1 plus fused | Unique-source bandwidth |
|---|---:|---:|---:|---:|
|5|73.82us|42.85us|44.06us|403.94GB/s|
|20|210.37us|120.90us|121.99us|490.90GB/s|

M5 runs at1290MHz core and877MHz memory. M20 boosts to1530MHz in later
epochs, so these rows are not a controlled clock comparison. The M5 result
does **not** meet the450GB/s expansion gate. Its49 active experts contain
17,310,720 bytes: the gate requires at most38.47us for this routing sample.
One M5 combined epoch reached241us; it remains in the raw samples.
Standalone quantization replay is host-starved and is not a reliable measure
of its GPU cost; combined minus fused is approximately1.2us.

The actual NVFP4 TP4 weights, prepared by the production converter, measure
41.25us for plan plus W13 at M5. This is a separately quantized weight
representation, not an output oracle. That comparator supports at most160
routes and is explicitly omitted at M20 (200 routes).

Nine focused GPU tests pass, including changed-input graph replay. On real
IQ3_S rows the relative L2 error against official GGUF dequantization with
the same Q8_1 activations is0.00020–0.00021; against the FP16-activation
control it is0.0053–0.0054. These are projection measurements, not model
KL, top1, acceptance or end-to-end latency evidence. No model dispatch is
enabled by this initial result.

## Two-row layout

Nsight Compute on the first IQ3_S M5 kernel confirms37.7% long-scoreboard
stalls,56 registers per thread, and762,602 shared-load bank conflicts.
DRAM bytes read are17.41MB, consistent with the unique-source estimate;
profiled kernel duration is not substituted for unprofiled timing.
Reducing registers to40 introduces stack spills and regresses M5 to68.54us
and M20 to194.75us. That variant is removed.

Using16 lanes per row instead gives two rows per warp and keeps all lanes
busy for K2560's80 activation groups. It retains56 registers without spills.
Sixteen calls per graph replay remove host starvation from the short encoder.
The new input standard deviation is1.0, reported with the benchmark.

| M | Native gate/up | dp4a fused | Q8_1 plus fused | Unique-source bandwidth |
|---|---:|---:|---:|---:|
|5|63.61us|34.28us|35.71us|505.00GB/s|
|20|213.83us|111.88us|113.55us|530.47GB/s|

M5 reaches1530MHz after the warm-up epochs; its later candidate epochs are
34.19–34.35us. The same M5 NVFP4 plan/W13 control is39.17us. M20's heavier
workload runs at1470–1485MHz. This passes the IQ3_S expansion threshold in
the sustained M5 clock band; it does not assert450GB/s at1290MHz.
The nine focused tests and real-weight oracle checks still pass. The operator
remains separate from model dispatch pending format, down-reduction and
model-quality checks.

## Additional expert readers and down reduction

The common integer reader now covers IQ3_XXS and IQ2_S as well. IQ2_S
applies its two group16 scales to separate integer partial sums; IQ3_XXS
restores the eighth sign from parity. Both routing index widths are accepted,
so the native Int32 router needs no conversion kernel. The47 focused GPU
tests cover these readers and IQ4_NL/Q2_0 down with TP4 K160 slices, changed
inputs, routes and probabilities under CUDA Graph replay.

| Format | M | Current gate/up | dp4a fused | Encode plus fused |
|---|---:|---:|---:|---:|
|IQ3_XXS|5|62.24us|30.54us|31.94us|
|IQ3_XXS|20|209.41us|96.27us|97.90us|
|IQ2_S|5|74.56us|35.54us|36.65us|
|IQ2_S|20|212.44us|114.21us|116.47us|

The IQ2_S M20 control is the production canonical grouped GEMM (two
launches), rather than the slower original-block vector candidate. M5
late epochs run at1530MHz. IQ2_S unique-source bandwidth is364GB/s at M5
and388GB/s at M20; despite that lower bandwidth it wins against its current
controls. IQ3_XXS M5 reaches505GB/s. The encoder plus fused gate/up has two
launches for all three formats, with gathering and separate SiLU removed.

Down reads existing N32/K8 canonical integer storage. This preserves the
exact Q2_0 integers and original scales across TP4's160-wide slices, which
cut the original64-wide blocks. IQ4_NL uses TurboMind's integer LUT. Routed
intermediates are encoded in shared memory; a single launch performs down
and the FP32 weighted route reduction, with the FP16 down boundary retained.

| Down format | M | Current down plus unroute | Fused integer down/unroute |
|---|---:|---:|---:|
|IQ4_NL|5|49.30us|33.15us|
|IQ4_NL|20|94.01us|79.60us|
|Q2_0|5|39.64us|23.62us|
|Q2_0|20|88.46us|62.89us|

IQ4_NL M5 spans1290/1530MHz; use per-epoch matched timings for clock-specific
comparisons. These down controls include unroute but exclude preparation of
sorted intermediate rows. Real-weight relative L2 against official weights
with identical Q8_1 activations is approximately1e-5; FP16-activation error
is approximately0.005. The full expert candidate has three launches: encode
the input once, direct-route fused gate/up, and down/unroute. Its M5 gate/up
savings, extrapolated over10 IQ3_S,17 IQ3_XXS and20 IQ2_S layers, are1.55ms;
this is an operator estimate and requires an independent whole-model result.

Model selection is declared by capabilities for original M5/M20 and exact
TP4 expert geometry. `kernel_config.sm70_gguf.small_m_dp4a` controls the
Q8_1 activation route; other batches retain the existing canonical route.
Missing operators, unsupported formats or shapes, unavailable original rows,
disabled policy and incompatible dtypes are reported as fallback reasons.
Model KL, top1, natural completion and acceptance intervals remain pending.
