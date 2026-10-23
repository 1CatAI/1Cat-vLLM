# GGUF projection clock diagnostics

CUDA graph node tracing inflates short SM70 projection service times. A separate
`gguf_dmv_sm70_clocked_out` operator records `%globaltimer` at the beginning and
end of each warp into caller-owned `[blocks, warps, 2]` int64 storage. The normal
operator selects a separate template specialization without timer stores. The
diagnostic performs no synchronization or event insertion between graph nodes.

The timestamps describe the on-SM warp envelope, excluding dispatch and retirement.
They are local to each device and must not be compared across TP ranks. Timer
instrumentation can change register allocation and scheduling; its result is a
diagnostic, not an unprofiled production latency or speedup.

## Calibration

V100-SXM2-32GB, SM70, CUDA 12.8, Torch 2.10, 1290 MHz SM / 877 MHz memory;
M=8, loaded TP4 rank0 projection planes, FP16 operands and FP32 accumulation.
Cold-bank graph replay compares ordinary and timer specializations on the same
GPU. These initial prototype measurements use a private extension only for
calibration and do not establish installed-wheel or model performance.

| Workload | Ordinary | Clocked | Output |
| --- | ---: | ---: | --- |
| IQ3_S gate/up, N=4352 K=5120 | 38.652 us | 37.807 us | identical FP16 bits |
| 16-layer GDN serving-operator chain | 0.999600 ms | 0.996441 ms | identical FP16 bits |

The second workload uses real projection planes and synthetic recurrent states
and small non-projection weights. It is not a full model. The gate/up specialization
uses 86 registers without timing and 94 with timing: the approximately 2% change
must remain explicit when interpreting model measurements.

CUDA graph event-node insertion was rejected: the same chain increased from
1.000 to 1.586 ms. Graph-node Nsight tracing also increased a control chain from
0.997 to 1.295 ms. Those measurements explain why isolated-versus-traced
projection differences cannot be promised as recoverable end-to-end latency.

## Verification

The focused GPU tests compare raw FP16 output bits at three activation amplitudes,
repeat CUDA graph replay, check split-K counter reset and timestamp coverage, and
exercise two/three-format outputs and fused gate/up. Source-complete wheel tests
and model diagnostics remain pending until the ordinary artifact is rebuilt.

## Installed artifact check

The source-complete wheel built from `aa275f045988bd3977c2ba3c5ba275269ae843dc`
passes all 156 focused GPU/codec/capability checks, including the ten diagnostic
cases. All 16 packaged native library hashes match the installed artifact.
The installed-wheel cold-bank gate/up replay measures 38.805 us for the ordinary
operator and 37.917 us for the diagnostic, with identical output bits.

Run the focused check with:

```bash
pytest -q tests/kernels/quantization/test_gguf_dmv_clock.py
```

## Model diagnostic

The installed artifact completes TP4 graph initialization, a 1K-input/96-output
request and two natural EOS checks. Its 96 output-token prefix matches the
previous ordinary boundary-event diagnostic with the same prompt and seed.
These short requests do not replace the matched sixteen-prompt latency baseline.

For the final target replay, rank0 records 237 resident-plane calls: 62 gate/up,
63 down, 39 GDN input, 48 GDN output, nine attention QKV and sixteen attention
output projections. Their warp envelopes sum to 6.781 ms over a 12.859 ms
first-to-last projection span. Other ranks measure 6.780–6.802 ms. The remaining
6.078 ms between projections includes attention, recurrent updates, collectives,
canonical/native projections outside this diagnostic and launch dependencies.
It is not empty time or a promised recoverable saving. Device durations are
quantized to 1.024 us on this GPU; per-warp envelopes also exclude retirement.

Within the instrumented last replay, the 39 GDN-input-to-output intervals average
31.823 us, the nine measured attention-QKV-to-output intervals 96.142 us, and
46 GDN-output-to-gate/up intervals 18.477 us. Cold isolated timing and traced
service remain separate cohorts. No end-to-end speedup is claimed.
