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
