# SM70 mHC projection and output staging

The FP16 mHC provider computes FP32 projection and squared-norm partials
before the existing mixing, Sinkhorn and normalization stage. The standalone
pre path, broadcast pre path and large-token post/pre path share
`sm70_mhc_prenorm_staging` in
[`kernels/mhc/triton.py`](../../vllm/model_executor/kernels/mhc/triton.py).
Their selection and split counts remain in
[`kernels/mhc/tilelang.py`](../../vllm/model_executor/kernels/mhc/tilelang.py).

## Execution and numerical contract

For K partials of at most 4096 elements, one Triton launch replaces the
separate projection and squared-norm launches.
Each CTA still computes one token, projection column and K split, with eight
warps. Column zero also writes that split's squared norm using the input it
already loaded. A zero-column projection still computes the squared norm.

Larger K partials retain the separate norm launch: keeping the input live for
both reductions increases register pressure and can make those shapes slower.
The broadcast path uses partials of 2048/4096 elements; the standalone path
uses 2048/4096 through 128 tokens and 8192 above that boundary. The existing
split choice is preserved. This is a local launch optimization, not a new
backend or model qualification.

The input is FP16 and the weights and partials are FP32. Products, reduction
trees, split boundaries and the broadcast norm multiplier retain their former
order. This preserves the partials bitwise; it does not substitute a different
GEMV reduction or change the subsequent Sinkhorn/normalization implementation.
No policy, environment variable, custom-op schema or native ABI is added.

The existing callers own both output buffers. The fused stage allocates no
temporary tensor and keeps no global workspace or cached buffer. Capture
records the caller's addresses; replay consumes updated input/weight contents
and overwrites both partial buffers. Independent callers retain independent
buffers.

## Output mixing

The existing `sm70_mhc_post` stage uses 256-element hidden-dimension tiles
for up to eight tokens, increasing independent CTAs for small batches.
Larger batches retain 512-element tiles. The FP32 operation order, FP16
saturation at 65504, explicit residual/mix/output strides and optional
caller-owned output remain unchanged. Input `x` retains its unit hidden
stride requirement. This reuses the existing kernel and adds no policy or
workspace.

Tile changes require bitwise validation: a smaller tile can change compiler
contraction even when the scalar expressions are identical. The regression
checks compare against the former 512-element launch, including extreme
values, padding, strided residual/output and independent graph replay.

## Validation

Run on SM70 with the usual source runtime:

```console
python -m pytest --confcutdir=tests/kernels -q tests/kernels/test_mhc_sm70_prenorm_staging.py
python -m pytest --confcutdir=tests/kernels -q tests/kernels/test_mhc_sm70_post_schedule.py
```

The regression oracle retains the former independent reductions because a
generic matrix multiplication uses a different FP32 reduction tree. Cases
cover the current standalone and broadcast split choices, the 16/17 and
128/129 token boundaries, empty input, padded strides, cancellation, FP16
extremes/subnormals, and alternating replay through independent graph buffers.
These are operator checks; they do not establish model throughput or validate
unrelated mHC implementations.
