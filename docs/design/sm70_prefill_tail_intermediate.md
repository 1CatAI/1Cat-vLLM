# SM70 prefill tail intermediate range

The Q8192 prefill route can produce nonfinite attention outputs from finite
FP16 Q/K/V. A captured 27B long-context input reproduces 34 positive infinities
in one GQA group. The FP32 reference values for these elements are between
1.636 and 1.660, so the final output range is not the cause.

This scope checks the tail probability/value numerator before normalization.
The correction must preserve FP16 operands and FP32 tensor-core accumulation,
protect the intermediate range, and retain finite normal-range behavior.

The remaining sampled-peak proposal in #844 overlaps tail shift selection;
its finite-score recovery threshold is a separate change. Compare the captured
input against that proposal before expanding the output workspace.

Validation uses the independent captured-input reproducer, FP32 attention,
Q8000/Q8192 stability checks, and matched model prefill and long-context quality.
Results will be recorded after the normal FA2 extension and wheel are built.

Integration base: `f551e0afea0736a636bf9e288da79c53aed56f6d`.

The 34 affected rows have complete tail maxima 10.53–10.77, but stride-eight
samples near -6.3. The gap is 16.77–17.03. Sampling therefore selects a shift
near -2.3; exponent clipping and the FP16 numerator are unsafe even though the
normalized result is small. Use the sampled shift only when its existing
four-unit margin bounds the complete maximum; otherwise use the complete
maximum plus the same margin. This is the tail condition proposed in #844
(commit `8e4d24e4b4c`), isolated from its finite-score recovery threshold change.
No workspace expansion or accumulation precision change is needed for this
candidate. The installed operator checks are recorded below.

## Operator validation

CUDA 12.8, Torch 2.10.0+cu128, V100 SM70, normal FA2 CMake target and whole
wheel, with no private binary or Python-path override:

- The new correlated-value regression fails on the preceding installed wheel
  for both Q8000 and Q8192, producing nonfinite output.
- All 11 stability, rejection, shared-workspace and graph-replay checks pass
  on the corrected installed wheel.
- All 16 captured GQA groups from four ranks and four model layers produce
  finite output. FP32 reference checks include every formerly nonfinite query.
  The formerly failing group has relative L2 error 0.000450 and maximum
  absolute error 0.00351 over the selected query rows.

The core extension is unchanged. The corrected normal FA2 fingerprint is
`485c7a25f03ef15304f18735794edfa1814831f6e69cfc5a5df9b096483c1001`;
the whole-wheel fingerprint is
`c1c3a5fc063ec70280761a80e4a5f43068ec81b7c32637c7c1edbd306c108411`.
Its dependencies resolve to the declared Torch/CUDA/cuBLAS libraries.
Model concurrency, 8K/32K prefill and the four frozen needle cases are running.
