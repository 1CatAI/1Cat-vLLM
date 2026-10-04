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
candidate. GPU validation is pending.
