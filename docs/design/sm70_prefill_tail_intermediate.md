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
