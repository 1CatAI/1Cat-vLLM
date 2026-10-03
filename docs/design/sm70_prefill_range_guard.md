# SM70 long-prefill range recovery

The compact D256 grouped-attention path keeps its FP32 QK/PV accumulation,
existing shape admission and FP16 input/output types. Two device-side checks
repair ranges that cannot safely pass through its FP16 intermediates.

The prefix consumer detects a missed block maximum or clipped partial before
online merging. Affected partials are rescanned and recomputed. The score
storage guard marks 64-query tiles whose logits exceed the compact range and
recomputes them from original Q/K with FP32 scores. Recovery retains the
centered/scaled V representation and restores outputs after FP32 normalization.
This changes results in exceptional ranges rather than saturating outputs.
The established score margin, V headroom and cuBLAS math mode are retained.
Tail scans retain the sampled maximum when its margin bounds every score;
only peaks outside that existing safety bound replace the shift.

Flags are cleared on every call and CUDA Graph replay. Completed prefix
workspace is reused for recovery. No model, tensor-parallel or concurrency
whitelist, environment variable or dispatch-policy change is introduced.

Validation covers overflow, close large logits, missed peaks, large constant
values, short suffix-causal chunks, prefix/tail boundaries and changed graph
inputs. GPU results and paired timings are recorded in the integration PR;
results for earlier revisions are not a new full-model quality evaluation.
