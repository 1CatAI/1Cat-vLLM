# SM70 finite-score recovery proposal

The compact D256 grouped-attention path keeps its FP32 QK/PV accumulation,
existing shape admission and FP16 input/output types. Two device-side checks
repair ranges that cannot safely pass through its FP16 intermediates.

The prefix clipped-peak repair and the overflowing-score repair are already
implemented separately. This proposal extends recovery to finite logits
outside the compact precision range and to unsampled tail peaks. The score
storage guard marks 64-query tiles whose logits exceed magnitude 128 and
recomputes them from original Q/K with FP32 scores. Recovery retains the
centered/scaled V representation and restores outputs after FP32 normalization.
This changes results in exceptional ranges rather than saturating outputs.
The established score margin, V headroom and cuBLAS math mode are retained.
Tail scans retain the sampled maximum when its margin bounds every score;
only peaks outside that existing safety bound replace the shift.

Flags are cleared on every call and CUDA Graph replay. Completed prefix
workspace is reused for recovery. No model, tensor-parallel or concurrency
whitelist, environment variable or dispatch-policy change is introduced.

This remains a proposal: the broader recovery passed native stability checks
but regressed the 32K release-profile speed screen and is not enabled in main.
Validation covers overflow, close large logits, missed peaks, large constant
values, short suffix-causal chunks, prefix/tail boundaries and changed graph
inputs. GPU results and paired timings are recorded in the integration PR;
results for earlier revisions are not a new full-model quality evaluation.
