# Retire the DFlash2 candidate-order experiment

The retained [quality audit](sm70_dflash2_quality_audit.md) records a paired
D2/D3 screen: dense-order selector QPN8 scored 14/24; candidate-order scored
13/24, with one lost pass and no new pass for a 2.48% mean-decode gain. That
small screen does not establish a population regression, but its unfavorable
quality/risk trade did not authorize production use. The earlier real-hidden
boundary evidence supports retaining the dense vocabulary tie-order contract.

Remove `VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER` and its typed experiment
field. Dense vocabulary tie ordering is mandatory; its typed option is also
removed. The obsolete `VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER` name remains registered
for one full released compatibility cycle, warns when read during policy
resolution, and has no replacement switch. An old value of 0 retains the
validated dense-order behavior. Native numerical code and the scored selector
QPN8/reranking acceleration remain unchanged.

This intentionally changes only stale benchmark configurations that explicitly
combined candidate-order opt-in with dense ordering disabled. The standard
324-configuration matrix and the other parser/override edges preserve their
routes. The snapshot command executes the historical policy class, getters,
GDN admission, and tie-order helper rather than synthesizing an old decision:

```console
python -m tools.sm70_route_snapshot --category dflash2 \
  --baseline-ref <previous-main-sha> \
  --expected-changes tests/config/data/sm70_dflash2_retired_order_changes.json
```

The numerical function AST comparison changes only the routing helper that
now enforces dense ordering. Existing paired quality evidence determines the
retirement; no new full-model benchmark is needed to remove an unsuccessful
research override. This does not establish a 35B AWQ/FP8 speed baseline.

At the #807 integration base, registrations decrease 999→998, two typed
experimental choices disappear, public SM70 controls remain 31, checked
unregistered reads remain zero, and startup environment-write sites remain 19.
