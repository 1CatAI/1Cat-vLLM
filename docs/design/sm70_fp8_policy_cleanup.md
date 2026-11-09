# SM70 FP8 policy cleanup

The FP8 environment defaults already select native TurboMind and disable the
expert dequantization fallback. Two startup environment writes no longer make
routing decisions:

- The TP≤2 / FP8-KV rule writes `VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK=0`
  only when that variable is unset. Its getter already returns `False` in that
  case. Removing the write preserves the effective value for every TP/KV pair.
- The rule that writes `VLLM_SM70_FP8_TURBOMIND=0` requires the variable to be
  unset, TurboMind to be disabled, and Marlin not to be forced. Its unset getter
  returns `True`: the auto and TurboMind backends enable it; the remaining
  Marlin backend fails the last condition. The rule is unreachable.

These writes are removed rather than replaced with another configuration
field. User overrides retain their existing parsing and precedence. No kernel,
quality qualification, tuning parameter, or precision changes in this step.
Metadata and the generated reference now describe the actual getter defaults
without advertising automatic rules that cannot change them.

Run the independent policy comparison with:

```console
python -m tools.sm70_route_snapshot --category fp8-policy --baseline-ref <previous-main-sha>
```

The command executes the original FP8 policy statements extracted from the
specified git revision. It compares the actual `Fp8LinearMethod` route flags
with the current policy, without loading weights or executing native kernels.
It retains the standard 324 configurations, including non-FP8 applicability,
and adds 216 FP8 dense/MoE configurations plus 54 explicit override combinations.
This checks the removed policy decisions; it does not claim coverage of every
QPN8 subkernel. The NVFP4 category snapshot remains a separate check.

At integration base `f350e2ebe7`, AST-counted `os.environ` write sites in
`config/vllm.py` decrease from 21 to 19. Environment names and registrations
are unchanged; unregistered Python reads remain zero. The removed TP condition
was redundant policy, not evidence for admitting a previously unqualified
model. FP8 kernel integration and admission validation continue separately.
