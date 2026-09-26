# SM70 DFlash2 FP8 verification below 20 ms

## Scope

- Date: 2026-08-28.
- Integration base: `62ad1e02693f4c857f3b7547cef1860ee54e8053`.
- Target: official Qwen3.8-27B-FP8 target plus official DFlash2 draft, TP4
  V100, batch one, seven draft tokens, E5M2 target KV, FP16 draft KV, and
  target/draft FULL CUDA Graphs.
- Performance gate: mean complete speculative round strictly below 20.000 ms
  at 1K context under the production 256K/4096/prefix-cache/Mamba-align
  engine contract.
- Quality gate: no task-score regression, no structured-output or tool-call
  regression, and request-mean DFlash2 acceptance may not fall by more than
  0.05 from the same-source control.

## Frozen starting evidence

The current-main trace uses the official FP8 checkpoint revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` and records 172 steady rank
cycles:

| Phase | Mean wall |
| --- | ---: |
| Draft | 3.832957 ms |
| Draft to target | 0.167954 ms |
| Target verify | 16.705168 ms |
| Target to draft | 1.192924 ms |
| Complete round | **21.899004 ms** |

The target graph contains 15.93275 ms of kernel service. Block-FP8 QPN8
projections account for 9.549 ms: 5.327 ms in ordinary projections and
4.222 ms in fused gate/up. The first optimization order is therefore exact
M=8 QPN8 kernel work, followed by TP4 reduction and recurrent-state residue;
sampling shortcuts or reduced verification are outside this campaign.

Raw trace and phase artifacts are retained under
`/data/minimax-h3/task-cache/v100-dflash2-fp8-verify-20260828/`.

## Experiment discipline

Each CUDA candidate is first built as an isolated operator-race library and
must beat the production exact shapes with real checkpoint weights in both
warm- and cold-cache CUDA Graph trials. Numerical comparison uses the
dequantized FP32 reference and graph replay stability. A candidate that does
not project enough complete-round saving is removed before a full vLLM build.
Only a same-source end-to-end trace can satisfy the 20 ms gate.
