# PLE split-placement integration audit (2026-09-07)

## Scope

Integrate PR #528, based on public main
`53199eb8b996831718330c23144c60ed9952f690`. Preserve automatic FP8 PLE
placement between VRAM and pinned host storage on pre-Ampere workers,
explicit placement overrides and stable UVA pointers. The memory estimate
is a placement heuristic, not a guarantee that any requested context fits;
hybrid allocator padding and the actual activation peak still matter.

## Repairs

- Separate physical vocabulary padding from the logical checkpoint TP range.
  A CPU regression reproduced the original all-device placement rejecting
  a valid padded table. Both halves now copy only their logical overlap,
  while insufficient combined capacity remains an error.
- Dummy loading sees a zero-row placeholder, not the private backing tables.
  Initialize those tables to finite zero bytes before their first post-load
  preparation when no checkpoint shard was loaded. Normal checkpoint copies
  and all decode arithmetic remain unchanged.
- Reject negative or non-finite explicit host/device budgets and reserves.
- Publish table state only after both allocations succeed, permitting a
  clean retry after a pinned-host allocation failure.

## Focused validation

Python 3.12, Torch 2.10.0+cu128, NVIDIA V100-SXM2-32GB (SM70), one reserved
physical GPU 3. Tests use tiny synthetic FP8 tables, no checkpoint or model
server, and task-owned fresh Triton caches. The existing native extension
is used for import support; no new C++ execution is claimed by this change.

```bash
.venv/bin/python -m pytest -q tests/models/qwen4_exp/test_ple.py
```

The first focused run passed 59 cases, including all-device, mixed and
all-host gathers, scale decoding, four changing-ID graph replays per split,
in-place checkpoint reload, stable pointers, poisoned dummy storage and
budget validation. The allocation-failure retry test was added afterward;
the final test result and CI head are recorded on the integration PR.

No new full-model throughput or maximum-context claim is made. The original
PR's heterogeneous-device capacity evidence is retained separately; it must
not be interpreted as a fresh same-criterion V100 speed measurement.
