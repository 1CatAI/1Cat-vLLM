# Flash-V100 baseline limitations (not fixed by A3 refactors)

Baseline: #1060 `8c96e32e56c09d4a3e3112cb5d1a367571f69476`.

- DDTree's Triton branch correction rejects uint8 E4M3 cache storage. In
  eager execution the existing verifier catches that error and falls back to
  dense masked attention; capture rethrows the unsupported-dtype error.
  The synthetic CPU matrix records the existing rejection. This is not GPU
  numerical evidence or a newly supported configuration.
- Prefix-anchored SWA accepts FP16 KV only and rejects DDTree drafting
  metadata. Invalid Cartesian-product combinations are explicit rejection
  cases, never silently skipped as successful attention routes.
- `forward` has three decode-cache reset sites. Capture-prefix and
  capture-small-query returns occur before those sites. Preserve the baseline
  call-site behavior; adding invalidation there would require a separate
  behavior-change investigation.
- With #1060's source-built shared-ABI extension, the policy test
  `test_flash_v100_decode_e4m3_respects_dflash_fp32_policy[False]` expects a
  partition hint of 64 but observes `None`. The parent GPU run reproduced
  this before any safety-net changes; A3 does not repair the expectation
  or alter the strategy. Full pass/fail comparison must retain this result.
