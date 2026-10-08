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
- All six `test_runner_does_not_dispatch_short_prefill_as_tail` variants fail
  on #1060 because their synthetic `GPUModelRunner` lacks `device`, which
  `execute_model` reads. The full baseline run reports 1656 passed / 7 failed
  including the E4M3 case above. These fixture repairs are outside A3.

- The first full Flash-Next model run on the #1060 + #1028 integration stopped
  during compiled warmup: its HC caller passes nine arguments but the reused
  #1060 `_C` binary exposes the older eight-argument `sm70_hc_ll_down_out` ABI.
  The #1028 integration adds native changes (including this optional argument);
  its CPU and host-cache unit tests did not exercise this full-model path.
  This is an integration-artifact mismatch, not an A3 attention regression.
  Preserve `a3-step1c/logs/host-parent-core-abi-failure.log` on 54633 and build
  the integration's `_C` in task-owned `a3-native-abi`. Do not repair production
  policy or disable HC to make the parity gate pass. All later A3 host
  integration native sources are identical to this pinned integration, verified
  with `git diff` over CMakeLists.txt and csrc; both comparison arms must use
  the same rebuilt artifact. The original #1060 eight-argument workloads retain
  their original binary contract.

- DFlash2's first real-model baseline with compile mode 3, FULL graphs and a
  1024-token budget fails in compiled warmup: the existing SM70 profile extends
  a compile range to 1025 while DFlash's hidden-state buffer has capacity 1024.
  This is reproduced on #1060 before A3 changes. Preserve
  `dflash-parent-compile-range-failure.log` and its exact engine JSON on 54633.
  The parity workload now explicitly requests compile mode 0 with FULL CUDA
  Graphs and capture size 8 on both arms; the buffer-capacity bug is not repaired
  here. This establishes a separate contract, not success of the failed compile
  configuration. Backend graph replay remains a mandatory numerical/pointer gate.
