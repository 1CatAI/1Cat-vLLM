# Qwen4Exp QSA DCP2 implementation and V100 validation plan

Status: allocator and operator prototype, 2026-09-24 (America/Vancouver).
CPU ownership/prefix/offload geometry and bounded V100 operator tests have
passed. QSA DCP serving is still disabled; the full model has not run with DCP2.

## Frozen source stack

The experiment branch starts at 1CatAI `main` `d49e32b3587d4d34ffccb0ffd376e63974b06c88`.
It merges the CT252 integration head `b4fef533ec427747a2d6c65a8c2b2374bde3d679`
(Leon #19 and 1CatAI #617/#598), then graph-padding fix #680 at
`790edbbdd6f23606c314c086f6834d1f67f5d906`. The two merge commits are
`ae496940d4` and `15a5bbe670`. The currently deployed CT252 image remains
based on `b4fef533ec`; this branch is an isolated source baseline, not a
deployment candidate. Upstream reference: vllm-project/vllm#57431 at
`def69eb79ccd1ba33179c4551fa706996e926521`, with #56723 as an ownership
reference. Record new upstream heads and re-diff before importing later fixes.

Local assessment and reproducible capacity calculation:
`/home/leon/1Cat/research/qsa-dcp2-assessment-20260924/assessment.md`.

## Goal and initial scope

Implement QSA decode context parallelism on TP4/V100, beginning with DCP2 for
the 12 target QSA layers while the single physical MTP draft QSA layer remains
replicated. Keep the target and draft E4M3 KV format, MTP3, prefix caching,
grouped CPU offload, and SM70 kernels available. `DCP1` must remain a working
control. This is a per-cache ownership change: target main K/V is sharded;
target selector, GDN, PLE, ring, and draft state are replicated.

The capacity calculation predicts about 1.18M global tokens for target-only
DCP2 before new workspace, or about 1.13M with 256 MiB/card of new workspace,
against a measured 775,096-token DCP1 baseline. These are layout projections,
not acceptance results. Four simultaneous full 262,144-token contexts are a
reference point, not a mandatory success threshold. Correctness, stability,
reproducible evidence, and service recovery are acceptance gates. Capacity,
latency, throughput, and prefix retention are measured optimization results,
not pass/fail thresholds; review their trade-offs before any deployment decision.

## Implementation sequence

1. **Lock the baseline.** Save `git rev-parse HEAD`, submodule/native build
   identities, model checkpoint and scale-overlay hashes, launch arguments,
   CUDA/NCCL versions, and DCP1 capacity/startup logs. Run the relevant CPU
   suites on this merged branch and confirm that imports resolve from this
   worktree. Compare effective config against the CT252 image before A/B.
2. **Define cache ownership and page geometry in the core.** Express each
   cache group's global token span and physical local slots separately. In
   `_get_csa_linear_cache_tuples`, `_get_kv_cache_groups_csa_linear`, and
   `_get_csa_linear_tensor_layout`, make target main K/V DCP2 local while
   selector, GDN, PLE, ring, and draft stay replicated. The target-only layout
   must support a 3,200-global-token target page backed by 1,600 local E4M3
   slots and a differently sized replicated draft page. Reject unsupported
   group combinations explicitly. Keep allocator accounting, slot mapping,
   block hash grid, and `get_max_concurrency_for_kv_cache_config` consistent.
3. **Make coordinator and offload ownership aware.** Replace the blanket
   `dcp_world_size == 1` hybrid rejection only for the supported Qwen4Exp
   layout. The coordinator must reserve/evict blocks according to each owner's
   global span. In `vllm/v1/kv_offload/base.py`, replace the uniform DCP factor
   for every group with per-owner mapping; verify grouped CPU and filesystem
   restore, prefix hashes, Mamba retention, null block, and retry ordering.
   CPU simulations should cover prompt/decode, prefix hit, partial hit,
   eviction/reload, and two concurrent requests before any model startup.
4. **Port QSA DCP metadata and math.** Adapt #57431's local index mapping to
   1Cat's actual selection ABI (token IDs and `-1` padding) and K/V tensor ABI
   (`[blocks, 2, block_size, kv_heads, head_size]`). Keep the original selected
   IDs intact for MTP's next steps. Handle owner changes, zero selected tokens,
   ragged graph padding, and offsets at the global/local page boundary.
   Compare per-rank partial outputs and log-sum-exp against an FP32 reference,
   then combine with the correct LSE base and output gate. The DCP2 local route
   requires a qualified G12 SM70 path; do not claim the existing G6/page4
   optimized route applies unchanged.
5. **Bound scratch and graph behavior.** Allocate one model-level target
   localized-index workspace, rather than #57431's per-layer
   `empty_like(topk_buffer)`: the latter would use about 769 MiB/card across
   12 target layers. Size query/output/LSE/collective scratch from actual
   maxima; preserve stable addresses under graph capture and replay. Test
   prefill, decode, mixed batches, MTP3 rounds, and repeated graph replay.
   Capture memory before/after each allocation and reconcile it with the
   profiler's KV budget.
6. **Advance by bounded GPU gates.** First run TP2/DCP2 synthetic QSA
   operator tests on leased V100s, then TP4/DCP2 with the real checkpoint at
   short context and no offload. Enable MTP3, prefix caching, grouped offload,
   and long-context pressure one at a time. A later branch may evaluate
   target+draft DCP2 only after target-only data exists; it is not included
   in the first acceptance result.

## Validation matrix and evidence

| Check | Cases | Evidence required |
| --- | --- | --- |
| CPU ownership | DCP1/2; main, selector, GDN, PLE, ring, draft; empty/ragged/two-request pages | Exact slots, block counts, global spans, hash and eviction/restore invariants |
| QSA operator | E4M3 and FP16 reference; short/long selection; empty owner; page boundary; prefill/decode | Selected IDs and output parity, finite tensors, per-rank LSE and gate values |
| Graph/MTP | Captured and eager, repeated replay, mixed prefill/decode, MTP3 | Same output tokens and acceptance decisions; stable scratch pointers and bounded memory |
| Integrated cache | Prefix miss/hit/partial hit, CPU/filesystem offload and restore, C1/C4 at several context lengths | Correct continuation and cache state after eviction/reload; no deadlock or stale block |
| Capacity/performance profiling | Same checkpoint/image/config except DCP1 vs DCP2; C1/C4, short/long prompts, repeated runs | Actual pool bytes/blocks/global tokens, peak/card, prefill TTFT, full MTP3 round latency, output tok/s, accept rate, prefix hit/eviction |

For numerical tests, set explicit tolerances from the DCP1 FP32 reference
before inspecting candidate outputs; exact token equality is required on the
deterministic text replay set. Record sampling seed, prompt hashes, output
hashes, and any divergence position. Measure cold and warmed runs separately.
An HTTP 200 or successful model load does not constitute acceptance.

## Final acceptance and optimization targets

The hard acceptance gates are:

1. The supported TP4/DCP2 target-only configuration runs the real checkpoint
   with MTP3, prefix caching, grouped offload, mixed prefill/decode, and graph
   replay. Fixed deterministic replay has the same generated token IDs and
   MTP acceptance decisions as DCP1. Operator outputs and LSE satisfy the
   predetermined numerical tolerances. No stale cache data, non-finite values,
   out-of-bounds access, deadlock, or unexpected OOM occurs in the validation
   matrix.
2. The DCP1 control remains functional. Results identify exact source/image,
   checkpoint, configuration, prompts, seeds, and raw logs so the comparison
   can be reproduced. Capacity and performance measurements must be reported
   even when their values miss an optimization target.
3. The temporary GRS lease is released, its cleanup and resident recovery are
   confirmed, and the original service passes a real serving request.

Optimization targets for the first A/B are at least 20% more usable KV
capacity than DCP1 at the same GPU budget, no more than 5% lower output
throughput, and no more than 10% higher TTFT in the primary scenarios. Repeat
each scenario at least five times and report the distribution, peak memory,
and prefix behavior. These numbers and the four-full-context reference are
not hard gates. Missing a target calls for profiling and an explicit trade-off
decision; it does not erase a technically correct result or automatically
prevent the next iteration.

## V100 execution and rollback

GRS currently reports `llm252/v100-0..3` in an active CT252 resident lease,
although compute utilization is zero. Before GPU work, reread GRS state and
the service recovery profile. Acquire the complete TP4 set in one temporary
lease, confirm it is `active`, bind the exact temporary Docker container ID,
report only real activity, and retain logs/results outside the container.
Keep the production image/config and its resident lease untouched. Release
the temporary lease on every exit path, verify cleanup and resident recovery,
then check the actual 18080 serving path and its configured model. A failed
recovery is an incident, not a benchmark result.

Do not roll this branch into production as part of the experiment. If any
correctness, memory, or recovery gate fails, keep the DCP1 service, save the
minimal reproducer and exact tested head, and fix the isolated branch before
another A/B. A decision to deploy follows a separate full-stack acceptance
and review of measured capacity, latency, and output parity.

## Current checkpoint

- Source merges completed without conflicts; `git diff --check` and merge
  pre-commit hooks passed.
- CPU baseline: 136 passed, 12 GPU-only skipped across QSA graph padding,
  Mamba retention, and E4M3 MTP overlay tests on this worktree.
- `212be8cf8b`: per-owner `dcp_sharded` metadata and global block spans;
  per-layer physical page sizes; shared Mamba views retain the correct stride.
- `5c93a7636d`: separate localized selection buffer using 1Cat's all-ID / -1
  ABI; optional base-2 LSE and FP32 attention output; per-rank output gating
  is rejected when LSE is requested.
- `6f6bbeee1d`: owner-aware hybrid prefix lookup, manager allocation, and
  grouped CPU offload budgets. The scheduler and worker agree on group spans
  and budgets even with heterogeneous Mamba padding.
- The real allocator, with the model's 12 target QSA / 36 GDN / 1 draft / 1 PLE
  owner counts and synthetic recurrent tensors that fit their padded pages,
  reproduces 775,096 DCP1 tokens and projects 1,178,375 target-only DCP2 tokens
  at the same `547 * 11,980,800` byte budget. This excludes additional runtime
  scratch and does not establish deployed capacity.
- CPU coverage: 14 QSA layout/ownership tests, 147 retention/grouped-offload
  tests, and 66 general cache-utils tests passed. One general DeepSeek test
  lacks `max_in_flight_tokens` in its mock config and fails identically at
  the untouched baseline `978f38fb9b`; it is not a DCP regression. General
  config tests require an explicit CPU platform in this GPU-free environment.
- V100 suites: 20 localization tests (13 GPU cases plus 7 argument guards)
  and 4 G12 attention cases passed. Attention covers FP16/E4M3, one/multiple
  split-K partitions, empty owners, nonidentity page tables, and the original
  DCP1 call. Predetermined tolerances: output `rtol=3e-3, atol=2e-3`; base-2 LSE
  `rtol=1e-4, atol=2e-3` against an independent FP32 reference.
- GPU execution used the existing `b4fef533ec` image with only the two QSA
  operator modules mounted. Their SHA256 values are recorded below; both
  match committed source. This is single-GPU simulation of each rank's math,
  not a distributed NCCL or model test. The graph test's synchronization call
  was renamed to `torch.accelerator.synchronize` after the GPU run to satisfy
  the repository lint rule; the tested kernels are unchanged.
- Lease `lease-c79436ef-946a-40d8-ab15-0e42b9508169` is released; all three
  temporary containers were cleaned, and the displaced resident reports
  `restored=1`. The 18080 health check returned 200, `/v1/models` returned
  `QWEN-Flash`, and a bounded chat request returned `OK` with the original
  `b4fef533ec` service fingerprint.

Operator SHA256:

```text
qsa.py      266b861cdf48c86529d8877538b1b30933a4b51134dde1dd118ae8578ffd3e05
qsa_dcp.py  b941fce8e2fe827be11fde79c69f3865d33fe3eebf8eff6e440cefc4c3490c7b
```

Evidence directory: `/home/leon/1Cat/research/qsa-dcp2-assessment-20260924/`:
`v100-indices-r1.log`, `v100-attention.log`, `cpu-cache-utils.log`,
`baseline-deepseek-fixture.log`, and `restored-serving-smoke.json`.

## Next integration boundary

1. Emit the correct target/selector/ring/draft ownership from the model's
   real cache specs and propagate it through worker block tables and slot
   mapping. Existing QSA DCP rejection must remain until these agree.
2. Wire localized metadata, query gather, LSE/output collectives, and the final
   output gate into `Qwen4ExpQSAFlashAttentionImpl`. Validate actual two-rank
   communication, including empty owners and graph replay. TP4 has two KV
   heads, so DCP pairs must stay within each replicated KV-head pair.
3. Allocate model-level persistent workspaces and measure their true footprint.
   Current LSE output is an operator API, not a graph-memory optimization.
4. Build a complete image from this main-plus-PR stack before full-model A/B;
   the old image used for operator tests does not validate current main's
   native libraries. Then qualify MTP3 metadata, offload transfers/restart,
   long-context cache pressure, output parity, and serving performance.
