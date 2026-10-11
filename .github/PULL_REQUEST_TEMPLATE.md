<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED.

## Purpose

## Test Plan

## Path coverage

For documentation-only changes, write **N/A — documentation/offline tooling only**
and list the applicable CPU/documentation checks. GPU tests are not required.
See the [1Cat development guide](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/contributing/1cat-development.md).

For runtime changes, distinguish supported ranges, accelerated subsets and
measured cases. Include affected threshold neighbors and rejected/fallback cases;
label context units (live tokens, score columns, page capacity or graph padding).

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Runtime acceptance (required for runtime changes)

Follow the [runtime acceptance contract](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/contributing/1cat-development.md#runtime-acceptance-contract).
Use N/A with a reason for unaffected dimensions; missing evidence is unverified.

- Existing family path/stages reused; missing capability and reason for any new provider:
- Admission versus model qualification versus tuning restrictions; fallback/rejection reason:
- Equivalent startup inputs and resolved policy; affected paired entry, process, cache or artifact checks:
- Boundary and interaction tests; output/state contract; justified omissions:
- Coverage gaps, responsible component and linked follow-up decision; evidence for any broader model/default-on claim:

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in:
- Required CLI options and environment switches:
- KV cache dtype used for the benchmark:
- Wheel SHA or source commit:
- PYTHONPATH, source overlays, or external native libraries used (write "none" if absent):
- User-entry native-hit and output/state evidence; comparable TTFT/prefill and pure decode results:
- Additional conversion, allocation, synchronization or launch costs in the affected execution path:
- New/changed environment metadata and generated reference check:
- Route snapshot command and intentional changed rows (write "none" for refactors):
- SM70 registered variables / unregistered reads / model-parameter restrictions, before -> after:

## Test Result

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [ ] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [ ] The test plan, such as providing test command.
- [ ] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
