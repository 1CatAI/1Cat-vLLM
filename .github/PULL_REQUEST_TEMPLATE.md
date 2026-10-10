<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED.

## Purpose

## Test Plan

## Path coverage

For documentation-only changes, write **N/A — documentation/offline tooling only**
and list the applicable CPU/documentation checks. GPU tests are not required.
See the [1Cat development guide](https://github.com/1CatAI/1Cat-vLLM/blob/main/docs/contributing/1cat-development.md).

| Path / provider | Format | Key conditions (shape, TP/EP, graph, capability) | Expected selection / fallback | Evidence (source / static / executed / measured) | Unverified scope |
| --- | --- | --- | --- | --- | --- |
| | | | | | |

## Acceleration and benchmark contract (required for performance changes)

- Default enabled or opt-in:
- Required CLI options and environment switches:
- KV cache dtype used for the benchmark:
- Wheel SHA or source commit:
- PYTHONPATH, source overlays, or external native libraries used (write "none" if absent):
- User-entry route-hit, speed, and output-quality evidence:
- Existing kernel/backend registry and capability-rejection reasons:
- Per-engine configuration and deprecated-variable compatibility:
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
