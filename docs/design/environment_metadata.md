# Environment registration metadata

Environment parsing and its explanation live together in `vllm/envs.py`.
`env_var` wraps the existing getter and attaches a description, category,
declared default, effective unset default, automatic conditions and acceleration
paths. The wrapper delegates lazily; generating documentation never invokes a
getter or reads a deployment's environment.

Use engine configuration for acceleration decisions. Existing kernel selectors,
attention validators and model policies still own routing. These metadata labels
document their controls; they do not form another acceleration registry.

## Adding or changing a control

Register the getter with complete metadata. Explain any computed default and
automatic model policy, including the difference between permission to use a
kernel and the local capability checks that select it. Categories are
`configuration`, `tuning`, `experimental`, `debug` and `deprecated`.

```python
"VLLM_EXAMPLE": env_var(
    lambda: bool(int(os.getenv("VLLM_EXAMPLE", "0"))),
    description="Explain the actual user control and its replacement, if any.",
    category="configuration",
    declared_default="False",
    effective_default="False",
    automatic_conditions=(),
    acceleration_paths=(),
),
```

Regenerate the reference and run its checks:

```bash
.venv/bin/python -m tools.generate_env_reference
.venv/bin/python tools/pre_commit/check_env_metadata.py
.venv/bin/python -m tools.generate_env_reference --check
```

Pre-commit/CI rejects registrations without metadata, invalid categories,
unexplained literal default discrepancies and stale generated documentation.
The existing registration check from #711 rejects unregistered direct reads.
Its legacy exception list is now empty; do not add new exceptions.

## Legacy direct reads and compilation

The 137 legacy direct-read names are registered as raw optional strings.
`None` describes absence at registration; their consumers still apply the
defaults listed in `automatic_conditions`. This preserves routing without
inventing one converted default for consumers that currently disagree. A
category migration must resolve that policy into configuration before replacing
its direct reads. Do not treat the raw string `"0"` as a Python boolean.

Registration adds those names to the compilation cache fingerprint. Deployments
may compile a fresh cache after upgrading. Debug controls retain conservative
cache participation until their consumers have been audited; diagnostic-only
entries can then join `ignored_factors` with evidence that they cannot change
compiled code.

The initial metadata migration preserves all 832 pre-existing getter ASTs and
registers 137 presence getters. It corrects only three `TYPE_CHECKING`
declarations: Ray v2's default is `True`, the DP master IP is `127.0.0.1`, and
the XGrammar cache is 512 MB. Runtime getters retain their previous values.
No kernel, numerical precision, experiment default or model qualification is
changed by this migration.

## Native environment readers

The registration check also covers C/C++/CUDA sources in `csrc`,
`flash-attention-v100`, `flash_qla`, and bundled `lmdeploy`. It finds literal
keys and simple constant aliases passed to `getenv` and native env helper
functions; C/C++ comments and ordinary diagnostic strings are ignored. Dynamic
name construction still needs an explicit registration for each resulting name.

Native-only registrations return raw `str | None`. They do not parse a native
boolean, assign an environment value, or replace the native fallback. Their
metadata distinguishes the raw Python default (`None`) from the native unset
default, and lists the actual reader locations. Their presence intentionally
invalidates compilation caches that previously omitted these controls.

For example, `VLLM_FLASH_V100_XQA_E4M3_G6_MERGED_WAVE_LAUNCH` is default-off
in the standalone long-attention implementation and default-on in bundled
Flash-V100. That existing difference is documented per implementation. Changing
it requires a separate route/quality/performance comparison in the attention
migration; registration alone preserves both defaults.
