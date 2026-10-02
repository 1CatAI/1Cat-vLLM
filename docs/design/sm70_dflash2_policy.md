# SM70 DFlash2 verifier policy

DFlash2 verifier decisions are resolved once in
`SpeculativeConfig.sm70_dflash2`. Model loading captures this engine's policy;
forward execution and graph capture do not obtain decisions from another
engine's process environment. Resolved graph-affecting options participate in
the speculative compilation hash. The existing kernel and attention frameworks
continue to own dtype, shape, layout, and native-operator admission.

The verifier is a coordinated pipeline rather than a linear kernel: it includes
context projection, graph replay, GDN metadata, normalization, sparse rejection,
and logits ordering. Keeping its scheduling policy in speculative configuration
avoids adding a parallel kernel selector. Serialized target FP8 QPN8 permission
is passed to the existing `KernelConfig.sm70_fp8` policy.

## Defaults and compatibility

The model quality boundary is retained in
`models/config.py::sm70_dflash2_verifier_qualified`: Qwen3.5 conditional generation,
FP16 activations, hidden size 5120, 24 attention heads, four KV heads, head width
256, DFlash with seven speculative tokens and selector top-k 16, PP1, and no DBO
or multi-ubatch scheduling. These restrictions describe the validated pipeline,
not the operator's implementation limits. KV dtype, quantization, and TP are
not added to this qualification.

Qualified configurations retain their former automatic options, including FP32
logits, reranking, and dense tie ordering. Candidate-only ordering stays off.
Other configurations retain the registered legacy defaults. Explicit typed
fields take precedence over legacy environment aliases. No default changes or
online weight requantization promotions occur in this migration.

The 19 DFlash-specific environment names remain readable, emit deprecation
warnings, and map to the corresponding fields. Keep compatibility for one
released version containing this migration, then remove the names in the
following release. The shared `VLLM_SM70_FP8_QPN8` alias also serves the existing
serialized FP8 policy and is not removed by this migration. Five former public
DFlash controls become internal compatibility entries. Name-count reduction
will occur when the compatibility interval ends, not when documentation hides
them.

## Local layout admission

Combined GDN tail materialization copies bits without arithmetic. Its capability
predicate accepts eight FP16 query rows and local `(qkv, z, ba)` widths
`(2560, 1536, 12)`. The constructor previously also required TP4 and model hidden
size 5120. Those global checks are replaced by the actual local layout, and the
forward call uses the same predicate. Ordinary 27B TP2 still falls back because
its local widths differ. An explicitly enabled TP2 layer with eight local key
heads and 24 local value heads can now use the identical copy kernel.

The rest of the pipeline's model quality boundary remains intact. Lifting it
requires paired requests, MBPP, long-context retrieval and Chinese QA, plus
target speed and concurrency/workspace checks. Copy-only layout admission
requires the bitwise copy and graph replay oracle and a focused operator timing;
it is not a 35B model performance baseline.

## Regression command

```console
python -m tools.sm70_route_snapshot --category dflash2 \
  --baseline-ref <previous-main-sha> \
  --expected-changes tests/config/data/sm70_dflash2_capability_changes.json
```

The command executes historical getter/default and GDN constructor predicates.
The standard 324 configurations preserve routing. Legacy parser/override edges
are also covered. Exactly one explicit TP2/local-layout edge is an intentional
admission change. Snapshots describe permissions and admission, not observed
request kernel hits. Existing linear-category snapshots remain separate gates.

## Progress accounting

At the integration base containing #800: registered names remain 999, public
SM70 controls decrease from 36 to 31, checked unregistered reads remain zero,
and `config/vllm.py` environment-write sites decrease from 20 to 19. Two combined
copy model/TP locks are replaced with one operator capability predicate. No
CUDA/C++ numerical implementation or computation precision changes.
