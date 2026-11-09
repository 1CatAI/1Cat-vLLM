# Loaded SM70 route reporting

The common linear selectors record their actual decision and rejection reasons
on the engine's `KernelConfig`. No second kernel registry or parallel capability
predicate is introduced. NVFP4, AWQ and FP8 use the same reporting lifecycle as
other linear kernels. Candidates below the chosen kernel are marked
`lower_priority`; reporting does not probe them or claim they are unsupported.

The diagnostic field is excluded from `KernelConfig.compute_hash`. Recording a
choice cannot invalidate a graph cache, alter priority or mutate environment.
Separate engines own separate dictionaries.

After initialization, the API process fetches worker reports once. Startup
logging and `/v1/sm70/acceleration` then read the same cached table. The report
contains resolved policies discovered from the existing SM70 configuration
fields, actual selector decisions, and the final kernel instances in loaded
layers. Final instances distinguish a retained kernel from an earlier candidate
whose QPN4 workspace preparation fell back. Local shapes and rejection reasons
remain available per rank.

FP16 and other operator preparation is reported from existing layer flags and
buffers. Resident CUDA packed-buffer bytes are counted once per storage, without
reading tensor values or launching an operator. This excludes allocator overhead,
CUDA graphs, temporary workspaces and KV cache, and is distinct from the
configuration-time Flash-Next packed-copy estimate.

An unrelated AWQ/FP8 configuration no longer claims the 27B NVFP4 release profile
or its eight required paths. The explicit strict-profile option retains its
existing validation semantics. Flash-Next retains its own required paths.
Missing reporting support in a custom executor is explained without changing
serving routes.

These are loaded choices and runtime capability/preparation guards, not request
hit counters. Attention/verifier/collective/prefill category migrations still
need to replace their retained legacy policy rows with their own ordinary
selector/backend decisions. This reporting change adds no route admission,
numerical kernel, precision setting or environment control.
