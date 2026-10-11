# SM70 grouped attention family

`sm70_grouped.py` owns one codec-parameterized admission function,
`grouped_fp32_reason(codec, ...)`, and native provider selection. A rejection
returns a reason string; an admitted contract returns None. The immutable entries in
`GROUPED_CONTRACTS` record operator/provider identity, batch revision,
page/head/group bounds, alignment and legacy validation order. The FP16/E4M3
compatibility functions delegate to this owner with their old signatures and
return types. Backend callers import from this owner.

| Contract | Request groups | Query rows | KV page | Storage | Layout |
| --- | --- | --- | --- | --- | --- |
| FP16 | 1–4 | B1: 2–8; B>1: B×8 | 832 | FP16 | CUDA, D256/GQA6, one KV head, aligned Q/output/cache |
| E4M3 | 1–16 | B1: 2–8; B>1: B×8 | positive multiple of 16 | uint8 | D256/GQA6, aligned cache; batch requires revision 6 |
| E4M3 explicit groups | 1–16 | G×8, padded rows allowed | positive multiple of 16 | uint8 | compact group table and device row lengths |
| Shared codec binding | 1–16 | B1: 2–8; B>1: B×8; explicit padded groups | positive multiple of 16 | codec payload plus explicit row scales | FP16 Q/output, D256/GQA6, one or more KV heads |

All three contracts retain the 266240-token capacity ceiling. The E4M3
predicate retains its existing CPU descriptor admission for policy tests; the
native loader determines availability. FP16's CUDA guard, query/output
alignment and narrower shape limits are unchanged. Preserve the legacy
partition policy distinction too: raw string `"0"` rejects E4M3 while the
FP16 typed adapter yields zero, which does not reject. Policy convergence is a
separate behavior decision.

`sm70_fp16_grouped.py` retains the FP16 native launch, short-split capability and
sole workspace owner. `sm70_e4m3_grouped.py` retains the E4M3 native loader;
its predicates re-export the common wrappers. `sm70_grouped_scalar.py` and
`sm70_grouped_long.py` own the existing specialized family members. Their old
`sm70_e4m3_{scalar,long}` paths alias the same module objects, preserving mutable
state, caches and patches; original logger names remain unchanged. New shared
bindings query the installed native codec ABI and borrow scale planes from the
cache view. Legacy entry points retain their signatures and arithmetic. Long
and compact-tail scratch use the existing engine/stream resource owner; public
standalone calls retain their standalone owner.

Adding an entry declares admission and a provider; it does not implement a new
native storage reader or ABI. An unsupported codec fails closed. Shared
grouped/scalar/long instances cover FP16, E4M3, E5M2 and both ordinary INT8
identities. Python admission requires the corresponding installed binding,
not the payload byte width. Format-specific conversions remain in codec traits.

CPU tests generate boundary cases from the declarations and compare 2940
shape/device/revision combinations plus 98 policy/layout cases to frozen
legacy predicates, including exact FP16 rejection strings. Other checks cover
legacy module identity, missing revision import behavior, provider dispatch,
workspace cleanup and historical hashes of 22 native/family functions. Those
hashes describe the admission-refactor baseline, not subsequent native changes.
Current GPU qualification, artifact identity and remaining engine/latency gates
are recorded in the [codec implementation record](../../../../docs/design/architecture/kv_codec_unification.md).
