# Independent checkpoint loading for speculative drafts

A GGUF target can use a separate safetensors draft checkpoint. V2 draft
loading must pass `SpeculativeConfig.draft_load_config` to the model loader;
otherwise the target's GGUF loader is applied to the safetensors checkpoint.
Draft quantization configuration must use the same independent load config.
When no override is provided, the existing target loader remains the fallback.

The change preserves target embedding/head sharing, including per-layer MTP
shared-head aliases. It changes checkpoint loading only; it does not quantize
MTP weights or change speculative sampling.

The focused source checks pass eight embedding/head-sharing and loader cases,
and three draft-quantization configuration cases. The mixed-format regression
selects the normal safetensors loader for a draft paired with a GGUF target.
The native modules come from the normal precompiled-package build. Installed
wheel checks and model integration are still pending.
