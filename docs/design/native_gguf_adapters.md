# Native GGUF architecture adapters

Native GGUF loading reconstructs configuration from checkpoint metadata and
keeps architecture-specific tensor conversions separate from the shared loader.
The loader handles shard discovery, pipeline-stage filtering, quantization policy,
and weight-loading orchestration. Adapters describe the checkpoint contract.

## Adding an architecture

1. Add a configuration builder under `vllm/transformers_utils/gguf_configs/` and
   register it in `_CONFIG_BUILDERS` in `gguf_config.py`. Register its configuration
   class in `_CONFIG_CLASSES` when Transformers does not supply that class.
   `supports_native_gguf_config()` is the shared admission check for both legacy
   builders and this registry. Missing dimensions and unsupported layouts must
   fail with a specific error; avoid defaults that silently change the model.
2. Add a text adapter in `vllm/model_executor/model_loader/gguf_adapters/` and
   register its lazy class reference in `_TEXT_ADAPTERS`. Its `build_name_map`
   checks the complete tensor directory before payloads are read. Record unused
   prediction layers explicitly. Define exact inverse conversions using the
   exporter as the reference, including fused projections, tensor order, and
   parameters that require FP32 storage or computation.
3. Reuse the shared tensor reader and dense shape/dtype checks in `common.py`.
   Preserve original expert quantization blocks when supported. Apply pipeline
   filtering before touching payloads; a transformation combining several source
   tensors must reject an incomplete filtered group.
4. For multimodal models, register a `VisionAdapterSpec` in `vision.py`. The spec
   declares the wrapper's language prefix and lazily imports the vision adapter.
   A vision adapter exposes `name_map`, `unquantized_modules`, and
   `weights(dtype, filtered_map=None)`. Construct the vision tower only on the
   pipeline stage that consumes images, and verify that every expected vision
   parameter was loaded. Processor assets and their geometry remain a separate
   requirement from the text checkpoint.
5. Reuse a native tokenizer preprocessing profile when its metadata agrees.
   Otherwise add a profile to `vllm/tokenizers/gguf.py` and compare real token IDs
   with the checkpoint's tokenizer. Segmentation, Unicode normalization, special
   tokens, and BOS/EOS insertion are independent parts of that contract.

## Completeness and numerical checks

An adapter may implement `validate_loaded_parameters(model, loaded)` to check
actual runtime parameter coverage after `model.load_weights()` and before weight
postprocessing. A complete name map alone cannot prove that a model's loader
accepted every tensor. Fused expert storage needs a per-projection loading ledger
because its public placeholder parameter names do not enumerate expert payloads.

GLM5-Next uses this hook for dense parameters, native expert banks, and the FP32
indexer gate copy. Its native and canonical GGUF MoE paths apply the model's
SwiGLU clamp. Fused activation kernels without a clamp argument are bypassed;
TurboMind expert-cache prefill rejects this configuration and directs users to
MMQ cache prefill instead. Existing models without a clamp retain their dispatch.

Keep the checkpoint's declared activation precision separate from its GGUF
storage type and the resolved runtime dtype. A BF16-trained draft can exceed
FP16 range even when its weights are stored as F16 and its GPU supports BF16.
DFlash2 uses its range-preserving path when that checkpoint executes in FP16,
including a draft that inherits FP16 from a heterogeneous target pipeline.
Native BF16 execution retains its normal path. Check intermediate activations
and actual draft acceptance; correct target answers alone can hide invalid
proposals that the verifier rejects on every step.

Speculative verification also changes the history a recurrent cache must retain.
The KPool tail keeps `round_up(pool_size + speculative_tokens, pool_size)`
positions so rejected proposals cannot overwrite the preceding committed pool.
Its seed and update kernels use physical page strides, including allocator
padding. Validate repeated reject/accept cycles and slot reuse against sequential
decoding before increasing context length.

## Validation for a new model

- Check a real tensor inventory, including shapes, data types, optional tensors,
  and separate prediction layers. Add failure cases for missing and unknown names.
- Test conversions with small real GGUF files whose values distinguish transposes
  and slice order. Assert that skipped pipeline stages never read their payloads.
- Compare native tokenizer output against a pinned tokenizer revision.
- Exercise the model's real parameter loaders and reject incomplete coverage.
- Test attention, quantization, and graph replay on every admitted GPU architecture
  before expanding capability selection. Reusing a Python/Triton path does not
  make an architecture-specific native CUDA binary portable.
- Finish with full-model generation, multimodal requests where supported, and
  speculative-decoding quality checks before enabling a persistent deployment.

Hardware capability helpers should expose a literal capability set that the
attention-backend documentation generator can resolve. This keeps generated
support tables aligned with runtime admission without duplicating model rules.
