# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping
from dataclasses import asdict
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods

import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.kernels.gguf import (
    GGUFOperatorCapability,
    decoder_family,
)
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoEQuantConfig,
    MoEActivation,
    RoutedExperts,
    SharedExperts,
    apply_moe_activation,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.gguf_layout import GGUFLinearLayout
from vllm.model_executor.layers.quantization.gguf_native import (
    NATIVE_TYPES,
    dense_admission,
    native_available,
    native_dense,
    native_dequantize,
    pad_weight_tail,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__()
        self.unquantized_modules = unquantized_modules or []
        self.linear_layouts: dict[str, GGUFLinearLayout] = {}
        self.fallback_reasons: dict[str, str] = {}
        self.native_expert_storage = False
        self.canonical_expert_storage = False

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_name(self) -> QuantizationMethods:
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        # GGUF dequantization kernels use half precision (fp16) internally.
        # bfloat16 has precision issues on Blackwell devices.
        if current_platform.has_device_capability(100):
            logger.warning_once("GGUF has precision issues with bfloat16 on Blackwell.")
            return [torch.half, torch.float32]
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        return cls()

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg: dict[str, Any], user_quant: str | None, hf_config=None
    ) -> "QuantizationMethods | None":
        # When user explicitly specifies --quantization gguf, override
        # whatever quantization method is in the HF model config (e.g. fp8).
        if user_quant == "gguf":
            return "gguf"
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedLinearMethod()
            method = GGUFLinearMethod(self, self.linear_layouts.get(prefix))
            method.fallback_reason = self.fallback_reasons.get(prefix)
            return method
        elif isinstance(layer, VocabParallelEmbedding):
            if is_layer_skipped_gguf(
                prefix, self.unquantized_modules, self.packed_modules_mapping
            ):
                return UnquantizedEmbeddingMethod()
            if isinstance(layer, ParallelLMHead):
                return GGUFLMHeadMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, RoutedExperts):
            if self.native_expert_storage:
                if self.canonical_expert_storage:
                    from .gguf_turbomind_moe import GGUFTurboMindMoEMethod

                    return GGUFTurboMindMoEMethod(self, layer.moe_config)
                from vllm.model_executor.layers.quantization.gguf_moe import (
                    GGUFNativeMoEMethod,
                )

                return GGUFNativeMoEMethod(self, layer.moe_config)
            # TODO: Select UnquantizedFusedMoEMethod on unquantized layers.
            return GGUFMoEMethod(self, layer.moe_config)
        return None

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        """
        Interface for models to update module names referenced in
        quantization configs in order to reflect the vllm model structure

        :param hf_to_vllm_mapper: maps from hf model structure (the assumed
            structure of the qconfig) to vllm model structure
        """
        if self.unquantized_modules is not None:
            self.unquantized_modules = hf_to_vllm_mapper.apply_list(
                self.unquantized_modules
            )


def is_layer_skipped_gguf(
    prefix: str,
    unquantized_modules: list[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
):
    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        shard_prefixes = [
            prefix.replace(proj_name, shard_proj_name)
            for shard_proj_name in fused_mapping[proj_name]
        ]

        is_skipped = None
        for shard_prefix in shard_prefixes:
            is_shard_skipped = any(
                shard_prefix in module_name for module_name in unquantized_modules
            )

            if is_skipped is None:
                is_skipped = is_shard_skipped
            elif is_shard_skipped != is_skipped:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} "
                    "are quantized. All shards of fused layers "
                    "to have the same precision."
                )
    else:
        is_skipped = any(module_name in prefix for module_name in unquantized_modules)

    assert is_skipped is not None
    return is_skipped


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def _fused_mul_mat_gguf(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    native_enabled: bool = True,
    prefill_min_m: int = 8,
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # Preserve established routes for existing formats. Packaged upstream
    # operators are fallbacks for missing formats and explicit benchmark
    # candidates; TurboMind supplies the primary accelerated GGUF routes.
    if native_enabled and qweight_type not in DEQUANT_TYPES:
        native_result = native_dense(x, qweight, qweight_type, prefill_min_m)
        if native_result is not None:
            return native_result
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES or qweight_type in NATIVE_TYPES:
        block_size, type_size = quant_size(qweight_type)
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = (
            native_dequantize(qweight, qweight_type, *shape, x.dtype)
            if native_enabled and qweight_type not in DEQUANT_TYPES
            else None
        )
        if weight is None:
            if qweight_type not in DEQUANT_TYPES:
                raise ValueError(
                    f"No admitted native GGUF route for {quant_type_name(qweight_type)}"
                )
            weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        type_name = quant_type_name(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {type_name}")
    return y


def _fused_mul_mat_gguf_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    native_enabled: bool = True,
    prefill_min_m: int = 8,
) -> torch.Tensor:
    return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_fused_mul_mat_gguf",
        op_func=_fused_mul_mat_gguf,
        fake_impl=_fused_mul_mat_gguf_fake,
    )
    fused_mul_mat_gguf = torch.ops.vllm._fused_mul_mat_gguf

except AttributeError as error:
    raise error


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    activation_enum = MoEActivation.from_str(activation)

    def act(x: torch.Tensor):
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        apply_moe_activation(activation_enum, out, x)
        return out

    # lazy import to avoid triggering triton import in CPU backend
    from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size

    out_hidden_states = torch.empty_like(x)
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ops.ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ops.ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ops.ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ops.ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ops.ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        ops.moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def _fused_moe_gguf_fake(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    return torch.empty_like(x)


try:
    direct_register_custom_op(
        op_name="_fused_moe_gguf",
        op_func=_fused_moe_gguf,
        fake_impl=_fused_moe_gguf_fake,
    )
    fused_moe_gguf = torch.ops.vllm._fused_moe_gguf

except AttributeError as error:
    raise error


def _dequantize_gguf_rows(
    quant: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    native_enabled: bool = True,
) -> torch.Tensor:
    if qweight_type not in DEQUANT_TYPES | NATIVE_TYPES:
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    rows = quant.shape[0]
    dequant = (
        native_dequantize(quant, qweight_type, rows, hidden_size, dtype)
        if native_enabled
        else None
    )
    if dequant is None:
        dequant = ops.ggml_dequantize(quant, qweight_type, hidden_size, rows, dtype)
    # The legacy operator labels its dimensions in the opposite order;
    # its contiguous payload is still a row-major embedding matrix.
    return dequant.view(rows, hidden_size)


def _dequantize_gguf_rows_fake(
    quant: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    native_enabled: bool = True,
) -> torch.Tensor:
    return torch.empty((quant.shape[0], hidden_size), dtype=dtype, device=quant.device)


direct_register_custom_op(
    op_name="dequantize_gguf_rows",
    op_func=_dequantize_gguf_rows,
    fake_impl=_dequantize_gguf_rows_fake,
)
dequantize_gguf_rows = torch.ops.vllm.dequantize_gguf_rows


def _apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    native_enabled: bool = True,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES or qweight_type in NATIVE_TYPES:
        block_size, type_size = quant_size(qweight_type)
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = dequantize_gguf_rows(
            quant, qweight_type, hidden_size, dtype, native_enabled
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


def _apply_gguf_embedding_fake(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
    native_enabled: bool = True,
) -> torch.Tensor:
    return torch.empty((*x.shape, hidden_size), dtype=dtype, device=x.device)


try:
    direct_register_custom_op(
        op_name="_apply_gguf_embedding",
        op_func=_apply_gguf_embedding,
        fake_impl=_apply_gguf_embedding_fake,
    )
    apply_gguf_embedding = torch.ops.vllm._apply_gguf_embedding

except AttributeError as error:
    raise error


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(
        self, quant_config: GGUFConfig, layout: GGUFLinearLayout | None = None
    ):
        self.quant_config = quant_config
        self.layout = layout
        self.fallback_reason: str | None = None
        config = get_current_vllm_config_or_none()
        policy = config.kernel_config.sm70_gguf if config is not None else None
        self.native_enabled = policy.enabled if policy is not None else True
        self.prefill_min_m = policy.prefill_min_m if policy is not None else 8
        self.native_prepared = False
        self.canonical_projections = ()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "gguf_layout": self.layout,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def _prepare_dense_weight(self, layer: torch.nn.Module) -> bool:
        """Restore ordinary floating projections to the dense linear lifecycle.

        A floating shard inside a mixed projection is still a GGUF projection.
        Only a complete, unpacked matrix with no input-layout transform can
        replace the method, without changing the layer's forward semantics.
        """
        if isinstance(self, GGUFEmbeddingMethod) or self.layout is not None:
            return False
        qweight = layer.qweight
        if qweight.data_container:
            ids = (
                ["q", "k", "v"] if "q" in qweight.shard_id else sorted(qweight.shard_id)
            )
            sources = [
                (
                    qweight.data_container[qweight.shard_id_map[index]],
                    layer.qweight_type.shard_weight_type[index],
                )
                for index in ids
            ]
        else:
            sources = [(qweight, layer.qweight_type.weight_type)]
        if not sources or any(
            source_type not in UNQUANTIZED_TYPES
            or weight.ndim != 2
            or weight.dtype != self.params_dtype
            or not weight.is_floating_point()
            for weight, source_type in sources
        ):
            return False
        expected_n, expected_k = qweight.tensor_shape
        if (
            any(weight.shape[1] != expected_k for weight, _ in sources)
            or sum(weight.shape[0] for weight, _ in sources) != expected_n
        ):
            return False
        weights = [weight.to(qweight.device) for weight, _ in sources]
        weight = weights[0] if len(weights) == 1 else torch.cat(weights, dim=0)
        layer.register_parameter("weight", Parameter(weight.contiguous(), False))
        # Replace the layer's references, not shared checkpoint storage.
        del layer.qweight
        del layer.qweight_type
        layer.quant_method = UnquantizedLinearMethod()
        layer.quant_method.process_weights_after_loading(layer)
        return True

    def process_weights_after_loading(self, layer: torch.nn.Module):
        if self._prepare_dense_weight(layer):
            return
        ready = self.native_enabled and native_available()
        self.native_admission: dict[str, Any] = {
            "enabled": self.native_enabled,
            "extension_available": native_available(),
            "prefill_min_m": self.prefill_min_m,
            "reason": (
                "disabled_by_kernel_config"
                if not self.native_enabled
                else "native_extension_missing"
                if not native_available()
                else "device_not_cuda"
                if layer.qweight.device.type != "cuda"
                else None
            ),
        }
        types = layer.qweight_type.shard_weight_type.values() or (
            layer.qweight_type.weight_type,
        )
        for weight_type in types:
            if weight_type not in UNQUANTIZED_TYPES | DEQUANT_TYPES and not (
                ready and weight_type in NATIVE_TYPES
            ):
                raise ValueError(
                    f"GGUF {quant_type_name(weight_type)} requires an admitted "
                    "packaged native operator; check kernel_config.sm70_gguf "
                    "and the _C_gguf extension"
                )
        if (
            ready
            and layer.qweight.device.type == "cuda"
            and (
                not isinstance(self, GGUFEmbeddingMethod)
                or getattr(self, "canonical_lm_head", False)
            )
        ):
            qweight = layer.qweight
            from vllm.model_executor.layers.quantization.gguf_turbomind import (
                mixed_projection_capabilities,
                prepare_gguf_projections,
            )

            if qweight.data_container:
                ids = (
                    ["q", "k", "v"]
                    if "q" in qweight.shard_id
                    else sorted(qweight.shard_id)
                )
                sources = [
                    (
                        qweight.data_container[qweight.shard_id_map[index]].to(
                            device=qweight.device
                        ),
                        layer.qweight_type.shard_weight_type[index],
                    )
                    for index in ids
                ]
            else:
                sources = [(qweight, layer.qweight_type.weight_type)]
            from vllm.model_executor.layers.quantization.gguf_dmv import (
                eligible_sources,
                prepare_layer,
            )

            dmv_enabled = eligible_sources(sources)
            config = get_current_vllm_config_or_none()
            plane_policy = (
                config.kernel_config.sm70_gguf.projection_planes if config else False
            )
            self.native_admission["projection_planes"] = {
                "reason": "projection_shape_or_formats_not_qualified"
                if plane_policy
                else "disabled_by_kernel_config",
                "min_m": 8,
                "max_m": 8,
                "fallback": "canonical",
            }
            projections = prepare_gguf_projections(
                sources,
                self.params_dtype,
                self.native_enabled,
                self.prefill_min_m,
                input_layout=self.layout,
                dmv_enabled=dmv_enabled,
            )
            if dmv_enabled:
                admission = prepare_layer(layer, projections)
                self.native_admission["projection_planes"] = admission
                if admission["reason"] is None:
                    layer.gguf_tm_projections = torch.nn.ModuleList(projections)
                    self.canonical_projections = layer.gguf_tm_projections
                    self.native_admission["canonical_projections"] = [
                        p.admission() for p in projections
                    ]
                    if all(
                        p.input_layout_restored or getattr(p, "dmv_gdn_heads", False)
                        for p in projections
                    ):
                        self.layout = None
                    qweight.data_container.clear()
                    empty = Parameter(
                        torch.empty(0, dtype=self.params_dtype, device=qweight.device),
                        False,
                    )
                    set_weight_attrs(empty, vars(qweight))
                    layer.register_parameter("qweight", empty)
                    return
            from vllm.model_executor.layers.quantization.gguf_dmvq import (
                prepare_layer as prepare_temporary_projection,
            )

            temporary = prepare_temporary_projection(
                layer,
                sources,
                projections,
                plane_policy,
            )
            self.native_admission["temporary_projection"] = temporary
            if temporary["reason"] is None:
                layer.gguf_tm_projections = torch.nn.ModuleList(projections)
                self.canonical_projections = layer.gguf_tm_projections
                self.native_admission["canonical_projections"] = [
                    p.admission() for p in projections
                ]
                qweight.data_container.clear()
                empty = Parameter(
                    torch.empty(0, dtype=self.params_dtype, device=qweight.device),
                    False,
                )
                set_weight_attrs(empty, vars(qweight))
                layer.register_parameter("qweight", empty)
                return
            from vllm.model_executor.layers.quantization.gguf_iq3_gated import (
                prepare_iq3_gated_pair,
            )

            if self.layout is None and not isinstance(self, GGUFEmbeddingMethod):
                self.native_admission["gated_pair"] = prepare_iq3_gated_pair(
                    layer, sources, projections, self.native_enabled
                )
                from vllm.model_executor.layers.quantization.gguf_native_pair import (
                    prepare_native_gated_pair,
                )

                self.native_admission["mixed_gated_pair"] = prepare_native_gated_pair(
                    layer, sources, projections, self.native_enabled
                )
                from vllm.model_executor.layers.quantization.gguf_native_linear import (
                    prepare_native_linear,
                )

                self.native_admission["single_projection"] = prepare_native_linear(
                    layer, sources, projections, self.native_enabled
                )
            if self.layout is None and not isinstance(self, GGUFEmbeddingMethod):
                from vllm.model_executor.layers.quantization.gguf_qkvz import (
                    prepare_native_qkvz,
                )

                self.native_admission["qkvz_projection"] = prepare_native_qkvz(
                    layer, sources, projections, self.native_enabled
                )
                from vllm.model_executor.layers.quantization.gguf_qkv import (
                    prepare_native_qkv,
                )

                self.native_admission["qkv_projection"] = prepare_native_qkv(
                    layer, sources, projections, self.native_enabled
                )
            self.native_admission["canonical_projections"] = [
                projection.admission() for projection in projections
            ]
            if not isinstance(self, GGUFEmbeddingMethod):
                from vllm.model_executor.layers.quantization.gguf_small_output import (
                    prepare_small_output,
                )

                output_admission = prepare_small_output(
                    layer, sources, projections, self.native_enabled, self.layout
                )
                self.native_admission["small_output_projection"] = output_admission
                if output_admission["reason"] is None:
                    # Runtime M8 restores GDN heads while loading shared A.
                    # The opaque fallback restores them only when canonical
                    # preparation could not restore the weight layout.
                    self.layout = None
            self.native_admission["mixed_projection_operators"] = [
                asdict(c) for c in mixed_projection_capabilities(projections)
            ]
            if any(projection.kernel is not None for projection in projections):
                layer.gguf_tm_projections = torch.nn.ModuleList(projections)
                self.canonical_projections = layer.gguf_tm_projections
                if all(p.input_layout_restored for p in projections):
                    self.layout = None
                qweight.data_container.clear()
                # Replace this layer's parameter rather than mutating shared
                # checkpoint storage (for example a tied embedding parameter).
                empty = Parameter(
                    torch.empty(0, dtype=self.params_dtype, device=qweight.device),
                    requires_grad=False,
                )
                set_weight_attrs(empty, vars(qweight))
                layer.register_parameter("qweight", empty)
                return
            if qweight.data_container:
                ids = (
                    ["q", "k", "v"]
                    if "q" in qweight.shard_id
                    else sorted(qweight.shard_id)
                )
                layer.gguf_native_shard_weights = torch.nn.ParameterList(
                    Parameter(
                        pad_weight_tail(
                            qweight.data_container[qweight.shard_id_map[index]].to(
                                device=qweight.device
                            ),
                            layer.qweight_type.shard_weight_type[index],
                        ),
                        requires_grad=False,
                    )
                    for index in ids
                )
                layer.gguf_native_shard_types = tuple(
                    layer.qweight_type.shard_weight_type[index] for index in ids
                )
                qweight.data_container.clear()
                qweight.materialize((0,), dtype=self.params_dtype)
            else:
                qweight.data = pad_weight_tail(
                    qweight.data, layer.qweight_type.weight_type
                )
            self.native_prepared = True
            if hasattr(layer, "gguf_native_shard_weights"):
                prepared = list(
                    zip(layer.gguf_native_shard_weights, layer.gguf_native_shard_types)
                )
            else:
                prepared = [(qweight, layer.qweight_type.weight_type)]
            self.native_admission["projections"] = [
                dense_admission(
                    weight, weight_type, self.params_dtype, self.prefill_min_m
                )
                for weight, weight_type in prepared
            ]
            return
        qweight_type = layer.qweight_type.weight_type
        if not (
            qweight_type in UNQUANTIZED_TYPES
            or qweight_type in DEQUANT_TYPES
            or (ready and qweight_type in NATIVE_TYPES)
        ):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)

    def apply_fused_silu_and_mul(self, layer, x):
        if hasattr(layer, "gguf_dmv_operands"):
            from vllm.model_executor.layers.quantization.gguf_dmv import apply_layer

            return apply_layer(layer, x, fused=True)
        if hasattr(layer, "gguf_native_gated_records"):
            from vllm.model_executor.layers.quantization.gguf_native_pair import (
                apply_native_gated_pair,
            )

            return apply_native_gated_pair(layer, x)
        if not hasattr(layer, "gguf_iq3_gated_records"):
            return None
        from vllm.model_executor.layers.quantization.gguf_iq3_gated import (
            apply_iq3_gated_pair,
        )

        return apply_iq3_gated_pair(layer, x)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            if len(dtype) > 1:
                order = ["q", "k", "v"] if "q" in shard_id else sorted(shard_id)
                layer.gguf_shard_weights = torch.nn.ParameterList(
                    Parameter(
                        data_container[shard_id_map[index]].to(device=qweight.device),
                        requires_grad=False,
                    )
                    for index in order
                )
                layer.gguf_shard_types = tuple(
                    layer.qweight_type.shard_weight_type[index] for index in order
                )
                data_container.clear()
                # Retain the registered parameter contract without unused storage.
                qweight.materialize((0,), dtype=self.params_dtype)
                return
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layout is not None:
            x = self.layout.input_to_gguf(x)
        if self.canonical_projections:
            from vllm.model_executor.layers.quantization.gguf_turbomind import (
                apply_prepared_gguf_projections,
            )

            if hasattr(layer, "gguf_dmv_operands"):
                from vllm.model_executor.layers.quantization.gguf_dmv import apply_layer

                out = apply_layer(layer, x)
            elif hasattr(layer, "gguf_dmvq_records"):
                from vllm.model_executor.layers.quantization.gguf_dmvq import (
                    apply_layer,
                )

                out = apply_layer(layer, x)
            elif hasattr(layer, "gguf_qkvz_weights"):
                from vllm.model_executor.layers.quantization.gguf_qkvz import (
                    apply_native_qkvz,
                )

                out = apply_native_qkvz(layer, x)
            elif hasattr(layer, "gguf_qkv_weights"):
                from vllm.model_executor.layers.quantization.gguf_qkv import (
                    apply_native_qkv,
                )

                out = apply_native_qkv(layer, x)
            elif hasattr(layer, "gguf_small_output_records"):
                from vllm.model_executor.layers.quantization.gguf_small_output import (
                    apply_small_output,
                )

                out = apply_small_output(layer, x)
            elif hasattr(layer, "gguf_native_linear_records"):
                from vllm.model_executor.layers.quantization.gguf_native_linear import (
                    apply_native_linear,
                )

                out = apply_native_linear(layer, x)
            else:
                out = apply_prepared_gguf_projections(x, layer.gguf_tm_projections)
            if bias is not None:
                out.add_(bias)
            return out
        if hasattr(layer, "gguf_native_shard_weights"):
            weights = layer.gguf_native_shard_weights
            types = layer.gguf_native_shard_types
        elif hasattr(layer, "gguf_shard_weights"):
            weights = layer.gguf_shard_weights
            types = layer.gguf_shard_types
        else:
            weights = None
        if weights is not None:
            out = torch.cat(
                [
                    fused_mul_mat_gguf(
                        x, weight, weight_type, self.native_enabled, self.prefill_min_m
                    )
                    for weight, weight_type in zip(weights, types)
                ],
                dim=-1,
            )
            if bias is not None:
                out.add_(bias)
            return out
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else sorted(shard_id)
            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x,
                        qweight[start:end, :offset].contiguous(),
                        qweight_type,
                        self.native_enabled,
                        self.prefill_min_m,
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(
                x, qweight, qweight_type, self.native_enabled, self.prefill_min_m
            )
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(
        self,
        quant_config: GGUFConfig,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def get_fused_moe_quant_config(
        self, layer: RoutedExperts
    ) -> FusedMoEQuantConfig | None:
        return None

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if layer.apply_router_weight_on_input:
            raise NotImplementedError(
                "Apply router weight on input is not supported for"
                "fused GGUF MoE method."
            )

        return fused_moe_gguf(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights,
            topk_ids,
            layer.w13_qweight_type.weight_type,
            layer.w2_qweight_type.weight_type,
            layer.activation.value,
        )


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x,
            qweight,
            qweight_type,
            hidden_size,
            dtype=self.params_dtype,
            native_enabled=self.native_enabled,
        )


def _gguf_lm_head_projection(
    x: torch.Tensor,
    raw: torch.Tensor,
    codes: torch.Tensor,
    stats: torch.Tensor,
    k_ld: int,
    q_ld: int,
    minimum_m: int,
    maximum_m: int,
    native_enabled: bool,
    prefill_min_m: int,
    source_type: int = int(WeightType.Q4_K),
    bits: int = 4,
    group_size: int = 32,
) -> torch.Tensor:
    rows = x.numel() // x.shape[-1]
    if minimum_m <= rows <= maximum_m:
        return torch.ops.vllm.prepared_gguf_projection(
            x,
            codes,
            stats,
            None,
            0,
            bits,
            group_size,
            k_ld,
            q_ld,
            raw.shape[0],
            raw.shape[0],
            [],
            [],
        )
    return fused_mul_mat_gguf(x, raw, source_type, native_enabled, prefill_min_m)


def _gguf_lm_head_projection_fake(
    x: torch.Tensor,
    raw: torch.Tensor,
    codes: torch.Tensor,
    stats: torch.Tensor,
    k_ld: int,
    q_ld: int,
    minimum_m: int,
    maximum_m: int,
    native_enabled: bool,
    prefill_min_m: int,
    source_type: int = int(WeightType.Q4_K),
    bits: int = 4,
    group_size: int = 32,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], raw.shape[0]), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    op_name="gguf_lm_head_projection",
    op_func=_gguf_lm_head_projection,
    fake_impl=_gguf_lm_head_projection_fake,
)


class GGUFLMHeadMethod(GGUFEmbeddingMethod):
    """Vocabulary projection policy with separate embedding storage semantics."""

    def process_weights_after_loading(self, layer):
        weight_type = layer.qweight_type.weight_type
        raw = layer.qweight.detach()
        calibration = {
            (int(WeightType.Q4_K), (62080, 2880)): (2, 16),
            (int(WeightType.Q6_K), (62080, 2100)): (1, 20),
        }.get((weight_type, tuple(raw.shape)))
        minimum_m, maximum_m = calibration or (2, 16)
        reason = None
        if not self.native_enabled:
            reason = "disabled_by_kernel_config"
        elif self.params_dtype != torch.float16:
            reason = "requires_fp16_activations"
        elif raw.device.type != "cuda" or not current_platform.is_device_capability(70):
            reason = "requires_sm70"
        elif calibration is None:
            reason = "lm_head_shape_or_format_has_no_calibration"
        self.canonical_lm_head = reason is None
        self.lm_head_capability = (
            GGUFOperatorCapability(
                decoder_family(weight_type),
                quant_type_name(weight_type),
                "gguf_lm_head_projection",
                True,
                min_m=minimum_m,
                max_m=maximum_m,
            )
            if self.canonical_lm_head
            else None
        )
        super().process_weights_after_loading(layer)
        if self.canonical_lm_head and self.canonical_projections:
            assert self.lm_head_capability is not None
            # Keep the faster M1 route and unmeasured M intervals. This raw
            # parameter is separate from the canonical streams and embedding.
            layer.register_parameter(
                "gguf_lm_head_raw", Parameter(pad_weight_tail(raw, weight_type), False)
            )
            self.native_admission["lm_head"] = {
                "operator": self.lm_head_capability.operator,
                "min_m": minimum_m,
                "max_m": maximum_m,
                "reason": None,
                "raw_fallback": "outside_measured_m_band",
            }
        else:
            self.native_admission["lm_head"] = {
                "operator": "gguf_lm_head_projection",
                "min_m": minimum_m,
                "max_m": maximum_m,
                "reason": reason or "canonical_kernel_unavailable",
            }

    def apply(self, layer, x, bias=None):
        if hasattr(layer, "gguf_lm_head_raw"):
            assert self.lm_head_capability is not None
            projection = layer.gguf_tm_projections[0]
            output = torch.ops.vllm.gguf_lm_head_projection(
                x,
                layer.gguf_lm_head_raw,
                projection.codes,
                projection.stats,
                projection.gguf_tm_k_ld,
                projection.gguf_tm_q_ld,
                self.lm_head_capability.min_m,
                self.lm_head_capability.max_m,
                self.native_enabled,
                self.prefill_min_m,
                projection.source_type,
                projection.kernel.bits,
                projection.kernel.config.group_size,
            )
            return output if bias is None else output + bias
        return super().apply(layer, x, bias)


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]
