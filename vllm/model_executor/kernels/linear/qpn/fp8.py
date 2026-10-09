# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The existing SM70 block-FP8 W8A16 dispatcher in the linear kernel lifecycle."""

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from vllm import _sm70_ops as sm70_ops
from vllm._sm70.policy import NativeBindings
from vllm.config import get_current_vllm_config
from vllm.config.kernel import Sm70Fp8Config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.scaled_mm.ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
    ScaledMMLinearKernel,
)
from vllm.model_executor.kernels.linear.sm70_provider import (
    flatten_linear_input,
    restore_linear_output,
)
from vllm.model_executor.layers.quantization.utils import sm70_layer_workspaces
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_weight_block_strategy,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
)
from vllm.model_executor.models.config import sm70_fp8_serialized_pipeline_qualified
from vllm.model_executor.utils import replace_parameter
from vllm.platforms import current_platform

logger = init_logger(__name__)

_SM70_FP8_PREFILL_DENSE_MIN_M = 3920
_SM70_FP8_EXACT_8K_PREFILL_M = 8000
_SM70_FP8_EXACT_8K_PREFILL_SHAPES = {
    "in_proj_qkvz": (5120, 4096),
    "qkv_proj": (5120, 3584),
}
_SM70_FP8_PREFILL_DENSE_SHAPES = {
    "gate_up_proj": (5120, 8704),
    "down_proj": (4352, 5120),
    "out_proj": (1536, 5120),
    "o_proj": (1536, 5120),
}
_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS = max(
    k * n for k, n in _SM70_FP8_PREFILL_DENSE_SHAPES.values()
)
_SM70_FP8_PREFILL_DENSE_WORKSPACE_BYTES = (
    _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS * torch.float16.itemsize
)
_SM70_FP8_QPN8_CONFIGS = {
    # (K, N, fused gated-SiLU): (split-K, accumulator chains, prefetch codes)
    (4352, 5120, False): (16, 1, False),
    (1536, 5120, False): (16, 1, False),
    (5120, 4096, False): (16, 2, False),
    (5120, 3584, False): (16, 2, False),
    (5120, 8704, False): (16, 1, True),
    (5120, 8704, True): (8, 2, True),
}
_SM70_FP8_QPN8_EXTRA_SHAPES = {
    "in_proj_qkvz": (5120, 4096),
    "qkv_proj": (5120, 3584),
}
_SM70_FP8_QPN8_PP2_TP4_CONFIGS = {
    # Real-weight speed winners remain numerically bounded for every admitted
    # projection. Shared-expert gate/up may use K4096/N1024 only through its
    # separately gated non-fused contract; its fused activation remains unsafe.
    (4096, 1536, False): (32, 2, False),
    (1024, 8192, False): (8, 2, False),
    (2048, 4096, False): (16, 2, False),
    (4096, 1024, False): (32, 2, False),
    (512, 4096, False): (16, 2, False),
}
_SM70_FP8_QPN8_PP2_TP4_SHAPES = {
    # Operator role: accepted (layer TP size, K, N) tuples. Gate/up has an
    # additional exact shared-expert and explicit opt-in check below. The
    # replicated indexer wq_b is excluded by TP size: its long-prefill work
    # may overlap the main wq_b and cannot share one dense fallback workspace.
    "fused_wqa_wkv": {(1, 4096, 1536)},
    "wq_b": {(4, 1024, 8192)},
    "wo_b": {(4, 2048, 4096)},
    "down_proj": {(4, 512, 4096)},
    "gate_up_proj": {(4, 4096, 1024)},
}
_SM70_FP8_QPN8_PP2_TP4_WORKSPACE_ELEMENTS = max(
    k * n for k, n, _ in _SM70_FP8_QPN8_PP2_TP4_CONFIGS
)
_SM70_FP8_QPN8_REQUIRED_OPS = (
    "fp8_qpn8_prepare_sm70",
    "fp8_qpn8_dequantize_sm70_out",
    "fp8_qpn8_prefill_sm70_out",
    "fp8_qpn8_dispatch_sm70_out",
    "fp8_qpn8_gemm_sm70_out",
    "fp8_qpn8_gated_pair_sm70_out",
)
# Layers retain only data_ptr(), so this cache owns each allocation's lifetime.
_sm70_fp8_prefill_dense_workspaces: dict[tuple, torch.Tensor] = {}
_sm70_fp8_qpn8_pp2_tp4_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def clear_sm70_fp8_workspaces() -> None:
    """Release process-global SM70 FP8 prefill and QPN8 workspaces."""
    sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_prefill_dense_workspaces", _sm70_fp8_prefill_dense_workspaces
    ).clear()
    sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_qpn8_pp2_tp4_workspaces", _sm70_fp8_qpn8_pp2_tp4_workspaces
    ).clear()
    sm70_layer_workspaces.clear_layer_workspaces()


def _bind_sm70_fp8_prefill_workspace(
    layer: torch.nn.Module, workspace: torch.Tensor
) -> None:
    sm70_layer_workspaces.register_layer_workspace(layer, workspace, family="fp8")
    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = workspace.data_ptr()


def _resolved_policy(policy: Sm70Fp8Config | None) -> Sm70Fp8Config:
    if policy is None:
        policy = Sm70Fp8Config()
    policy.resolve()
    policy.native.resolve("fp8")
    if policy.native.fp8_grouped_bmm_decode is not None:
        policy.legacy_grouped_bmm_decode = policy.native.fp8_grouped_bmm_decode
    if policy.native.fp8_prefill_fast_selector is not None:
        policy.legacy_prefill_fast_selector = policy.native.fp8_prefill_fast_selector
    return policy


def _is_sm70_fp8_pp2_tp4_shared_gate_layer(layer: torch.nn.Module) -> bool:
    """Match the exact PP2 x TP4 shared-expert gate/up tensor."""
    prefix = str(getattr(layer, "prefix", ""))
    return bool(
        prefix.endswith(".shared_experts.gate_up_proj")
        and int(getattr(layer, "tp_size", 1)) == 4
        and getattr(layer, "weight_block_size", None) == [128, 128]
        and int(getattr(layer, "input_size_per_partition", 0)) == 4096
        and int(getattr(layer, "output_size_per_partition", 0)) == 1024
        and getattr(layer, "output_partition_sizes", None) == [512, 512]
        and tuple(layer.weight.shape) == (1024, 4096)
    )


def _is_sm70_fp8_prescaled_m1_decode_layer(
    layer: torch.nn.Module, policy: Sm70Fp8Config | None = None
) -> bool:
    """Admit the measured WQA or exact shared-gate contract."""
    fused_wqa_wkv = bool(
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] == "fused_wqa_wkv"
        and int(getattr(layer, "tp_size", 1)) == 1
        and getattr(layer, "weight_block_size", None) == [128, 128]
        and int(getattr(layer, "input_size_per_partition", 0)) == 4096
        and int(getattr(layer, "output_size_per_partition", 0)) == 1536
        and tuple(layer.weight.shape) == (1536, 4096)
    )
    return bool(
        fused_wqa_wkv
        or (
            _resolved_policy(policy).prescaled_shared_gate
            and _is_sm70_fp8_pp2_tp4_shared_gate_layer(layer)
        )
    )


def _is_sm70_fp8_prescaled_m1_decode_runtime_contract() -> bool:
    return sm70_fp8_serialized_pipeline_qualified(get_current_vllm_config())


def _try_sm70_fp8_prescaled_decode_scales(
    scales: torch.Tensor,
) -> torch.Tensor | None:
    """Prescale only when the FP16 exponent shift is finite and reversible."""
    if scales.dtype != torch.float16 or not bool(torch.all(scales >= 0).item()):
        return None
    prescaled = scales.mul(256)
    if not bool(torch.isfinite(prescaled).all().item()):
        return None
    if not torch.equal(prescaled.mul(1.0 / 256.0), scales):
        return None
    return prescaled


def _is_sm70_fp8_exact_8k_prefill_layer(layer: torch.nn.Module) -> bool:
    return _sm70_fp8_dense_layout_allowed(layer, _SM70_FP8_EXACT_8K_PREFILL_SHAPES)


def _sm70_fp8_dense_layout_allowed(layer: torch.nn.Module, roles: dict) -> bool:
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return False
    if len(layer.weight.shape) != 2:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    k, n = layer.weight.shape
    return suffix in roles and k > 0 and k % 128 == 0 and n > 0 and n % 128 == 0


def _is_sm70_fp8_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    return _sm70_fp8_dense_layout_allowed(
        layer, _SM70_FP8_PREFILL_DENSE_SHAPES | _SM70_FP8_EXACT_8K_PREFILL_SHAPES
    )


def _sm70_fp8_qpn8_config(k: int, n: int, gated: bool) -> tuple[int, int, bool]:
    return _SM70_FP8_QPN8_CONFIGS.get(
        (k, n, gated), (8 if gated or k % 256 else 16, 2, False)
    )


def _is_sm70_fp8_qpn8_layer(layer: torch.nn.Module) -> bool:
    """Admit the native packed layout independently of tensor parallelism."""
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return False
    if len(layer.weight.shape) != 2:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    if suffix not in _SM70_FP8_PREFILL_DENSE_SHAPES | _SM70_FP8_QPN8_EXTRA_SHAPES:
        return False
    n, k = layer.weight.shape
    return bool(
        k > 0
        and k % 128 == 0
        and n > 0
        and n % 128 == 0
        and (suffix != "gate_up_proj" or n % 64 == 0)
        and getattr(layer, "input_size_per_partition", 0) == k
        and getattr(layer, "output_size_per_partition", 0) == n
    )


def _is_sm70_fp8_qpn8_runtime_contract() -> bool:
    """Keep scheduler capacity out of the QPN8 weight-layout contract.

    The opaque dispatcher selects the measured QPN8 decode kernel from the
    live ``M <= 8`` shape and the exact dense-prefill fallback for larger M.
    Model-level capacity and speculative decoding therefore do not need an
    explicit environment opt-in; layer shape and operator availability remain
    the actual admission gates.
    """
    return True


def _sm70_fp8_qpn8_pp2_tp4_enabled(policy: Sm70Fp8Config | None = None) -> bool:
    return bool(_resolved_policy(policy).qpn8_pp2_tp4)


def _is_sm70_fp8_qpn8_pp2_tp4_runtime_contract() -> bool:
    return sm70_fp8_serialized_pipeline_qualified(get_current_vllm_config())


def _is_sm70_fp8_qpn8_pp2_tp4_shared_gate_contract(
    layer: torch.nn.Module,
) -> bool:
    """Match only the measured non-fused shared-expert gate/up tensor."""
    return _is_sm70_fp8_pp2_tp4_shared_gate_layer(layer)


def _sm70_fp8_qpn8_pp2_tp4_config(
    layer: torch.nn.Module, *, gated_silu: bool, policy: Sm70Fp8Config | None = None
) -> tuple[int, int, bool] | None:
    """Select by operator/tensor contract, never model or checkpoint identity."""
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return None
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    if suffix == "gate_up_proj" and (
        gated_silu
        or not _resolved_policy(policy).qpn8_shared_gate
        or not _is_sm70_fp8_qpn8_pp2_tp4_shared_gate_contract(layer)
    ):
        return None
    accepted = _SM70_FP8_QPN8_PP2_TP4_SHAPES.get(suffix)
    if accepted is None:
        return None
    k_dim = int(getattr(layer, "input_size_per_partition", 0))
    n_dim = int(getattr(layer, "output_size_per_partition", 0))
    layer_contract = (int(getattr(layer, "tp_size", 1)), k_dim, n_dim)
    if layer_contract not in accepted:
        return None
    if tuple(reversed(layer.weight.shape)) != (k_dim, n_dim):
        return None
    if gated_silu:
        output_partitions = getattr(layer, "output_partition_sizes", None)
        if (
            suffix != "gate_up_proj"
            or not isinstance(output_partitions, list)
            or output_partitions != [n_dim // 2, n_dim // 2]
        ):
            return None
    return _SM70_FP8_QPN8_PP2_TP4_CONFIGS.get((k_dim, n_dim, gated_silu))


def _sm70_fp8_qpn8_pp2_tp4_bmm_config(
    layer: torch.nn.Module,
) -> tuple[int, int, bool] | None:
    if (
        getattr(layer, "prefix", "").rsplit(".", 1)[-1] != "wo_a"
        or getattr(layer, "weight_block_size", None) != [128, 128]
        or int(getattr(layer, "tp_size", 1)) != 4
        or int(getattr(layer, "bmm_batch_size", 0)) != 2
        or int(getattr(layer, "input_size_per_partition", 0)) != 4096
        or int(getattr(layer, "output_size_per_partition", 0)) != 2048
        or tuple(layer.weight.shape) != (2048, 4096)
    ):
        return None
    return _SM70_FP8_QPN8_PP2_TP4_CONFIGS[(4096, 1024, False)]


def _missing_sm70_fp8_qpn8_ops() -> list[str]:
    return [
        name for name in _SM70_FP8_QPN8_REQUIRED_OPS if not hasattr(torch.ops._C, name)
    ]


def _get_sm70_fp8_prefill_exact_dense_workspace(
    weight: torch.Tensor,
) -> torch.Tensor | None:
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    elements = max(_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS, weight.numel())
    # Never replace a live allocation: prepared layers retain its raw pointer.
    cache_key = (
        (device_index, torch.float16)
        if elements == _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS
        else (device_index, torch.float16, elements)
    )
    workspace = sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_prefill_dense_workspaces", _sm70_fp8_prefill_dense_workspaces
    ).get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (elements,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        logger.warning_once(
            "Insufficient memory for the bounded SM70 FP8 prefill workspace; "
            "falling back to TurboMind FP8."
        )
        return None
    sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_prefill_dense_workspaces", _sm70_fp8_prefill_dense_workspaces
    )[cache_key] = workspace
    return workspace


def _get_sm70_fp8_qpn8_pp2_tp4_workspace(
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Allocate one bounded FP16 prefill fallback per device."""
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, torch.float16)
    workspace = sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_qpn8_pp2_tp4_workspaces", _sm70_fp8_qpn8_pp2_tp4_workspaces
    ).get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (_SM70_FP8_QPN8_PP2_TP4_WORKSPACE_ELEMENTS,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        logger.warning_once(
            "Insufficient memory for the bounded SM70 PP2 x TP4 QPN8 "
            "prefill workspace; retaining TurboMind FP8."
        )
        return None
    sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_qpn8_pp2_tp4_workspaces", _sm70_fp8_qpn8_pp2_tp4_workspaces
    )[cache_key] = workspace
    return workspace


def _sm70_fp8_prefill_visible_dense_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    dense_weight_ptr: int | None,
    *,
    gated_silu: bool,
    min_prefill_m: int,
    policy: Sm70Fp8Config | None = None,
) -> torch.Tensor | None:
    """Expose the long-prefill dense MM to AsyncTP pattern matching.

    This diagnostic route intentionally keeps the accepted dequantization and
    FP16 MM arithmetic while moving ``aten.mm`` out of the opaque C++ wrapper.
    """
    if not _resolved_policy(policy).prefill_visible_dense_mm:
        return None
    if dense_weight_ptr is None:
        return None
    if input.dtype != torch.float16 or input.shape[0] < min_prefill_m:
        return None

    device_index = input.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    elements = max(_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS, weight.numel())
    cache_key = (
        (device_index, torch.float16)
        if elements == _SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS
        else (device_index, torch.float16, elements)
    )
    workspace = sm70_layer_workspaces.workspace_pool(
        "_sm70_fp8_prefill_dense_workspaces", _sm70_fp8_prefill_dense_workspaces
    ).get(cache_key)
    if workspace is None:
        return None
    if not torch.compiler.is_compiling() and workspace.data_ptr() != dense_weight_ptr:
        return None

    dense_weight = workspace.narrow(0, 0, weight.numel()).view(weight.shape)
    sm70_ops.fp8_sm70_dequantize_out(dense_weight, weight, scales, 128)
    dense_out = torch.mm(input, dense_weight)
    if not gated_silu:
        return dense_out

    out = torch.empty(
        (input.shape[0], dense_out.shape[1] // 2),
        dtype=input.dtype,
        device=input.device,
    )
    sm70_ops.silu_and_mul_interleaved(out, dense_out)
    return out


@dataclass
class Sm70Fp8LinearLayerConfig(FP8ScaledMMLinearLayerConfig):
    """Serialized block-FP8 loader contract, separate from activation-FP8 layouts."""

    is_scale_e8m0: bool = False
    is_bmm: bool = False
    policy: Sm70Fp8Config = field(default_factory=Sm70Fp8Config)


class TurboMindFp8LinearKernel(FP8ScaledMMLinearKernel):
    """Weight-only FP8, with the existing QPN8 and prefill variants retained."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA"
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = capability.to_int() if capability is not None else None
        if compute_capability not in (70, 72):
            return False, "requires Volta SM70 or SM72"
        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        if not isinstance(c, Sm70Fp8LinearLayerConfig):
            return False, "requires the serialized SM70 weight-only FP8 layout"
        if c.weight_quant_key != kFp8Static128BlockSym:
            return False, "requires 128 x 128 block weight scales"
        if c.input_dtype != torch.float16:
            return False, "requires float16 activations"
        if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
            return False, "native fp8_sm70_prepare operator is unavailable"
        return True, None

    def __init__(
        self, c: Sm70Fp8LinearLayerConfig, layer_param_names: Sequence[str]
    ) -> None:
        # W8A16 never quantizes its activations. The common lifecycle still
        # validates support and layout, without constructing an unused QuantFP8.
        ScaledMMLinearKernel.__init__(self, c, layer_param_names)
        self.policy = _resolved_policy(c.policy)
        self.native_ops = NativeBindings(self.policy.native.values)
        self.is_scale_e8m0 = c.is_scale_e8m0
        self.weight_block_size = [128, 128]

    def apply_scaled_mm(self, **kwargs):
        raise RuntimeError(
            "SM70 W8A16 consumes unquantized activations via apply_weights"
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        weight_scale_inv = layer.weight_scale_inv
        assert self.weight_block_size is not None
        if layer.orig_dtype != torch.float16:
            raise RuntimeError(
                "SM70 TurboMind FP8 dense path currently requires fp16 "
                f"original weights, got {layer.orig_dtype}."
            )
        if not hasattr(torch.ops._C, "fp8_sm70_prepare"):
            raise RuntimeError(
                "VLLM_SM70_FP8_TURBOMIND=1 requires a build with CUDA "
                "arch 7.0 and the SM70 TurboMind extension."
            )

        weight, weight_scale_inv = process_fp8_weight_block_strategy(
            weight, weight_scale_inv
        )
        if weight_scale_inv.dtype != torch.float32:
            weight_scale_inv = weight_scale_inv.to(torch.float32)
        if getattr(layer, "is_bmm", False):
            qpn8_bmm_config = (
                _sm70_fp8_qpn8_pp2_tp4_bmm_config(layer)
                if _sm70_fp8_qpn8_pp2_tp4_enabled(self.policy)
                else None
            )
            qpn8_bmm_runtime = bool(
                qpn8_bmm_config is not None
                and _is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
            )
            group_count = int(getattr(layer, "bmm_batch_size", 0))
            if group_count <= 0 or weight.shape[0] % group_count != 0:
                raise RuntimeError(
                    "SM70 TurboMind grouped FP8 requires a positive group "
                    f"count dividing weight rows, got groups={group_count}, "
                    f"weight={tuple(weight.shape)}."
                )
            rows_per_group = weight.shape[0] // group_count
            if rows_per_group % self.weight_block_size[0] != 0:
                raise RuntimeError(
                    "SM70 TurboMind grouped FP8 requires each group output "
                    f"to align to block_n={self.weight_block_size[0]}, got "
                    f"{rows_per_group}."
                )
            scale_rows_per_group = rows_per_group // self.weight_block_size[0]
            expected_scale_rows = scale_rows_per_group * group_count
            if weight_scale_inv.shape[0] != expected_scale_rows:
                raise RuntimeError(
                    "SM70 TurboMind grouped FP8 scale rows do not match "
                    f"weights: expected {expected_scale_rows}, got "
                    f"{weight_scale_inv.shape[0]}."
                )

            if qpn8_bmm_config is not None and not qpn8_bmm_runtime:
                logger.info_once(
                    "Grouped SM70 QPN8 retains TurboMind outside the "
                    "serialized PP2 x TP4 single-request contract."
                )
            if qpn8_bmm_runtime:
                missing_ops = _missing_sm70_fp8_qpn8_ops()
                explicitly_enabled = (
                    "qpn8" in self.policy.explicit_enables
                    or "qpn8_pp2_tp4" in self.policy.explicit_enables
                )
                if missing_ops and explicitly_enabled:
                    raise RuntimeError(
                        "The explicitly enabled SM70 PP2 x TP4 QPN8 route "
                        f"requires source-built operators; missing: {missing_ops}."
                    )
                if missing_ops:
                    logger.warning_once(
                        "The requested SM70 PP2 x TP4 QPN8 route is unavailable "
                        "in the loaded vllm._C; retaining TurboMind FP8."
                    )
                workspace = (
                    None
                    if missing_ops
                    else _get_sm70_fp8_qpn8_pp2_tp4_workspace(weight)
                )
                if workspace is not None:
                    qpn8_weights = []
                    qpn8_scales = []
                    for group_idx in range(group_count):
                        row_start = group_idx * rows_per_group
                        scale_start = group_idx * scale_rows_per_group
                        qpn8_weight, qpn8_scale = self.native_ops.fp8_qpn8_prepare_sm70(
                            weight[row_start : row_start + rows_per_group].contiguous(),
                            weight_scale_inv[
                                scale_start : scale_start + scale_rows_per_group
                            ].contiguous(),
                        )
                        qpn8_weights.append(qpn8_weight)
                        qpn8_scales.append(qpn8_scale)
                    replace_parameter(layer, "weight", torch.stack(qpn8_weights))
                    replace_parameter(
                        layer, "weight_scale_inv", torch.stack(qpn8_scales)
                    )
                    assert qpn8_bmm_config is not None
                    split_k, nacc, prefetch = qpn8_bmm_config
                    layer.input_scale = None
                    layer.sm70_fp8_turbomind = True
                    layer.sm70_fp8_qpn8 = True
                    layer.sm70_fp8_qpn8_bmm = True
                    layer.sm70_fp8_bmm = True
                    layer.sm70_fp8_bmm_groups = group_count
                    layer.sm70_fp8_bmm_output_size = rows_per_group
                    layer.sm70_fp8_qpn8_split_k = split_k
                    layer.sm70_fp8_qpn8_nacc = nacc
                    layer.sm70_fp8_qpn8_prefetch = prefetch
                    _bind_sm70_fp8_prefill_workspace(layer, workspace)
                    logger.info_once(
                        "Default SM70 grouped QPN8 enabled for the validated "
                        "serialized PP2 x TP4 tensor contract."
                    )
                    return

            prepared_weights = []
            prepared_scales = []
            metas = []
            for group_idx in range(group_count):
                row_start = group_idx * rows_per_group
                scale_start = group_idx * scale_rows_per_group
                tm_weight, tm_scale, meta = self.native_ops.fp8_sm70_prepare(
                    weight[row_start : row_start + rows_per_group].contiguous(),
                    weight_scale_inv[
                        scale_start : scale_start + scale_rows_per_group
                    ].contiguous(),
                    self.weight_block_size[0],
                    False,
                )
                prepared_weights.append(tm_weight)
                prepared_scales.append(tm_scale)
                metas.append(meta)

            first_meta = metas[0]
            if any(
                int(meta[0].item()) != int(first_meta[0].item())
                or int(meta[1].item()) != int(first_meta[1].item())
                for meta in metas[1:]
            ):
                raise RuntimeError(
                    "SM70 TurboMind grouped FP8 produced inconsistent layouts."
                )
            replace_parameter(layer, "weight", torch.stack(prepared_weights))
            replace_parameter(layer, "weight_scale_inv", torch.stack(prepared_scales))
            layer.input_scale = None
            layer.sm70_fp8_turbomind = True
            layer.sm70_fp8_bmm = True
            layer.sm70_fp8_bmm_groups = group_count
            layer.sm70_fp8_bmm_output_size = rows_per_group
            layer.register_buffer("sm70_fp8_meta", first_meta, persistent=False)
            layer.sm70_fp8_k_ld = int(first_meta[0].item())
            layer.sm70_fp8_q_ld = int(first_meta[1].item())
            if (
                self.policy.legacy_grouped_bmm_decode
                and group_count == 2
                and rows_per_group == 1024
                and int(layer.weight.shape[1]) == 4096
                and hasattr(
                    torch.ops._C,
                    "fp8_moe_gemm_sm70_per_expert_dispatch_out",
                )
                and hasattr(torch.ops._C, "awq_moe_build_strided_ptrs")
            ):
                ptrs_w, ptrs_s = self.native_ops.awq_moe_build_strided_ptrs(
                    layer.weight,
                    layer.weight_scale_inv,
                    layer.sm70_fp8_k_ld,
                    layer.sm70_fp8_q_ld,
                    group_count,
                )
                layer.register_buffer(
                    "sm70_fp8_bmm_grouped_ptrs_w", ptrs_w, persistent=False
                )
                layer.register_buffer(
                    "sm70_fp8_bmm_grouped_ptrs_s", ptrs_s, persistent=False
                )
                layer.register_buffer(
                    "sm70_fp8_bmm_grouped_offsets",
                    torch.arange(
                        group_count + 1,
                        dtype=torch.int32,
                        device=layer.weight.device,
                    ),
                    persistent=False,
                )
                layer.sm70_fp8_bmm_grouped_decode = True
                logger.info_once("SM70 FP8 one-launch grouped-BMM decode path enabled.")
            logger.info_once(
                "SM70 FP8 TurboMind grouped-BMM path enabled for DeepSeek V4."
            )
            return
        is_gated_silu_layer = self._is_sm70_gated_silu_layer(layer)
        use_gated_silu = is_gated_silu_layer and bool(self.policy.gated_silu)
        generic_qpn8_candidate = self.policy.qpn8 and _is_sm70_fp8_qpn8_layer(layer)
        pp2_tp4_qpn8_config = (
            _sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False, policy=self.policy)
            if _sm70_fp8_qpn8_pp2_tp4_enabled(self.policy)
            else None
        )
        pp2_tp4_qpn8_candidate = pp2_tp4_qpn8_config is not None
        qpn8_candidate_layer = generic_qpn8_candidate or pp2_tp4_qpn8_candidate
        qpn8_runtime = bool(
            (pp2_tp4_qpn8_candidate and _is_sm70_fp8_qpn8_pp2_tp4_runtime_contract())
            or (generic_qpn8_candidate and _is_sm70_fp8_qpn8_runtime_contract())
        )
        nonfused_shared_gate = bool(
            pp2_tp4_qpn8_candidate
            and qpn8_runtime
            and _is_sm70_fp8_qpn8_pp2_tp4_shared_gate_contract(layer)
        )
        if nonfused_shared_gate:
            use_gated_silu = False
        if qpn8_candidate_layer and not qpn8_runtime:
            logger.info_once(
                "The SM70 FP8 QPN8 route retains TurboMind unless its "
                "bounded-concurrency runtime contract is explicit."
            )
        if qpn8_candidate_layer and qpn8_runtime:
            pp2_tp4_gated_config = None
            if pp2_tp4_qpn8_candidate and use_gated_silu:
                pp2_tp4_gated_config = _sm70_fp8_qpn8_pp2_tp4_config(
                    layer, gated_silu=True, policy=self.policy
                )
                if pp2_tp4_gated_config is None:
                    raise RuntimeError(
                        "The SM70 PP2 x TP4 QPN8 gate/up layer violated "
                        "its fused-SiLU tensor contract."
                    )
            missing_ops = _missing_sm70_fp8_qpn8_ops()
            if missing_ops:
                explicitly_enabled = (
                    "qpn8" in self.policy.explicit_enables
                    or "qpn8_pp2_tp4" in self.policy.explicit_enables
                    or "qpn8_shared_gate" in self.policy.explicit_enables
                )
                if explicitly_enabled:
                    raise RuntimeError(
                        "The explicitly enabled SM70 QPN8 route requires "
                        f"source-built operators; missing: {missing_ops}."
                    )
                logger.warning_once(
                    "The requested SM70 FP8 QPN8 route is unavailable in "
                    "the loaded vllm._C; retaining the TurboMind layout."
                )

            if missing_ops:
                workspace = None
            elif pp2_tp4_qpn8_candidate:
                workspace = _get_sm70_fp8_qpn8_pp2_tp4_workspace(weight)
            else:
                workspace = _get_sm70_fp8_prefill_exact_dense_workspace(weight)
            if not missing_ops and workspace is not None:
                qpn8_codes, qpn8_scales = self.native_ops.fp8_qpn8_prepare_sm70(
                    weight, weight_scale_inv
                )
                k_dim, n_dim = (int(dim) for dim in qpn8_codes.shape)
                if pp2_tp4_qpn8_candidate:
                    assert pp2_tp4_qpn8_config is not None
                    split_k, nacc, prefetch = pp2_tp4_qpn8_config
                else:
                    split_k, nacc, prefetch = _sm70_fp8_qpn8_config(k_dim, n_dim, False)
                replace_parameter(layer, "weight", qpn8_codes)
                replace_parameter(layer, "weight_scale_inv", qpn8_scales)
                layer.input_scale = None
                layer.sm70_fp8_turbomind = True
                layer.sm70_fp8_qpn8 = True
                layer.sm70_fp8_qpn8_split_k = split_k
                layer.sm70_fp8_qpn8_nacc = nacc
                layer.sm70_fp8_qpn8_prefetch = prefetch
                _bind_sm70_fp8_prefill_workspace(layer, workspace)
                if use_gated_silu:
                    if pp2_tp4_qpn8_candidate:
                        assert pp2_tp4_gated_config is not None
                        gated_split_k, gated_nacc, gated_prefetch = pp2_tp4_gated_config
                    else:
                        gated_split_k, gated_nacc, gated_prefetch = (
                            _sm70_fp8_qpn8_config(k_dim, n_dim, True)
                        )
                    layer.sm70_fp8_gated_silu = True
                    layer.sm70_fp8_gated_silu_primary = True
                    layer.sm70_fp8_qpn8_gated_split_k = gated_split_k
                    layer.sm70_fp8_qpn8_gated_nacc = gated_nacc
                    layer.sm70_fp8_qpn8_gated_prefetch = gated_prefetch
                if pp2_tp4_qpn8_candidate:
                    logger.info_once(
                        "Default SM70 QPN8 enabled for the validated "
                        "serialized PP2 x TP4 operator contract."
                    )
                    if nonfused_shared_gate:
                        logger.info_once(
                            "Default SM70 QPN8 shared-expert gate/up route "
                            "enabled with its external activation retained."
                        )
                else:
                    logger.info_once(
                        "Memory-neutral SM70 FP8 QPN8 path enabled for "
                        "accepted TP4 block-FP8 operator shapes."
                    )
                return
            if not missing_ops:
                logger.warning_once(
                    "Insufficient memory for the SM70 FP8 QPN8 prefill "
                    "workspace; retaining the TurboMind layout."
                )

        prescaled_decode_requested = self.policy.prescaled_decode
        prescaled_shared_gate_layer = _is_sm70_fp8_pp2_tp4_shared_gate_layer(layer)
        prescaled_decode_explicit = bool(
            "prescaled_decode" in self.policy.explicit_enables
            or (
                prescaled_shared_gate_layer
                and "prescaled_shared_gate" in self.policy.explicit_enables
            )
        )
        prescaled_decode_layer = _is_sm70_fp8_prescaled_m1_decode_layer(
            layer, self.policy
        )
        prescaled_decode_runtime = bool(
            prescaled_decode_layer
            and _is_sm70_fp8_prescaled_m1_decode_runtime_contract()
        )
        if (
            prescaled_decode_explicit
            and prescaled_decode_layer
            and not (self.is_scale_e8m0 and prescaled_decode_runtime)
        ):
            raise RuntimeError(
                "Explicit SM70 FP8 prescaled decode requires UE8M0 128x128 "
                "scales and the PP2 x TP4 no-spec single-request contract."
            )
        use_prescaled_m1_decode = bool(
            prescaled_decode_requested
            and self.is_scale_e8m0
            and prescaled_decode_runtime
        )
        tm_weight, tm_scales, meta = self.native_ops.fp8_sm70_prepare(
            weight,
            weight_scale_inv,
            self.weight_block_size[0],
            use_gated_silu,
        )
        if is_gated_silu_layer and not use_gated_silu:
            logger.info_once(
                "SM70 FP8 dense gated-SiLU layout disabled; skipping "
                "the extra gate_up_proj TurboMind copy. Set "
                "VLLM_SM70_FP8_DENSE_GATED_SILU=1 to enable it."
            )
        if use_gated_silu:
            layer.sm70_fp8_gated_silu = True
            layer.sm70_fp8_gated_silu_primary = True
            layer.sm70_fp8_gated_silu_k_ld = int(meta[0].item())
            layer.sm70_fp8_gated_silu_q_ld = int(meta[1].item())
            logger.info_once("SM70 FP8 dense gated-SiLU single-layout path enabled.")
        replace_parameter(layer, "weight", tm_weight)
        replace_parameter(layer, "weight_scale_inv", tm_scales)
        layer.input_scale = None
        layer.sm70_fp8_turbomind = True
        layer.register_buffer("sm70_fp8_meta", meta, persistent=False)
        layer.sm70_fp8_k_ld = int(meta[0].item())
        layer.sm70_fp8_q_ld = int(meta[1].item())
        if use_prescaled_m1_decode:
            has_prescaled_op = hasattr(torch.ops._C, "fp8_gemm_sm70_prescaled_m1_out")
            prescaled_scales = (
                _try_sm70_fp8_prescaled_decode_scales(tm_scales)
                if has_prescaled_op
                else None
            )
            if prescaled_scales is None and prescaled_decode_explicit:
                raise RuntimeError(
                    "Explicit SM70 FP8 prescaled decode requires the "
                    "source-built operator and finite, reversible FP16 "
                    "scales after the 256x exponent shift."
                )
            if prescaled_scales is None:
                logger.warning_once(
                    "SM70 FP8 prescaled decode is unavailable for the "
                    "loaded extension or scale range; retaining the "
                    "ordinary TurboMind transform."
                )
            else:
                layer.register_buffer(
                    "sm70_fp8_decode_prescaled_scales",
                    prescaled_scales,
                    persistent=False,
                )
                if prescaled_shared_gate_layer:
                    logger.info_once(
                        "Exact SM70 shared-expert gate/up prescaled "
                        "decode path enabled with external activation retained."
                    )
                else:
                    logger.info_once(
                        "Exact SM70 fused-WQA/WKV prescaled decode path enabled."
                    )
        if (
            self.policy.legacy_prefill_fast_selector
            and self.policy.prefill_prescaled
            and hasattr(torch.ops._C, "fp8_gemm_sm70_prefill_prescaled_out")
            and _is_sm70_fp8_exact_8k_prefill_layer(layer)
        ):
            layer.register_buffer(
                "sm70_fp8_prefill_prescaled_scales",
                tm_scales.mul(256),
                persistent=False,
            )
            logger.info_once(
                "SM70 block-FP8 exact-8K pre-scaled projection path enabled."
            )
        if (
            self.policy.prefill_exact_dense
            and hasattr(torch.ops._C, "fp8_gemm_sm70_prefill_dispatch_out")
            and _is_sm70_fp8_prefill_exact_dense_layer(layer)
        ):
            is_exact_8k_projection = _is_sm70_fp8_exact_8k_prefill_layer(layer)
            workspace = _get_sm70_fp8_prefill_exact_dense_workspace(tm_weight)
            if workspace is not None:
                _bind_sm70_fp8_prefill_workspace(layer, workspace)
                layer.sm70_fp8_prefill_exact_dense_min_m = (
                    _SM70_FP8_EXACT_8K_PREFILL_M
                    if is_exact_8k_projection
                    else _SM70_FP8_PREFILL_DENSE_MIN_M
                )
                logger.info_once(
                    "SM70 FP8 exact-dense prefill path enabled with a bounded "
                    "85 MiB workspace."
                )
        logger.info_once("SM70 FP8 TurboMind W8A16 dense path enabled.")
        return

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if getattr(layer, "sm70_fp8_qpn8", False):
            if x.dtype != torch.float16:
                raise RuntimeError(
                    "SM70 FP8 QPN8 currently requires float16 activations, "
                    f"got {x.dtype}."
                )
            if getattr(layer, "sm70_fp8_qpn8_bmm", False):
                group_count = int(layer.sm70_fp8_bmm_groups)
                output_size = int(layer.sm70_fp8_bmm_output_size)
                if x.ndim < 2 or x.shape[-2] != group_count:
                    raise RuntimeError(
                        "SM70 grouped QPN8 input must end in [groups, K], got "
                        f"{tuple(x.shape)} for groups={group_count}."
                    )
                x_grouped = x.reshape(-1, group_count, x.shape[-1])
                x_by_group = x_grouped.transpose(0, 1).contiguous()
                out_by_group = torch.empty(
                    (group_count, x_grouped.shape[0], output_size),
                    device=x.device,
                    dtype=x.dtype,
                )
                if x_grouped.shape[0] == 0:
                    return out_by_group.transpose(0, 1).reshape(
                        *x.shape[:-2], group_count, output_size
                    )
                for group_idx in range(group_count):
                    torch.ops.vllm.sm70_fp8_qpn8_dispatch(
                        out_by_group[group_idx],
                        layer.prefix,
                        x_by_group[group_idx],
                        layer.weight[group_idx],
                        layer.weight_scale_inv[group_idx],
                        int(layer.sm70_fp8_qpn8_split_k),
                        int(layer.sm70_fp8_qpn8_nacc),
                        bool(layer.sm70_fp8_qpn8_prefetch),
                        False,
                    )
                out = out_by_group.transpose(0, 1).reshape(
                    *x.shape[:-2], group_count, output_size
                )
                if bias is not None:
                    out.add_(bias.view(group_count, output_size))
                return out

            x_2d = flatten_linear_input(x)
            if x_2d.stride(-1) != 1:
                x_2d = x_2d.contiguous()
            out_2d = torch.empty(
                (x_2d.shape[0], layer.output_size_per_partition),
                device=x.device,
                dtype=x.dtype,
            )
            if x_2d.shape[0] == 0:
                return restore_linear_output(out_2d, x, layer.output_size_per_partition)
            torch.ops.vllm.sm70_fp8_qpn8_dispatch(
                out_2d,
                layer.prefix,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                int(layer.sm70_fp8_qpn8_split_k),
                int(layer.sm70_fp8_qpn8_nacc),
                bool(layer.sm70_fp8_qpn8_prefetch),
                False,
            )
            out = restore_linear_output(out_2d, x, layer.output_size_per_partition)
            if bias is not None:
                out.add_(bias)
            return out

        if getattr(layer, "sm70_fp8_bmm", False):
            group_count = int(layer.sm70_fp8_bmm_groups)
            output_size = int(layer.sm70_fp8_bmm_output_size)
            if x.ndim < 2 or x.shape[-2] != group_count:
                raise RuntimeError(
                    "SM70 grouped FP8 input must end in [groups, K], got "
                    f"{tuple(x.shape)} for groups={group_count}."
                )
            x_grouped = x.reshape(-1, group_count, x.shape[-1])
            x_by_group = x_grouped.transpose(0, 1).contiguous()
            out_by_group = torch.empty(
                (group_count, x_grouped.shape[0], output_size),
                device=x.device,
                dtype=x.dtype,
            )
            if (
                getattr(layer, "sm70_fp8_bmm_grouped_decode", False)
                and x_grouped.shape[0] == 1
            ):
                self.native_ops.fp8_moe_gemm_sm70_per_expert_dispatch_out(
                    out_by_group.reshape(group_count, output_size),
                    x_by_group.reshape(group_count, x.shape[-1]),
                    layer.sm70_fp8_bmm_grouped_offsets,
                    layer.sm70_fp8_bmm_grouped_ptrs_w,
                    layer.sm70_fp8_bmm_grouped_ptrs_s,
                    group_count,
                    x.shape[-1],
                    output_size,
                    128,
                    False,
                )
            else:
                for group_idx in range(group_count):
                    self.native_ops.fp8_gemm_sm70_out(
                        out_by_group[group_idx],
                        x_by_group[group_idx],
                        layer.weight[group_idx],
                        layer.weight_scale_inv[group_idx],
                        128,
                        layer.sm70_fp8_k_ld,
                        layer.sm70_fp8_q_ld,
                        False,
                    )
            out = out_by_group.transpose(0, 1).reshape(
                *x.shape[:-2], group_count, output_size
            )
            if bias is not None:
                out.add_(bias.view(group_count, output_size))
            return out

        x_2d = flatten_linear_input(x)
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        out_2d = torch.empty(
            (x_2d.shape[0], layer.output_size_per_partition),
            device=x.device,
            dtype=x.dtype,
        )
        prefill_workspace_ptr = getattr(
            layer, "sm70_fp8_prefill_exact_dense_workspace_ptr", None
        )
        prefill_prescaled_scales = getattr(
            layer, "sm70_fp8_prefill_prescaled_scales", None
        )
        decode_prescaled_scales = getattr(
            layer, "sm70_fp8_decode_prescaled_scales", None
        )
        prefill_min_m = getattr(
            layer,
            "sm70_fp8_prefill_exact_dense_min_m",
            _SM70_FP8_PREFILL_DENSE_MIN_M,
        )
        visible_dense_out = _sm70_fp8_prefill_visible_dense_mm(
            x_2d,
            layer.weight,
            layer.weight_scale_inv,
            prefill_workspace_ptr,
            gated_silu=False,
            min_prefill_m=prefill_min_m,
            policy=self.policy,
        )
        if (
            decode_prescaled_scales is not None
            and self.policy.prescaled_decode
            and x_2d.shape[0] == 1
        ):
            self.native_ops.fp8_gemm_sm70_prescaled_m1_out(
                out_2d,
                x_2d,
                layer.weight,
                decode_prescaled_scales,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
            )
        elif visible_dense_out is not None:
            out_2d = visible_dense_out
        elif prefill_workspace_ptr is not None and x_2d.dtype == torch.float16:
            torch.ops.vllm.sm70_fp8_prefill_dispatch(
                out_2d,
                layer.prefix,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
                prefill_min_m,
            )
        elif (
            prefill_prescaled_scales is not None
            and self.policy.legacy_prefill_fast_selector
            and self.policy.prefill_prescaled
            and x_2d.shape[0] == _SM70_FP8_EXACT_8K_PREFILL_M
        ):
            self.native_ops.fp8_gemm_sm70_prefill_prescaled_out(
                out_2d,
                x_2d,
                layer.weight,
                prefill_prescaled_scales,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
            )
        else:
            self.native_ops.fp8_gemm_sm70_out(
                out_2d,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                128,
                layer.sm70_fp8_k_ld,
                layer.sm70_fp8_q_ld,
                False,
            )
        if getattr(layer, "sm70_fp8_gated_silu_primary", False):
            out_features = layer.output_size_per_partition // 2
            out_2d = (
                out_2d.reshape(x_2d.shape[0], out_features, 2)
                .transpose(1, 2)
                .reshape(x_2d.shape[0], layer.output_size_per_partition)
            )
        out = restore_linear_output(out_2d, x, layer.output_size_per_partition)
        if bias is not None:
            out.add_(bias)
        return out

    def apply_fused_silu_and_mul(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
    ) -> torch.Tensor | None:
        if getattr(layer, "sm70_fp8_qpn8", False):
            if not getattr(layer, "sm70_fp8_gated_silu", False):
                return None
            if x.dtype != torch.float16:
                raise RuntimeError(
                    "SM70 FP8 QPN8 gated-SiLU requires float16 activations, "
                    f"got {x.dtype}."
                )
            x_2d = x.reshape(-1, x.shape[-1])
            if x_2d.stride(-1) != 1:
                x_2d = x_2d.contiguous()
            out_features = layer.output_size_per_partition // 2
            out_2d = torch.empty(
                (x_2d.shape[0], out_features), device=x.device, dtype=x.dtype
            )
            if x_2d.shape[0] == 0:
                return out_2d.reshape(*x.shape[:-1], out_features)
            torch.ops.vllm.sm70_fp8_qpn8_dispatch(
                out_2d,
                layer.prefix,
                x_2d,
                layer.weight,
                layer.weight_scale_inv,
                int(layer.sm70_fp8_qpn8_gated_split_k),
                int(layer.sm70_fp8_qpn8_gated_nacc),
                bool(layer.sm70_fp8_qpn8_gated_prefetch),
                True,
            )
            return out_2d.reshape(*x.shape[:-1], out_features)

        if not getattr(layer, "sm70_fp8_gated_silu", False):
            return None
        if not getattr(layer, "sm70_fp8_turbomind", False):
            return None

        x_2d = x.reshape(-1, x.shape[-1])
        if x_2d.stride(-1) != 1:
            x_2d = x_2d.contiguous()
        out_features = layer.output_size_per_partition // 2
        out_2d = torch.empty(
            (x_2d.shape[0], out_features),
            device=x.device,
            dtype=x.dtype,
        )
        if getattr(layer, "sm70_fp8_gated_silu_primary", False):
            weight = layer.weight
            scales = layer.weight_scale_inv
            k_ld = int(layer.sm70_fp8_k_ld)
            q_ld = int(layer.sm70_fp8_q_ld)
        else:
            weight = layer.sm70_fp8_gated_silu_weight
            scales = layer.sm70_fp8_gated_silu_scales
            k_ld = int(layer.sm70_fp8_gated_silu_k_ld)
            q_ld = int(layer.sm70_fp8_gated_silu_q_ld)
        prefill_workspace_ptr = getattr(
            layer, "sm70_fp8_prefill_exact_dense_workspace_ptr", None
        )
        if prefill_workspace_ptr is not None and x_2d.dtype == torch.float16:
            visible_dense_out = _sm70_fp8_prefill_visible_dense_mm(
                x_2d,
                weight,
                scales,
                prefill_workspace_ptr,
                gated_silu=True,
                min_prefill_m=getattr(
                    layer,
                    "sm70_fp8_prefill_exact_dense_min_m",
                    _SM70_FP8_PREFILL_DENSE_MIN_M,
                ),
                policy=self.policy,
            )
            if visible_dense_out is not None:
                return visible_dense_out.reshape(*x.shape[:-1], out_features)
            min_prefill_m = getattr(
                layer,
                "sm70_fp8_prefill_exact_dense_min_m",
                _SM70_FP8_PREFILL_DENSE_MIN_M,
            )
            torch.ops.vllm.sm70_fp8_prefill_dispatch(
                out_2d,
                layer.prefix,
                x_2d,
                weight,
                scales,
                128,
                k_ld,
                q_ld,
                True,
                min_prefill_m,
            )
            return out_2d.reshape(*x.shape[:-1], out_features)
        self.native_ops.fp8_gemm_sm70_out(
            out_2d,
            x_2d,
            weight,
            scales,
            128,
            k_ld,
            q_ld,
            True,
        )
        return out_2d.reshape(*x.shape[:-1], out_features)

    @staticmethod
    def _is_sm70_gated_silu_layer(layer: torch.nn.Module) -> bool:
        prefix = getattr(layer, "prefix", "")
        if prefix.rsplit(".", 1)[-1] != "gate_up_proj":
            return False
        output_partition_sizes = getattr(layer, "output_partition_sizes", None)
        return (
            isinstance(output_partition_sizes, list)
            and len(output_partition_sizes) == 2
            and output_partition_sizes[0] == output_partition_sizes[1]
        )
