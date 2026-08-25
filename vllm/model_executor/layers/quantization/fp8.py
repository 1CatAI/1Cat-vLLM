# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import TYPE_CHECKING, Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm import _sm70_ops as sm70_ops
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.kernels.linear.scaled_mm import (
    CutlassFP8ScaledMMLinearKernel,
    MarlinFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    RoutedExperts,
    SharedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    Fp8MoeBackend,
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import QuantizationMethods
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_block_strategy,
    process_fp8_weight_tensor_strategy,
    process_fp8_weight_tensor_strategy_moe,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    is_layer_skipped,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
    cutlass_fp8_supported,
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.model_loader.reload.layerwise import (
    initialize_online_processing,
)
from vllm.model_executor.parameter import (
    BlockQuantScaleParameter,
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    is_deep_gemm_supported,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

ACTIVATION_SCHEMES = ["static", "dynamic"]

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
    # Measured B1 winners: (K, N, fused gated-SiLU) maps to
    # (split-K, accumulator chains, prefetch codes).
    (4096, 1536, False): (32, 2, False),
    (1024, 8192, False): (8, 2, False),
    (2048, 4096, False): (16, 2, False),
    (4096, 1024, False): (32, 2, False),
    (4096, 1024, True): (16, 2, False),
    (512, 4096, False): (16, 2, False),
}
_SM70_FP8_QPN8_PP2_TP4_SHAPES = {
    # Operator role: accepted (layer TP size, K, N) tuples. The replicated
    # indexer wq_b is deliberately excluded by TP size: its long-prefill work
    # may overlap the main wq_b and cannot share one dense fallback workspace.
    "fused_wqa_wkv": {(1, 4096, 1536)},
    "wq_b": {(4, 1024, 8192)},
    "wo_b": {(4, 2048, 4096)},
    "gate_up_proj": {(4, 4096, 1024)},
    "down_proj": {(4, 512, 4096)},
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
_SM70_FP8_QPN8_MAX_NUM_SEQS = 8
# Layers retain only data_ptr(), so this cache owns each allocation's lifetime.
_sm70_fp8_prefill_dense_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}
_sm70_fp8_qpn8_pp2_tp4_workspaces: dict[tuple[int, torch.dtype], torch.Tensor] = {}


def _is_sm70_fp8_exact_8k_prefill_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_FP8_EXACT_8K_PREFILL_SHAPES.get(suffix)
    return expected is not None and tuple(layer.weight.shape) == expected


def _is_sm70_fp8_prefill_exact_dense_layer(layer: torch.nn.Module) -> bool:
    if getattr(layer, "tp_size", 1) != 4:
        return False
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected = _SM70_FP8_PREFILL_DENSE_SHAPES.get(suffix)
    if expected is None:
        expected = _SM70_FP8_EXACT_8K_PREFILL_SHAPES.get(suffix)
    if expected is None:
        return False
    return tuple(layer.weight.shape) == expected


def _is_sm70_fp8_qpn8_layer(layer: torch.nn.Module) -> bool:
    """Admit only shapes with an accepted bounded-workspace prefill route."""
    if getattr(layer, "tp_size", 1) != 4:
        return False
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return False
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
    expected_kn = _SM70_FP8_PREFILL_DENSE_SHAPES.get(suffix)
    if expected_kn is None:
        expected_kn = _SM70_FP8_QPN8_EXTRA_SHAPES.get(suffix)
    if expected_kn is None:
        return False
    # Checkpoint-native block-FP8 weights are [N, K]; the shared prefill
    # workspace and QPN8 replacement parameter are [K, N].
    if tuple(reversed(layer.weight.shape)) != expected_kn:
        return False
    expected_k, expected_n = expected_kn
    return bool(
        getattr(layer, "input_size_per_partition", 0) == expected_k
        and getattr(layer, "output_size_per_partition", 0) == expected_n
    )


def _is_sm70_fp8_qpn8_runtime_contract() -> bool:
    """Admit only the no-MTP M range with a measured native decode kernel."""
    vllm_config = get_current_vllm_config()
    scheduler_config = getattr(vllm_config, "scheduler_config", None)
    max_num_seqs = int(getattr(scheduler_config, "max_num_seqs", 1))
    speculative_config = getattr(vllm_config, "speculative_config", None)
    if max_num_seqs > _SM70_FP8_QPN8_MAX_NUM_SEQS:
        return False
    if speculative_config is None:
        return True
    # Speculative workloads require an explicit operator opt-in; admission is
    # still based only on bounded concurrency and the layer tensor contract.
    return os.getenv("VLLM_SM70_FP8_QPN8") is not None


def _sm70_fp8_qpn8_pp2_tp4_enabled() -> bool:
    """Resolve the validated default while retaining explicit rollback."""
    generic_override = os.getenv("VLLM_SM70_FP8_QPN8")
    if generic_override is not None and not envs.VLLM_SM70_FP8_QPN8:
        return False
    specific_override = os.getenv("VLLM_SM70_FP8_QPN8_PP2_TP4")
    if specific_override is not None:
        return envs.VLLM_SM70_FP8_QPN8_PP2_TP4
    if generic_override is not None:
        return envs.VLLM_SM70_FP8_QPN8
    return envs.VLLM_SM70_FP8_QPN8_PP2_TP4


def _is_sm70_fp8_qpn8_pp2_tp4_runtime_contract() -> bool:
    """Require the measured serialized PP2 x TP4 single-request route."""
    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    scheduler_config = vllm_config.scheduler_config
    return bool(
        parallel_config.pipeline_parallel_size == 2
        and parallel_config.tensor_parallel_size == 4
        and scheduler_config.max_num_seqs == 1
        and not getattr(parallel_config, "enable_dbo", False)
        and int(getattr(parallel_config, "ubatch_size", 0)) <= 1
        and getattr(vllm_config, "speculative_config", None) is None
    )


def _sm70_fp8_qpn8_pp2_tp4_config(
    layer: torch.nn.Module, *, gated_silu: bool
) -> tuple[int, int, bool] | None:
    """Select by operator/tensor contract, never model or checkpoint identity."""
    if getattr(layer, "weight_block_size", None) != [128, 128]:
        return None
    suffix = getattr(layer, "prefix", "").rsplit(".", 1)[-1]
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
    cache_key = (device_index, torch.float16)
    workspace = _sm70_fp8_prefill_dense_workspaces.get(cache_key)
    if workspace is not None:
        return workspace
    try:
        workspace = torch.empty(
            (_SM70_FP8_PREFILL_DENSE_WORKSPACE_ELEMENTS,),
            dtype=torch.float16,
            device=weight.device,
        )
    except torch.OutOfMemoryError:
        logger.warning_once(
            "Insufficient memory for the bounded SM70 FP8 prefill workspace; "
            "falling back to TurboMind FP8."
        )
        return None
    _sm70_fp8_prefill_dense_workspaces[cache_key] = workspace
    return workspace


def _get_sm70_fp8_qpn8_pp2_tp4_workspace(
    weight: torch.Tensor,
) -> torch.Tensor | None:
    """Allocate one bounded FP16 prefill fallback per device."""
    device_index = weight.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    cache_key = (device_index, torch.float16)
    workspace = _sm70_fp8_qpn8_pp2_tp4_workspaces.get(cache_key)
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
    _sm70_fp8_qpn8_pp2_tp4_workspaces[cache_key] = workspace
    return workspace


def _sm70_fp8_prefill_visible_dense_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    dense_weight_ptr: int | None,
    *,
    gated_silu: bool,
    min_prefill_m: int,
) -> torch.Tensor | None:
    """Expose the long-prefill dense MM to AsyncTP pattern matching.

    This diagnostic route intentionally keeps the accepted dequantization and
    FP16 MM arithmetic while moving ``aten.mm`` out of the opaque C++ wrapper.
    """
    if not envs.VLLM_SM70_FP8_PREFILL_VISIBLE_DENSE_MM:
        return None
    if dense_weight_ptr is None:
        return None
    if input.dtype != torch.float16 or input.shape[0] < min_prefill_m:
        return None

    device_index = input.device.index
    if device_index is None:
        device_index = torch.accelerator.current_device_index()
    workspace = _sm70_fp8_prefill_dense_workspaces.get((device_index, torch.float16))
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


class Fp8Config(QuantizationConfig):
    """Config class for FP8."""

    def __init__(
        self,
        is_checkpoint_fp8_serialized: bool = False,
        activation_scheme: str = "dynamic",
        ignored_layers: list[str] | None = None,
        weight_block_size: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.is_checkpoint_fp8_serialized = is_checkpoint_fp8_serialized

        if activation_scheme not in ACTIVATION_SCHEMES:
            raise ValueError(f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now."
                )
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions"
                )
            if activation_scheme != "dynamic":
                raise ValueError(
                    "The block-wise quantization only supports "
                    "dynamic activation scheme for now, but got "
                    f"{activation_scheme} activation scheme."
                )
        self.weight_block_size = weight_block_size
        self.use_deep_gemm: bool | None = None

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "fp8"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        if (
            current_platform.is_cuda()
            and current_platform.has_device_capability(70)
            and not current_platform.has_device_capability(75)
            and (
                envs.VLLM_SM70_FP8_DEQUANT_FALLBACK
                or sm70_tm.forces_marlin()
                or sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
            )
        ):
            return 70
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.ignored_layers is not None:
            self.ignored_layers = hf_to_vllm_mapper.apply_list(self.ignored_layers)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Fp8Config":
        quant_method = cls.get_from_keys(config, ["quant_method"])
        is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = cls.get_from_keys(config, ["activation_scheme"])
        ignored_layers = cls.get_from_keys_or(config, ["ignored_layers"], None)
        weight_block_size = cls.get_from_keys_or(config, ["weight_block_size"], None)
        if not ignored_layers:
            ignored_layers = cls.get_from_keys_or(
                config, ["modules_to_not_convert"], None
            )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            activation_scheme=activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        if isinstance(layer, LinearBase):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedLinearMethod()
            if not self.is_checkpoint_fp8_serialized:
                online_method = Fp8OnlineLinearMethod(self)
                online_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return online_method
            else:
                offline_method = Fp8LinearMethod(self)
                offline_method.marlin_input_dtype = get_marlin_input_dtype(prefix)
                return offline_method
        elif isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if (
                self.is_checkpoint_fp8_serialized
                and current_platform.is_cuda()
                and current_platform.has_device_capability(70)
                and not current_platform.has_device_capability(75)
                and envs.VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK
                and not sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
                and not sm70_tm.forces_marlin()
            ):
                return Fp8MoEMethod(self, layer)
            if (
                self.is_checkpoint_fp8_serialized
                and current_platform.is_cuda()
                and current_platform.has_device_capability(70)
                and not current_platform.has_device_capability(75)
                and sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
            ):
                from vllm.model_executor.layers.quantization.fp8_sm70_moe import (
                    Fp8SM70MoEMethod,
                )

                return Fp8SM70MoEMethod(self, layer)
            if self.is_checkpoint_fp8_serialized:
                moe_quant_method = Fp8MoEMethod(self, layer)
            else:
                moe_quant_method = Fp8OnlineMoEMethod(self, layer)
            return moe_quant_method
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        """
        Check whether the param name matches the format for k/v cache scales
        in compressed-tensors. If this is the case, return its equivalent
        param name expected by vLLM

        :param name: param name
        :return: matching param name for KV cache scale in vLLM
        """
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None


class CopyNumelCounter(TorchDispatchMode):
    """
    Tracks total number of elements modified with `copy_`. Useful for keeping
    track of weight loading where underlying weights can be arbitrarily
    transformed (such as with `narrow`) before calling copy.
    """

    def __init__(self):
        super().__init__()
        self.copied_numel = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = func(*args, **kwargs)
        if func == torch.ops.aten.copy_.default:
            self.copied_numel += args[0].numel()
        return out


def _copy_missing_attrs(old: torch.Tensor, new: torch.Tensor) -> None:
    """Copies any attrs present in `old` but not in `new` to `new`"""
    new_attrs = set(dir(new))
    attrs_to_set = {}
    for attr in dir(old):
        if attr not in new_attrs:
            attrs_to_set[attr] = getattr(old, attr)
    set_weight_attrs(new, attrs_to_set)


class Fp8LinearMethod(LinearMethodBase):
    """Linear method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Limitations:
    1. Only support float8_e4m3fn data type due to the limitation of
       torch._scaled_mm (https://github.com/pytorch/pytorch/blob/2e48b39603411a41c5025efbe52f89560b827825/aten/src/ATen/native/cuda/Blas.cpp#L854-L856)

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config):
        self.quant_config = quant_config
        self.is_scale_e8m0 = getattr(quant_config, "is_scale_e8m0", False)
        self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype

        # For GPUs that lack FP8 hardware support, we can leverage the Marlin
        # kernel for fast weight-only FP8 quantization
        self.marlin_input_dtype = None
        self.use_marlin = False

        if self.quant_config.use_deep_gemm is not None:
            self.use_deep_gemm = self.quant_config.use_deep_gemm
        else:
            self.use_deep_gemm = is_deep_gemm_supported()

        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant = self.weight_block_size is not None
        self.act_q_static = self.quant_config.activation_scheme == "static"
        self._sm70_without_fp8_hw = (
            current_platform.is_cuda()
            and current_platform.has_device_capability(70)
            and not current_platform.has_device_capability(75)
        )
        self.use_sm70_dequant_fallback = (
            self._sm70_without_fp8_hw
            and envs.VLLM_SM70_FP8_DEQUANT_FALLBACK
            and not sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
            and not sm70_tm.forces_marlin()
        )
        self.use_sm70_fp8_turbomind = (
            self._sm70_without_fp8_hw
            and sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
            and self.block_quant
            and self.weight_block_size == [128, 128]
        )

        if self.block_quant:
            assert not self.act_q_static
            assert self.weight_block_size is not None

            self.activation_quant_key = create_fp8_quant_key(
                static=self.act_q_static,
                group_shape=GroupShape(1, self.weight_block_size[0]),
            )
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
        else:
            self.weight_quant_key = kFp8StaticTensorSym
            # Use per-token quantization for better perf if dynamic and cutlass
            if self.act_q_static:
                self.activation_quant_key = kFp8StaticTensorSym
            elif cutlass_fp8_supported():
                self.activation_quant_key = kFp8DynamicTokenSym
            else:
                self.activation_quant_key = kFp8DynamicTensorSym

    def create_weights(
        self,
        layer: RoutedExperts,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE
        if not self.block_quant:
            scale = create_fp8_scale_parameter(
                PerTensorScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                None,
                weight_loader,
            )
            layer.register_parameter("weight_scale", scale)
        else:
            assert not self.act_q_static
            assert self.weight_block_size is not None
            scale = create_fp8_scale_parameter(
                BlockQuantScaleParameter,
                output_partition_sizes,
                input_size_per_partition,
                self.weight_block_size,
                weight_loader,
                scale_dtype=(torch.float8_e8m0fnu if self.is_scale_e8m0 else None),
            )
            # The weight_scale_inv name is intentional for deepseekv3
            layer.register_parameter("weight_scale_inv", scale)

        # INPUT ACTIVATION SCALE
        if self.act_q_static:
            scale = create_fp8_input_scale(output_partition_sizes, weight_loader)
            set_weight_attrs(scale, {"scale_type": "input_scale"})
            layer.register_parameter("input_scale", scale)

        if self.use_sm70_fp8_turbomind:
            return
        if self.use_sm70_dequant_fallback:
            return

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )

        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        if getattr(layer, "sm70_fp8_turbomind", False):
            return

        if self.use_marlin:
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
            return

        input_scale = None
        # TODO(rob): refactor block quant into separate class.
        if self.use_sm70_fp8_turbomind:
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
                    if _sm70_fp8_qpn8_pp2_tp4_enabled()
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
                        os.getenv("VLLM_SM70_FP8_QPN8") == "1"
                        or os.getenv("VLLM_SM70_FP8_QPN8_PP2_TP4") == "1"
                    )
                    if missing_ops and explicitly_enabled:
                        raise RuntimeError(
                            "The explicitly enabled SM70 PP2 x TP4 QPN8 route "
                            f"requires source-built operators; missing: {missing_ops}."
                        )
                    if missing_ops:
                        logger.warning_once(
                            "The default SM70 PP2 x TP4 QPN8 route is unavailable "
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
                            qpn8_weight, qpn8_scale = sm70_ops.fp8_qpn8_prepare_sm70(
                                weight[
                                    row_start : row_start + rows_per_group
                                ].contiguous(),
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
                        layer.sm70_fp8_prefill_exact_dense_workspace_ptr = (
                            workspace.data_ptr()
                        )
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
                    tm_weight, tm_scale, meta = sm70_ops.fp8_sm70_prepare(
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
                replace_parameter(
                    layer, "weight_scale_inv", torch.stack(prepared_scales)
                )
                layer.input_scale = None
                layer.sm70_fp8_turbomind = True
                layer.sm70_fp8_bmm = True
                layer.sm70_fp8_bmm_groups = group_count
                layer.sm70_fp8_bmm_output_size = rows_per_group
                layer.register_buffer("sm70_fp8_meta", first_meta, persistent=False)
                layer.sm70_fp8_k_ld = int(first_meta[0].item())
                layer.sm70_fp8_q_ld = int(first_meta[1].item())
                logger.info_once(
                    "SM70 FP8 TurboMind grouped-BMM path enabled for DeepSeek V4."
                )
                return
            is_gated_silu_layer = self._is_sm70_gated_silu_layer(layer)
            use_gated_silu = is_gated_silu_layer and envs.VLLM_SM70_FP8_DENSE_GATED_SILU
            generic_qpn8_candidate = (
                envs.VLLM_SM70_FP8_QPN8 and _is_sm70_fp8_qpn8_layer(layer)
            )
            pp2_tp4_qpn8_config = (
                _sm70_fp8_qpn8_pp2_tp4_config(layer, gated_silu=False)
                if _sm70_fp8_qpn8_pp2_tp4_enabled()
                else None
            )
            pp2_tp4_qpn8_candidate = pp2_tp4_qpn8_config is not None
            qpn8_candidate_layer = generic_qpn8_candidate or pp2_tp4_qpn8_candidate
            qpn8_runtime = bool(
                (
                    pp2_tp4_qpn8_candidate
                    and _is_sm70_fp8_qpn8_pp2_tp4_runtime_contract()
                )
                or (generic_qpn8_candidate and _is_sm70_fp8_qpn8_runtime_contract())
            )
            if qpn8_candidate_layer and not qpn8_runtime:
                logger.info_once(
                    "The SM70 FP8 QPN8 route retains TurboMind unless its "
                    "bounded-concurrency runtime contract is explicit."
                )
            if qpn8_candidate_layer and qpn8_runtime:
                pp2_tp4_gated_config = None
                if pp2_tp4_qpn8_candidate and use_gated_silu:
                    pp2_tp4_gated_config = _sm70_fp8_qpn8_pp2_tp4_config(
                        layer, gated_silu=True
                    )
                    if pp2_tp4_gated_config is None:
                        raise RuntimeError(
                            "The SM70 PP2 x TP4 QPN8 gate/up layer violated "
                            "its fused-SiLU tensor contract."
                        )
                missing_ops = _missing_sm70_fp8_qpn8_ops()
                if missing_ops:
                    explicitly_enabled = (
                        os.getenv("VLLM_SM70_FP8_QPN8") == "1"
                        or os.getenv("VLLM_SM70_FP8_QPN8_PP2_TP4") == "1"
                    )
                    if explicitly_enabled:
                        raise RuntimeError(
                            "The explicitly enabled SM70 QPN8 route requires "
                            f"source-built operators; missing: {missing_ops}."
                        )
                    logger.warning_once(
                        "The default SM70 FP8 QPN8 route is unavailable in "
                        "the loaded vllm._C; retaining the TurboMind layout."
                    )

                if missing_ops:
                    workspace = None
                elif pp2_tp4_qpn8_candidate:
                    workspace = _get_sm70_fp8_qpn8_pp2_tp4_workspace(weight)
                else:
                    workspace = _get_sm70_fp8_prefill_exact_dense_workspace(weight)
                if not missing_ops and workspace is not None:
                    qpn8_codes, qpn8_scales = sm70_ops.fp8_qpn8_prepare_sm70(
                        weight, weight_scale_inv
                    )
                    k_dim, n_dim = (int(dim) for dim in qpn8_codes.shape)
                    if pp2_tp4_qpn8_candidate:
                        assert pp2_tp4_qpn8_config is not None
                        split_k, nacc, prefetch = pp2_tp4_qpn8_config
                    else:
                        split_k, nacc, prefetch = _SM70_FP8_QPN8_CONFIGS[
                            (k_dim, n_dim, False)
                        ]
                    replace_parameter(layer, "weight", qpn8_codes)
                    replace_parameter(layer, "weight_scale_inv", qpn8_scales)
                    layer.input_scale = None
                    layer.sm70_fp8_turbomind = True
                    layer.sm70_fp8_qpn8 = True
                    layer.sm70_fp8_qpn8_split_k = split_k
                    layer.sm70_fp8_qpn8_nacc = nacc
                    layer.sm70_fp8_qpn8_prefetch = prefetch
                    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = (
                        workspace.data_ptr()
                    )
                    if use_gated_silu:
                        if pp2_tp4_qpn8_candidate:
                            assert pp2_tp4_gated_config is not None
                            gated_split_k, gated_nacc, gated_prefetch = (
                                pp2_tp4_gated_config
                            )
                        else:
                            gated_split_k, gated_nacc, gated_prefetch = (
                                _SM70_FP8_QPN8_CONFIGS[(k_dim, n_dim, True)]
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

            tm_weight, tm_scales, meta = sm70_ops.fp8_sm70_prepare(
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
                logger.info_once(
                    "SM70 FP8 dense gated-SiLU single-layout path enabled."
                )
            replace_parameter(layer, "weight", tm_weight)
            replace_parameter(layer, "weight_scale_inv", tm_scales)
            layer.input_scale = None
            layer.sm70_fp8_turbomind = True
            layer.register_buffer("sm70_fp8_meta", meta, persistent=False)
            layer.sm70_fp8_k_ld = int(meta[0].item())
            layer.sm70_fp8_q_ld = int(meta[1].item())
            if (
                envs.VLLM_SM70_FP8_PREFILL_FAST_SELECTOR
                and envs.VLLM_SM70_FP8_PREFILL_PRESCALED
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
                envs.VLLM_SM70_FP8_PREFILL_EXACT_DENSE
                and hasattr(torch.ops._C, "fp8_gemm_sm70_prefill_dispatch_out")
                and _is_sm70_fp8_prefill_exact_dense_layer(layer)
            ):
                is_exact_8k_projection = _is_sm70_fp8_exact_8k_prefill_layer(layer)
                workspace = _get_sm70_fp8_prefill_exact_dense_workspace(tm_weight)
                if workspace is not None:
                    layer.sm70_fp8_prefill_exact_dense_workspace_ptr = (
                        workspace.data_ptr()
                    )
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

        if self.block_quant:
            assert not self.act_q_static
            if self.use_sm70_dequant_fallback:
                weight, weight_scale_inv = process_fp8_weight_block_strategy(
                    layer.weight, layer.weight_scale_inv
                )
                weight = self._dequantize_block_weight(
                    weight, weight_scale_inv, layer.orig_dtype
                )
                replace_parameter(layer, "weight", weight)
                layer.input_scale = None
                logger.warning_once(
                    "SM70 FP8 fallback enabled: FP8 block weights are "
                    "dequantized to %s at load time because this shape is not "
                    "covered by the TurboMind W8A16 dense kernel.",
                    layer.orig_dtype,
                )
                return

        # If checkpoint not serialized fp8, quantize the weights.
        else:
            # If checkpoint is fp8 per-tensor, handle that there are N scales for N
            # shards in a fused module
            weight = layer.weight
            weight_scale = layer.weight_scale

            # If using w8a8, torch._scaled_mm needs per tensor, so
            # requantize the logical shards as a single weight.
            weight, weight_scale, input_scale = process_fp8_weight_tensor_strategy(
                weight,
                weight_scale,
                layer.logical_widths,
                getattr(layer, "input_scale", None),
            )
            if self.act_q_static:
                assert input_scale is not None
                input_scale = input_scale.max()
            weight = weight.t()

            # Update layer with new values.
            replace_parameter(layer, "weight", weight.data)
            replace_parameter(layer, "weight_scale", weight_scale.data)

            if self.use_sm70_dequant_fallback:
                weight = self._dequantize_tensor_weight(
                    weight, weight_scale, layer.orig_dtype
                )
                replace_parameter(layer, "weight", weight.t().contiguous())
                layer.input_scale = None
                logger.warning_once(
                    "SM70 FP8 fallback enabled: FP8 tensor weights are "
                    "dequantized to %s at load time because this shape is not "
                    "covered by the TurboMind W8A16 dense kernel.",
                    layer.orig_dtype,
                )
                return

        if input_scale is not None:
            replace_parameter(layer, "input_scale", input_scale)
        else:
            layer.input_scale = None

        self.fp8_linear.process_weights_after_loading(layer)

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

    def _dequantize_block_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.weight_block_size is not None
        block_n, block_k = self.weight_block_size
        scales = weight_scale.to(dtype)
        scales = scales.repeat_interleave(block_n, dim=0)
        scales = scales.repeat_interleave(block_k, dim=1)
        scales = scales[: weight.shape[0], : weight.shape[1]]
        return (weight.to(dtype) * scales).contiguous()

    @staticmethod
    def _dequantize_tensor_weight(
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scales = weight_scale.to(dtype)
        if scales.numel() == 1:
            return (weight.to(dtype) * scales).contiguous()
        if scales.ndim == 1 and scales.numel() == weight.shape[1]:
            return (weight.to(dtype) * scales.view(1, -1)).contiguous()
        if scales.ndim == 1 and scales.numel() == weight.shape[0]:
            return (weight.to(dtype) * scales.view(-1, 1)).contiguous()
        return (weight.to(dtype) * scales).contiguous()

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(layer, "sm70_fp8_turbomind", False):
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
                        sm70_ops.fp8_qpn8_dispatch_sm70_out(
                            out_by_group[group_idx],
                            int(layer.sm70_fp8_prefill_exact_dense_workspace_ptr),
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

                out_shape = (*x.shape[:-1], layer.output_size_per_partition)
                x_2d = x.reshape(-1, x.shape[-1])
                if x_2d.stride(-1) != 1:
                    x_2d = x_2d.contiguous()
                out_2d = torch.empty(
                    (x_2d.shape[0], layer.output_size_per_partition),
                    device=x.device,
                    dtype=x.dtype,
                )
                if x_2d.shape[0] == 0:
                    return out_2d.reshape(out_shape)
                sm70_ops.fp8_qpn8_dispatch_sm70_out(
                    out_2d,
                    int(layer.sm70_fp8_prefill_exact_dense_workspace_ptr),
                    x_2d,
                    layer.weight,
                    layer.weight_scale_inv,
                    int(layer.sm70_fp8_qpn8_split_k),
                    int(layer.sm70_fp8_qpn8_nacc),
                    bool(layer.sm70_fp8_qpn8_prefetch),
                    False,
                )
                out = out_2d.reshape(out_shape)
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
                for group_idx in range(group_count):
                    sm70_ops.fp8_gemm_sm70_out(
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

            out_shape = (*x.shape[:-1], layer.output_size_per_partition)
            x_2d = x.reshape(-1, x.shape[-1])
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
            )
            if visible_dense_out is not None:
                out_2d = visible_dense_out
            elif prefill_workspace_ptr is not None and x_2d.dtype == torch.float16:
                sm70_ops.fp8_gemm_sm70_prefill_dispatch_out(
                    out_2d,
                    prefill_workspace_ptr,
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
                and envs.VLLM_SM70_FP8_PREFILL_FAST_SELECTOR
                and envs.VLLM_SM70_FP8_PREFILL_PRESCALED
                and x_2d.shape[0] == _SM70_FP8_EXACT_8K_PREFILL_M
            ):
                sm70_ops.fp8_gemm_sm70_prefill_prescaled_out(
                    out_2d,
                    x_2d,
                    layer.weight,
                    prefill_prescaled_scales,
                    128,
                    layer.sm70_fp8_k_ld,
                    layer.sm70_fp8_q_ld,
                )
            else:
                sm70_ops.fp8_gemm_sm70_out(
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
            out = out_2d.reshape(out_shape)
            if bias is not None:
                out.add_(bias)
            return out

        if self.use_sm70_dequant_fallback:
            return torch.nn.functional.linear(x, layer.weight, bias)

        # if batch invariant mode is enabled, prefer direct FP8 path
        # we will use BF16 dequant when direct FP8 is not supported.
        if envs.VLLM_BATCH_INVARIANT:
            if self.block_quant:
                assert self.weight_block_size is not None
                return self.fp8_linear.apply_weights(
                    layer,
                    x,
                    bias,
                )
            else:
                if isinstance(self.fp8_linear, CutlassFP8ScaledMMLinearKernel):
                    return self.fp8_linear.apply_weights(layer, x, bias)

                # per-tensor/channel: dequant to BF16 and run GEMM
                weight_fp8 = layer.weight.to(torch.bfloat16)
                weight_scale = layer.weight_scale.to(torch.bfloat16)
                if weight_scale.numel() == 1:
                    # Per-tensor: simple scalar multiplication
                    weight_bf16 = weight_fp8 * weight_scale
                else:
                    # Multiple scales (fused modules like QKV)
                    # Try to infer correct broadcasting
                    # weight is [K, N], scale could be [num_logical_weights]
                    # Need to figure out how to broadcast - for now just try
                    # direct multiplication
                    if (
                        weight_scale.dim() == 1
                        and weight_scale.shape[0] == weight_fp8.shape[0]
                    ):
                        # Per-row scaling
                        weight_bf16 = weight_fp8 * weight_scale.unsqueeze(1)
                    else:
                        # Fallback
                        weight_bf16 = weight_fp8 * weight_scale
                return torch.nn.functional.linear(x, weight_bf16.t(), bias)

        if self.use_marlin:
            return self.fp8_linear.apply_weights(layer, x, bias)

        return self.fp8_linear.apply_weights(layer, x, bias)

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
            sm70_ops.fp8_qpn8_dispatch_sm70_out(
                out_2d,
                int(layer.sm70_fp8_prefill_exact_dense_workspace_ptr),
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
            )
            if visible_dense_out is not None:
                return visible_dense_out.reshape(*x.shape[:-1], out_features)
            min_prefill_m = getattr(
                layer,
                "sm70_fp8_prefill_exact_dense_min_m",
                _SM70_FP8_PREFILL_DENSE_MIN_M,
            )
            sm70_ops.fp8_gemm_sm70_prefill_dispatch_out(
                out_2d,
                prefill_workspace_ptr,
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
        sm70_ops.fp8_gemm_sm70_out(
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


# TODO(future PR): remove this class in favor of
# online/fp8.py::Fp8PerTensorOnlineLinearMethod
class Fp8OnlineLinearMethod(Fp8LinearMethod):
    """Online version of Fp8LinearMethod which loads a full precision checkpoint
    and quantizes weights during loading."""

    uses_meta_device: bool = True

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
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        weight = ModelWeightParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        initialize_online_processing(layer)

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            weight_shape=layer.weight.shape,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            module_name=self.__class__.__name__,
        )
        self.use_marlin = isinstance(self.fp8_linear, MarlinFP8ScaledMMLinearKernel)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        # TODO(future): support block_quant in online quant path
        assert not self.block_quant

        layer.input_scale = None
        qweight, weight_scale = ops.scaled_fp8_quant(layer.weight, scale=None)

        # Update layer with new values.
        replace_parameter(layer, "weight", qweight.data)
        replace_parameter(layer, "weight_scale", weight_scale.data)

        if self.use_marlin:
            # Only Marlin kernels support `marlin_input_dtype`; guard to avoid
            # AttributeError if backend selection changes.
            if hasattr(self.fp8_linear, "marlin_input_dtype"):
                self.fp8_linear.marlin_input_dtype = self.marlin_input_dtype
            self.fp8_linear.process_weights_after_loading(layer)
        else:
            weight = qweight.t()
            replace_parameter(layer, "weight", weight.data)
            self.fp8_linear.process_weights_after_loading(layer)

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super().__init__(layer.moe_config)
        self.quant_config = quant_config
        self.weight_block_size = self.quant_config.weight_block_size
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = (
            "weight_scale_inv" if self.block_quant else "weight_scale"
        )
        self._sm70_dequant_fallback = (
            current_platform.is_cuda()
            and current_platform.has_device_capability(70)
            and not current_platform.has_device_capability(75)
            and envs.VLLM_SM70_FP8_DEQUANT_FALLBACK
            and envs.VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK
            and not sm70_tm.use_turbomind(envs.VLLM_SM70_FP8_TURBOMIND)
            and not sm70_tm.forces_marlin()
        )
        self._fallback_unquantized_method: UnquantizedFusedMoEMethod | None = None

        # Set weight key and activation key for kernel compatibility
        if self.block_quant:
            weight_key = kFp8Static128BlockSym
            activation_key = kFp8Dynamic128Sym
        else:
            weight_key = kFp8StaticTensorSym
            activation_key = (
                kFp8StaticTensorSym
                if self.quant_config.activation_scheme == "static"
                else kFp8DynamicTensorSym
            )

        if self._sm70_dequant_fallback:
            self.fp8_backend = None
            self.experts_cls = None
            self._fallback_unquantized_method = UnquantizedFusedMoEMethod(
                layer.moe_config
            )
            logger.warning_once(
                "SM70 FP8 MoE fallback enabled: FP8 MoE expert weights will "
                "be dequantized to fp16 after loading and executed with the "
                "unquantized Triton MoE path, matching the 0.0.3 V100 FP8 "
                "baseline lane. Set VLLM_SM70_FP8_MOE_DEQUANT_FALLBACK=0 "
                "to use the native SM70 FP8 MoE diagnostic route."
            )
            return

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=False,
        )

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        assert self.quant_config.is_checkpoint_fp8_serialized
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=layer.orig_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # For per-tensor quant, the scales are per expert and weight.
            w13_scale_data = torch.ones(num_experts, 2, dtype=torch.float32)
            w2_scale_data = torch.ones(num_experts, dtype=torch.float32)
        else:
            # For block quant, the scales are per block (typically 128x128).
            w13_scale_data = torch.ones(
                num_experts,
                2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
                dtype=torch.float32,
            )
            w2_scale_data = torch.ones(
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_per_partition + block_k - 1) // block_k,
                dtype=torch.float32,
            )
        w13_weight_scale = torch.nn.Parameter(w13_scale_data, requires_grad=False)
        w2_weight_scale = torch.nn.Parameter(w2_scale_data, requires_grad=False)
        # Note: name is weight_scale for tensor, weight_scale_inv for block.
        layer.register_parameter(f"w13_{self.weight_scale_name}", w13_weight_scale)
        layer.register_parameter(f"w2_{self.weight_scale_name}", w2_weight_scale)

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            if self.block_quant
            else {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def _setup_kernel(
        self,
        layer: RoutedExperts,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_input_scale: torch.Tensor | None,
        w2_input_scale: torch.Tensor | None,
    ) -> None:
        fp8_backend = self.fp8_backend
        assert fp8_backend is not None
        # Shuffle weights to runtime format.
        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, f"w13_{self.weight_scale_name}", w13_scale)
        replace_parameter(layer, f"w2_{self.weight_scale_name}", w2_scale)

        # AITER backend requires weights to be marked as shuffled.
        if fp8_backend == Fp8MoeBackend.AITER:
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._expert_routing_tables(),
            )

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13,
                w13_scale,
                w13_input_scale,
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2,
                w2_scale,
                w2_input_scale,
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.quant_config.activation_scheme == "static":
            assert not self.block_quant
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per tensor kernels require single weight scale for w13 per expert, but
        # on disk there is a scale for w1 and w3. Use the max to requantize.
        if not self.block_quant:
            shard_size = layer.intermediate_size_per_partition
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13, w13_scale, shard_size, layer.local_num_experts
            )

        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            if self.block_quant:
                w13 = self._dequantize_block_moe_weight(
                    w13, w13_scale, layer.orig_dtype
                )
                w2 = self._dequantize_block_moe_weight(w2, w2_scale, layer.orig_dtype)
            else:
                w13 = self._dequantize_tensor_moe_weight(
                    w13, w13_scale, layer.orig_dtype
                )
                w2 = self._dequantize_tensor_moe_weight(w2, w2_scale, layer.orig_dtype)
            replace_parameter(layer, "w13_weight", w13)
            replace_parameter(layer, "w2_weight", w2)
            layer.w13_input_scale = None
            layer.w2_input_scale = None
            self._fallback_unquantized_method._setup_kernel(
                layer=layer,
                w13=layer.w13_weight,
                w2=layer.w2_weight,
            )
            return

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer, w13, w2, w13_scale, w2_scale, w13_input_scale, w2_input_scale
        )

    def _dequantize_block_moe_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        assert self.weight_block_size is not None
        block_n, block_k = self.weight_block_size
        scales = weight_scale.to(dtype)
        scales = scales.repeat_interleave(block_n, dim=1)
        scales = scales.repeat_interleave(block_k, dim=2)
        scales = scales[:, : weight.shape[1], : weight.shape[2]]
        return (weight.to(dtype) * scales).contiguous()

    def _dequantize_tensor_moe_weight(
        self,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        scales = weight_scale.to(dtype)
        if scales.ndim == 2 and scales.shape[1] == 1:
            scales = scales.squeeze(1)
        if scales.ndim == 1:
            scales = scales.view(-1, 1, 1)
        return (weight.to(dtype) * scales).contiguous()

    @property
    def is_monolithic(self) -> bool:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.is_monolithic
        return super().is_monolithic

    @property
    def supports_internal_mk(self) -> bool:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.supports_internal_mk
        return super().supports_internal_mk

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.mk_can_overlap_shared_experts
        return super().mk_can_overlap_shared_experts

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.topk_indices_dtype
        return super().topk_indices_dtype

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(self, layer: RoutedExperts) -> FusedMoEQuantConfig:
        fp8_backend = self.fp8_backend
        assert fp8_backend is not None
        w1_scale = getattr(layer, f"w13_{self.weight_scale_name}")
        w2_scale = getattr(layer, f"w2_{self.weight_scale_name}")
        a1_scale = layer.w13_input_scale
        a2_scale = layer.w2_input_scale

        quant_config = make_fp8_moe_quant_config(
            fp8_backend=fp8_backend,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=self.weight_block_size,
            swiglu_limit=getattr(layer, "swiglu_limit", None),
        )

        # Inject biases into the quant config if the model has them
        # (e.g. GPT-OSS biased MoE)
        if quant_config is not None and self.moe.has_bias:
            w13_bias = getattr(layer, "w13_bias", None)
            w2_bias = getattr(layer, "w2_bias", None)
            if w13_bias is not None:
                quant_config._w1.bias = w13_bias
            if w2_bias is not None:
                quant_config._w2.bias = w2_bias

        return quant_config

    @property
    def supports_eplb(self) -> bool:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.supports_eplb
        return True

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.apply_monolithic(
                layer, x, router_logits, input_ids
            )
        assert self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        if self._sm70_dequant_fallback:
            assert self._fallback_unquantized_method is not None
            return self._fallback_unquantized_method.apply(
                layer,
                x,
                topk_weights,
                topk_ids,
                shared_experts,
                shared_experts_input,
            )
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )


# TODO(future PR): remove this class in favor of
# online/fp8.py::Fp8PerTensorOnlineMoEMethod
class Fp8OnlineMoEMethod(Fp8MoEMethod):
    """MoE method for online FP8 quantization.
    Supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    uses_meta_device: bool = True

    def __init__(self, quant_config: Fp8Config, layer: RoutedExperts):
        super().__init__(quant_config, layer)
        assert not quant_config.is_checkpoint_fp8_serialized
        assert quant_config.activation_scheme == "dynamic"
        assert quant_config.weight_block_size is None

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                device="meta",
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                device="meta",  # materialized and processed during loading
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # BIASES (for models like GPT-OSS that have biased MoE)
        if self.moe.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    device="meta",  # materialized and processed during loading
                    dtype=layer.orig_dtype,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        initialize_online_processing(layer)

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        # TODO(@ksayers): inplace fp8 quant kernel, initialize scales with ones
        if getattr(layer, "_already_called_process_weights_after_loading", False):
            return

        fp8_dtype = current_platform.fp8_dtype()
        w13 = torch.empty_like(layer.w13_weight, dtype=fp8_dtype)
        w2 = torch.empty_like(layer.w2_weight, dtype=fp8_dtype)
        w13_scale = torch.ones(
            layer.num_experts, device=w13.device, dtype=torch.float32
        )
        w2_scale = torch.ones(layer.num_experts, device=w2.device, dtype=torch.float32)
        layer.w13_input_scale = None
        layer.w2_input_scale = None

        for expert in range(layer.local_num_experts):
            w13[expert, :, :], w13_scale[expert] = ops.scaled_fp8_quant(
                layer.w13_weight[expert, :, :]
            )
            w2[expert, :, :], w2_scale[expert] = ops.scaled_fp8_quant(
                layer.w2_weight[expert, :, :]
            )

        # Shuffle weights to runtime format and setup kernel.
        self._setup_kernel(
            layer,
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_input_scale=layer.w13_input_scale,
            w2_input_scale=layer.w2_input_scale,
        )

        # Prevent duplicate processing (e.g., during weight reload)
        layer._already_called_process_weights_after_loading = True


class Fp8KVCacheMethod(BaseKVCacheMethod):
    """
    Supports loading kv-cache scaling factors from FP8 checkpoints.
    """

    def __init__(self, quant_config: Fp8Config):
        super().__init__(quant_config)
