# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp position-learning enhancement layers."""

import ctypes
import math
import os
import resource
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import vllm.envs as envs
from vllm.config import CacheConfig, ModelConfig, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
    is_conv_state_dim_first,
)
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    is_offload_process,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    is_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.parameter import (
    ModelWeightParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)
from vllm.triton_utils import tl, triton
from vllm.utils.mem_utils import format_gib
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import (
    direct_register_custom_op,
    get_accelerator_view_from_cpu_tensor,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionBackend,
    PleShortConvAttentionMetadata,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

from ..common.ple import copy_ple_embedding_shard_

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PLE_LAYER_PRIME = 10007
_MADV_RANDOM = 1

logger = init_logger(__name__)


def _advise_random_file_access(tensor: torch.Tensor) -> str:
    """Require a lazy file mapping and disable destructive mmap read-around."""
    if tensor.device.type != "cpu" or tensor.is_meta:
        raise RuntimeError("PLE disk shards must be real CPU tensors")
    address = tensor.data_ptr()
    mapped_path = None
    with open("/proc/self/maps") as mappings:
        for line in mappings:
            fields = line.rstrip().split(maxsplit=5)
            start_text, end_text = fields[0].split("-", maxsplit=1)
            if int(start_text, 16) <= address < int(end_text, 16):
                if len(fields) == 6 and fields[5].startswith("/"):
                    mapped_path = fields[5]
                break
    if mapped_path is None:
        raise RuntimeError(
            "VLLM_PLE_DISK_OFFLOAD requires lazy file-backed safetensor "
            "weights; eager or copied tensors are unsupported"
        )

    page_size = os.sysconf("SC_PAGE_SIZE")
    byte_count = tensor.numel() * tensor.element_size()
    aligned_address = address - address % page_size
    aligned_end = (address + byte_count + page_size - 1) // page_size * page_size
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.madvise(
        ctypes.c_void_p(aligned_address),
        ctypes.c_size_t(aligned_end - aligned_address),
        _MADV_RANDOM,
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return mapped_path


@triton.jit
def _apply_float32_sign_bit(value, sign_bit):
    """Apply an FP8 sign bit without canonicalizing negative zero."""

    value_bits = tl.cast(value, tl.uint32, bitcast=True)
    signed_bits = value_bits | (sign_bit.to(tl.uint32) << 31)
    return tl.cast(signed_bits, tl.float32, bitcast=True)


@triton.jit
def _e4m3fn_byte_to_float(raw):
    """Decode E4M3FN bytes without requiring native FP8 on SM70."""

    raw_i32 = raw.to(tl.int32)
    sign_bit = (raw_i32 >> 7) & 1
    exponent = (raw_i32 >> 3) & 0x0F
    mantissa = raw_i32 & 0x07
    mantissa_f32 = mantissa.to(tl.float32)
    normal = (1.0 + mantissa_f32 * 0.125) * tl.exp2(exponent.to(tl.float32) - 7.0)
    subnormal = mantissa_f32 * 0.001953125  # 2**-9
    value = _apply_float32_sign_bit(
        tl.where(exponent == 0, subnormal, normal), sign_bit
    )
    is_nan = (exponent == 0x0F) & (mantissa == 0x07)
    return tl.where(is_nan, float("nan"), value)


@triton.jit
def _gather_ple_fp8_from_pinned_kernel(
    weight_ptr,
    ids_ptr,
    scale_ptr,
    output_ptr,
    embedding_dim: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather and dequantize one PLE row per program from pinned host memory."""

    row_idx = tl.program_id(0)
    local_idx = tl.load(ids_ptr + row_idx)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    byte_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.uint8))
    raw = tl.load(
        byte_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0,
    )
    scale = tl.load(scale_ptr).to(tl.float32)
    values = _e4m3fn_byte_to_float(raw) * scale
    tl.store(output_ptr + row_idx * embedding_dim + offsets, values, mask=mask)


@triton.jit
def _dequantize_ple_fp8_bytes_kernel(
    input_ptr,
    scale_ptr,
    output_ptr,
    numel,
    BLOCK: tl.constexpr,
):
    """Dequantize raw E4M3FN bytes without exposing FP8 to SM70 Inductor."""

    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < numel
    raw = tl.load(input_ptr + offsets, mask=mask, other=0)
    scale = tl.load(scale_ptr).to(tl.float32)
    values = _e4m3fn_byte_to_float(raw) * scale
    tl.store(output_ptr + offsets, values, mask=mask)


@triton.jit
def _qwen38_ple_m1_ngram_ids_kernel(
    input_ids_ptr,
    ngram_context_ptr,
    multipliers_ptr,
    sizes_ptr,
    offsets_ptr,
    output_ptr,
    EOS_TOKEN_ID: tl.constexpr,
):
    """Compute the exact Qwen3.8 M=1 ngram-2/3 IDs in one launch."""

    head = tl.arange(0, 16)
    current = tl.load(input_ids_ptr).to(tl.int64)
    older = tl.load(ngram_context_ptr).to(tl.int64)
    previous = tl.load(ngram_context_ptr + 1).to(tl.int64)

    # ``compute_ngram_ids`` resets history at EOS. The immediately previous
    # token remains the ngram-2 source (and is itself EOS), while ngram-3 must
    # not reach across that boundary.
    older = tl.where(previous == EOS_TOKEN_ID, EOS_TOKEN_ID, older)
    multiplier0 = tl.load(multipliers_ptr).to(tl.int64)
    multiplier1 = tl.load(multipliers_ptr + 1).to(tl.int64)
    multiplier2 = tl.load(multipliers_ptr + 2).to(tl.int64)
    mixed2 = (current * multiplier0) ^ (previous * multiplier1)
    mixed3 = mixed2 ^ (older * multiplier2)
    mixed = tl.where(head < 8, mixed2, mixed3)

    size = tl.load(sizes_ptr + head).to(tl.int64)
    offset = tl.load(offsets_ptr + head).to(tl.int64)
    remainder = mixed % size
    # PTX signed remainder follows the dividend, whereas torch.remainder is
    # always non-negative for these positive vocabulary sizes.
    remainder = tl.where(remainder < 0, remainder + size, remainder)
    tl.store(output_ptr + head, remainder + offset)


@triton.jit
def _qwen38_ple_m1_short_conv_kernel(
    x_ptr,
    state_ptr,
    weight_ptr,
    output_ptr,
    state_index_ptr,
    has_initial_ptr,
    STATE_STRIDE_0: tl.constexpr,
    STATE_STRIDE_1: tl.constexpr,
    STATE_STRIDE_2: tl.constexpr,
    HAS_INITIAL: tl.constexpr,
    NULL_STATE_ID: tl.constexpr,
    HIDDEN_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fuse exact Qwen3.8 M=1 dilated conv and state-cache update."""

    hidden = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hidden_mask = hidden < HIDDEN_SIZE
    state_index = tl.load(state_index_ptr).to(tl.int64)
    valid = state_index != NULL_STATE_ID
    safe_state_index = tl.where(valid, state_index, 0)
    use_initial = valid
    if HAS_INITIAL:
        use_initial &= tl.load(has_initial_ptr).to(tl.int1)
    state_base = safe_state_index * STATE_STRIDE_0 + hidden * STATE_STRIDE_1
    state_mask = hidden_mask & use_initial

    s0 = tl.load(state_ptr + state_base, mask=state_mask, other=0.0).to(tl.float32)
    s1 = tl.load(
        state_ptr + state_base + STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s2 = tl.load(
        state_ptr + state_base + 2 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s3 = tl.load(
        state_ptr + state_base + 3 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s4 = tl.load(
        state_ptr + state_base + 4 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s5 = tl.load(
        state_ptr + state_base + 5 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s6 = tl.load(
        state_ptr + state_base + 6 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s7 = tl.load(
        state_ptr + state_base + 7 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    s8 = tl.load(
        state_ptr + state_base + 8 * STATE_STRIDE_2,
        mask=state_mask,
        other=0.0,
    ).to(tl.float32)
    x = tl.load(x_ptr + hidden, mask=hidden_mask, other=0.0).to(tl.float32)
    w0 = tl.load(weight_ptr + hidden * 4).to(tl.float32)
    w1 = tl.load(weight_ptr + hidden * 4 + 1).to(tl.float32)
    w2 = tl.load(weight_ptr + hidden * 4 + 2).to(tl.float32)
    w3 = tl.load(weight_ptr + hidden * 4 + 3).to(tl.float32)
    conv = s0 * w0
    conv += s3 * w1
    conv += s6 * w2
    conv += x * w3
    # Preserve the depthwise-conv FP16 output boundary. The caller deliberately
    # retains native F.silu because its SM70 rounding differs slightly from
    # Triton's sigmoid approximation.
    conv = conv.to(tl.float16)
    tl.store(output_ptr + hidden, tl.where(valid, conv, 0.0), mask=hidden_mask)

    update_mask = hidden_mask & valid
    tl.store(state_ptr + state_base, s1, mask=update_mask)
    tl.store(state_ptr + state_base + STATE_STRIDE_2, s2, mask=update_mask)
    tl.store(state_ptr + state_base + 2 * STATE_STRIDE_2, s3, mask=update_mask)
    tl.store(state_ptr + state_base + 3 * STATE_STRIDE_2, s4, mask=update_mask)
    tl.store(state_ptr + state_base + 4 * STATE_STRIDE_2, s5, mask=update_mask)
    tl.store(state_ptr + state_base + 5 * STATE_STRIDE_2, s6, mask=update_mask)
    tl.store(state_ptr + state_base + 6 * STATE_STRIDE_2, s7, mask=update_mask)
    tl.store(state_ptr + state_base + 7 * STATE_STRIDE_2, s8, mask=update_mask)
    tl.store(state_ptr + state_base + 8 * STATE_STRIDE_2, x, mask=update_mask)


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _is_prime_64(value: int) -> bool:
    if value < 2:
        return False
    for prime in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % prime == 0:
            return value == prime
    exponent = value - 1
    shifts = 0
    while exponent % 2 == 0:
        exponent //= 2
        shifts += 1
    for base in (2, 325, 9375, 28178, 450775, 9780504, 1795265022):
        if base % value == 0:
            continue
        witness = pow(base, exponent, value)
        if witness in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            witness = pow(witness, 2, value)
            if witness == value - 1:
                break
        else:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = int(start)
    for _ in range(count):
        candidate = prime + 1
        if candidate <= 2:
            prime = 2
            continue
        if candidate % 2 == 0:
            candidate += 1
        while not _is_prime_64(candidate):
            candidate += 2
        prime = candidate
    return prime


class Qwen4ExpPLEGroupedNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float,
        group_size: int | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if group_size is not None and hidden_size % group_size:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"group_size ({group_size})"
            )
        self.eps = eps
        self.group_size = group_size
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        if self.group_size is None:
            variance = hidden_states.square().mean(dim=-1, keepdim=True)
            normalized = hidden_states * torch.rsqrt(variance + self.eps)
        else:
            grouped = hidden_states.unflatten(
                -1, (hidden_states.shape[-1] // self.group_size, self.group_size)
            )
            variance = grouped.square().mean(dim=-1, keepdim=True)
            normalized = (grouped * torch.rsqrt(variance + self.eps)).flatten(-2)
        return (normalized * (1.0 + self.weight.float())).to(input_dtype)


class Qwen4ExpPLEFp8EmbeddingMethod(QuantizeMethodBase):
    """FP8 PLE embedding with one global checkpoint scale."""

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del input_size, output_size
        weight_loader = extra_weight_attrs.get("weight_loader")
        weight = create_fp8_weight_parameter(
            sum(output_partition_sizes), input_size_per_partition, weight_loader
        )
        layer.register_parameter("weight", weight)

        weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            output_partition_sizes,
            input_size_per_partition,
            None,
            weight_loader,
            # Keep graph inputs in the requested model dtype. In particular,
            # an otherwise-FP16 graph cannot retain a BF16 scale parameter on
            # SM70 because Inductor rejects BF16 graph inputs there.
            scale_dtype=params_dtype,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("PLE FP8 weights only support embedding lookup")

    def embedding(self, layer: nn.Module, input_: torch.Tensor) -> torch.Tensor:
        pinned_lookup = getattr(layer, "embedding_lookup", None)
        if pinned_lookup is not None:
            return pinned_lookup(input_)
        get_accelerator_weight = getattr(layer, "get_accelerator_weight", None)
        weight = (
            get_accelerator_weight(input_.device)
            if get_accelerator_weight is not None
            else layer.weight
        )
        return F.embedding(input_, weight)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        prepare_accelerator_weight = getattr(layer, "prepare_accelerator_weight", None)
        if prepare_accelerator_weight is not None:
            prepare_accelerator_weight()


def _get_ple_embedding_quant_method(
    quant_config: QuantizationConfig | None,
    prefix: str,
    *,
    force_fp8_storage: bool = False,
) -> QuantizeMethodBase | None:
    """Select global-scale FP8 only for quantized PLE checkpoint shards."""

    if force_fp8_storage:
        return Qwen4ExpPLEFp8EmbeddingMethod()
    if not isinstance(quant_config, Fp8Config):
        return None
    if not quant_config.is_checkpoint_fp8_serialized:
        return None

    ignored_layers = quant_config.ignored_layers
    if is_layer_skipped(
        prefix,
        ignored_layers,
        quant_config.packed_modules_mapping,
    ):
        return None
    # PLE checkpoint shards form one runtime embedding parameter.
    shard_prefix = f"{prefix}.shard_"
    if any(name.startswith(shard_prefix) for name in ignored_layers):
        return None
    return Qwen4ExpPLEFp8EmbeddingMethod()


def _should_use_pinned_host_ple(config: Qwen4ExpTextConfig) -> bool:
    # The dedicated CPU-offload process owns and executes the embedding table
    # directly. The UVA pinned-host implementation is only for GPU execution.
    if is_offload_process():
        return False
    explicit = getattr(config, "ple_offload_embedding", None)
    if explicit is not None:
        return bool(explicit)
    capability = current_platform.get_device_capability()
    return capability is not None and capability.to_int() == 70


class Qwen4ExpPinnedHostEmbedding(VocabParallelEmbedding):
    """TP-sharded FP8 PLE table backed directly by pinned host memory.

    The base embedding is constructed on the meta device, so the full shard is
    never allocated on a GPU. Checkpoint shards copy directly into the pinned
    CPU parameter. A stable UVA view is created after loading and used only for
    embedding gathers.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        params_dtype: torch.dtype | None,
        padding_size: int,
        prefix: str,
        quant_method: QuantizeMethodBase,
    ) -> None:
        if not is_pin_memory_available():
            raise RuntimeError("Qwen4Exp PLE host offload requires pinned host memory")
        if not isinstance(quant_method, Qwen4ExpPLEFp8EmbeddingMethod):
            raise NotImplementedError(
                "Qwen4Exp pinned-host PLE currently requires FP8 checkpoint storage"
            )

        with torch.device("meta"):
            super().__init__(
                num_embeddings,
                embedding_dim,
                params_dtype=params_dtype,
                padding_size=padding_size,
                prefix=prefix,
                quant_method=quant_method,
            )

        meta_weight = self._parameters.get("weight")
        if not isinstance(meta_weight, torch.Tensor):
            raise RuntimeError("Qwen4Exp PLE meta weight was not initialized")
        host_weight = ModelWeightParameter(
            data=torch.empty(
                tuple(meta_weight.shape),
                dtype=meta_weight.dtype,
                device="cpu",
                pin_memory=True,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=self.weight_loader,
        )
        host_weight._vllm_keep_on_cpu = True
        self.weight = host_weight
        self.weight_scale = create_fp8_scale_parameter(
            PerTensorScaleParameter,
            [self.num_embeddings_per_partition],
            self.embedding_dim,
            None,
            self.weight_loader,
            scale_dtype=params_dtype,
        )
        self._accelerator_weight_views: dict[int, torch.Tensor] = {}
        self._accelerator_weight_ptrs: dict[int, int] = {}
        self._output_dtype = self.weight_scale.dtype
        logger.info(
            "Qwen4Exp PLE shard allocated in pinned host memory: %s",
            format_gib(self.weight.numel() * self.weight.element_size()),
        )

    def get_accelerator_weight(self, device: torch.device) -> torch.Tensor:
        if device.type != "cuda":
            raise RuntimeError(
                f"Qwen4Exp pinned-host PLE requires a CUDA input, got {device}"
            )
        device_index = (
            torch.accelerator.current_device_index()
            if device.index is None
            else device.index
        )
        view = self._accelerator_weight_views.get(device_index)
        if view is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Qwen4Exp PLE UVA view must be prepared before CUDA graph capture"
                )
            with torch.accelerator.device_index(device_index):
                view = get_accelerator_view_from_cpu_tensor(self.weight)
            self._accelerator_weight_views[device_index] = view
            self._accelerator_weight_ptrs[device_index] = view.data_ptr()
        return view

    def prepare_accelerator_weight(self) -> None:
        self.get_accelerator_weight(
            torch.device("cuda", torch.accelerator.current_device_index())
        )

    def embedding_lookup(self, input_: torch.Tensor) -> torch.Tensor:
        """Gather FP8 UVA rows and emit scaled model-dtype values."""

        device_index = (
            torch.accelerator.current_device_index()
            if input_.device.index is None
            else input_.device.index
        )
        weight_ptr = self._accelerator_weight_ptrs.get(device_index)
        if weight_ptr is None:
            self.get_accelerator_weight(input_.device)
            weight_ptr = self._accelerator_weight_ptrs[device_index]
        output = torch.empty(
            (*input_.shape, self.embedding_dim),
            dtype=self._output_dtype,
            device=input_.device,
        )
        torch.ops.vllm.qwen4_exp_ple_pinned_gather(
            input_.reshape(-1),
            output.reshape(-1, self.embedding_dim),
            self.weight_scale,
            weight_ptr,
            self.embedding_dim,
        )
        return output


class Qwen4ExpNGramEmbedding(PleOffloadLayer):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        embedding_dim: int,
        ple_dense_layer_id: int,
        max_total_tokens: int,
        max_num_reqs: int,
        prefix: str,
        layer_name: str,
        quant_config: QuantizationConfig | None = None,
        params_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.embedding_dim = embedding_dim
        self.ngram_size = int(config.ngram_size)
        self.heads_per_ngram = int(config.heads_per_ngram)
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        if self.ngram_size < 2:
            raise ValueError(f"ngram_size must be >= 2, got {self.ngram_size}")
        if self.heads_per_ngram <= 0:
            raise ValueError(f"heads_per_ngram must be > 0, got {self.heads_per_ngram}")
        if embedding_dim % self.ngram_heads:
            raise ValueError(
                "ple_embed_dim must be divisible by total ngram heads: "
                f"{embedding_dim} % {self.ngram_heads} != 0"
            )
        self.head_dim = embedding_dim // self.ngram_heads
        self.eos_token_id = int(config.eos_token_id)
        self.unigram_vocab_size = int(config.vocab_size)
        self.split_ngram_parts = int(getattr(config, "split_ngram_parts", 512))
        if self.split_ngram_parts <= 0:
            raise ValueError("split_ngram_parts must be positive")

        max_multiplier = ((1 << 63) - 1) // self.unigram_vocab_size
        half_bound = max(1, max_multiplier // 2)
        seed = int(getattr(config, "seed", None) or 1234)
        base_seed = seed + _PLE_LAYER_PRIME * ple_dense_layer_id
        multipliers = []
        for index in range(self.ngram_size):
            value = base_seed + _SPLITMIX_GAMMA * (index + 1)
            multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
        self.register_buffer(
            "layer_multipliers",
            torch.tensor(multipliers, dtype=torch.long),
            persistent=True,
        )

        ngram_vocab_size_base = int(config.ngram_vocab_size_base)
        sizes: list[int] = []
        offsets: list[int] = []
        offset = 0
        for local_head in range(self.ngram_heads):
            global_head = ple_dense_layer_id * self.ngram_heads + local_head
            size = _nth_prime_after(ngram_vocab_size_base - 1, global_head + 1)
            sizes.append(size)
            offsets.append(offset)
            offset += size
        self.register_buffer(
            "ngram_heads_vocab_sizes",
            torch.tensor(sizes, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "ngram_heads_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=True,
        )
        divisor = int(config.make_ngram_vocab_size_divisible_by)
        padded_vocab_size = ((offset + divisor - 1) // divisor) * divisor
        embedding_prefix = f"{prefix}.ngram_embedding"
        ple_storage_dtype = str(
            getattr(config, "ple_embedding_dtype", "")
        ).removeprefix("torch.")
        quant_method = _get_ple_embedding_quant_method(
            quant_config,
            embedding_prefix,
            force_fp8_storage=ple_storage_dtype == "float8_e4m3fn",
        )
        self._disk_offload = bool(envs.VLLM_PLE_DISK_OFFLOAD and is_offload_process())
        self._disk_shards: list[torch.Tensor | None] = []
        self._disk_mapped_paths: set[str] = set()
        self._disk_executor: ThreadPoolExecutor | None = None
        if self._disk_offload:
            if quant_method is None:
                raise NotImplementedError(
                    "Qwen4Exp PLE disk offload requires FP8 checkpoint storage"
                )
            with torch.device("meta"):
                self.ngram_embedding = VocabParallelEmbedding(
                    padded_vocab_size,
                    self.head_dim,
                    params_dtype=params_dtype,
                    padding_size=divisor,
                    prefix=embedding_prefix,
                    quant_method=quant_method,
                )
            self._disk_shards = [None] * self.split_ngram_parts
            shard_size = (
                self.ngram_embedding.org_vocab_size + self.split_ngram_parts - 1
            ) // self.split_ngram_parts
            self._disk_shard_size = shard_size
            self._disk_shard_boundaries = (
                torch.arange(
                    1,
                    self.split_ngram_parts,
                    dtype=torch.int64,
                )
                * shard_size
            )
            num_threads = envs.VLLM_PLE_DISK_OFFLOAD_NUM_THREADS
            if num_threads < 0:
                raise ValueError(
                    "VLLM_PLE_DISK_OFFLOAD_NUM_THREADS must be non-negative"
                )
            if num_threads == 0:
                num_threads = min(32, os.cpu_count() or 1)
            self._disk_executor = ThreadPoolExecutor(
                max_workers=num_threads,
                thread_name_prefix="ple-mmap",
            )
            logger.info(
                "Qwen4Exp PLE disk mmap enabled "
                "(shards=%d, mmap workers=%d, access=random).",
                self.split_ngram_parts,
                num_threads,
            )
        elif _should_use_pinned_host_ple(config):
            if quant_method is None:
                raise NotImplementedError(
                    "Qwen4Exp pinned-host PLE requires FP8 checkpoint storage"
                )
            self.ngram_embedding = Qwen4ExpPinnedHostEmbedding(
                padded_vocab_size,
                self.head_dim,
                params_dtype=params_dtype,
                padding_size=divisor,
                prefix=embedding_prefix,
                quant_method=quant_method,
            )
        else:
            self.ngram_embedding = VocabParallelEmbedding(
                padded_vocab_size,
                self.head_dim,
                params_dtype=params_dtype,
                padding_size=divisor,
                prefix=embedding_prefix,
                quant_method=quant_method,
            )
        self.register_buffer(
            "positions_buffer",
            torch.arange(max_total_tokens, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "padded_buffer",
            torch.full(
                (max_num_reqs, max_total_tokens),
                self.eos_token_id,
                dtype=torch.int64,
            ),
            persistent=False,
        )

    @staticmethod
    def _shift_precompute(
        tokens: torch.Tensor, eos_token_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor")
        batch_size, seq_len = tokens.shape
        positions = torch.arange(seq_len, device=tokens.device, dtype=torch.int64)
        eos_positions = torch.where(tokens == eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [
                eos_positions.new_full((batch_size, 1), -1),
                previous_eos_inclusive[:, :-1],
            ],
            dim=1,
        )
        return positions, positions.unsqueeze(0) - previous_eos - 1

    @staticmethod
    def _shift_apply(
        tokens: torch.Tensor,
        positions: torch.Tensor,
        position_in_segment: torch.Tensor,
        shift: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        if shift == 0:
            return tokens
        source = positions - shift
        gather_indices = source.clamp_min(0).unsqueeze(0).expand(tokens.shape[0], -1)
        shifted = tokens.gather(1, gather_indices)
        valid = (source.unsqueeze(0) >= 0) & (position_in_segment >= shift)
        return torch.where(valid, shifted, tokens.new_full((), eos_token_id))

    def compute_ngram_ids(
        self,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PLE indices for the current, unpadded request layout."""
        input_ids = input_ids.reshape(-1)
        num_reqs = query_start_loc.numel() - 1
        num_tokens = input_ids.shape[0]

        if num_tokens > self.positions_buffer.numel():
            raise ValueError(
                f"PLE received {num_tokens} tokens, but its workspace supports "
                f"at most {self.positions_buffer.numel()}"
            )
        if num_reqs > self.padded_buffer.shape[0]:
            raise ValueError(
                f"PLE received {num_reqs} requests, but its workspace supports "
                f"at most {self.padded_buffer.shape[0]}"
            )
        if num_reqs <= 0:
            raise ValueError("PLE requires at least one request")

        if (
            not is_offload_process()
            and input_ids.is_cuda
            and current_platform.is_device_capability((7, 0))
            and num_tokens == 1
            and num_reqs == 1
            and input_ids.dtype in (torch.int32, torch.int64)
            and self.ngram_size == 3
            and self.heads_per_ngram == 8
            and self.ngram_heads == 16
            and ngram_context.ndim == 2
            and ngram_context.shape[0] >= 1
            and ngram_context.shape[1] == 2
            and ngram_context.is_cuda
            and ngram_context.is_contiguous()
            and self.layer_multipliers.is_cuda
            and self.ngram_heads_vocab_sizes.is_cuda
            and self.ngram_heads_offsets.is_cuda
            and input_ids.device
            == ngram_context.device
            == self.layer_multipliers.device
            == self.ngram_heads_vocab_sizes.device
            == self.ngram_heads_offsets.device
        ):
            output = torch.empty((1, 16), dtype=torch.long, device=input_ids.device)
            _qwen38_ple_m1_ngram_ids_kernel[(1,)](
                input_ids,
                ngram_context,
                self.layer_multipliers,
                self.ngram_heads_vocab_sizes,
                self.ngram_heads_offsets,
                output,
                EOS_TOKEN_ID=self.eos_token_id,
                num_warps=1,
            )
            logger.info_once("SM70 Qwen3.8 fused M=1 PLE ngram-ID path enabled.")
            return output

        input_ids = input_ids.long()
        query_start_loc = query_start_loc.long()

        if is_offload_process():
            max_seq_len = max(
                1,
                int((query_start_loc[1:] - query_start_loc[:-1]).max().item()),
            )
            # CUDA-graph batches can include padded token IDs while
            # query_start_loc describes only real tokens. Do not let those
            # padded IDs overwrite the final real token after index clamping.
            num_valid_tokens = min(int(query_start_loc[-1].item()), num_tokens)
        else:
            # This method runs behind a splitting custom op on GPU, so request
            # dimensions are evaluated for every replay instead of being
            # specialized into a PIECEWISE graph keyed only by token count.
            max_seq_len = num_tokens
            num_valid_tokens = num_tokens

        positions = self.positions_buffer[:num_tokens]
        packed = self.padded_buffer[:num_reqs, :max_seq_len]
        packed.fill_(self.eos_token_id)
        request_indices = torch.searchsorted(query_start_loc, positions, right=True) - 1
        request_indices.clamp_(max=num_reqs - 1)
        columns = (positions - query_start_loc[request_indices]).clamp(
            0, packed.shape[1] - 1
        )
        if is_offload_process():
            packed[request_indices[:num_valid_tokens], columns[:num_valid_tokens]] = (
                input_ids[:num_valid_tokens]
            )
        else:
            packed[request_indices, columns] = input_ids
        ngram_context = ngram_context[:num_reqs].to(
            device=input_ids.device, dtype=torch.long
        )

        context = torch.cat([ngram_context, packed], dim=-1)
        positions_2d, position_in_segment = self._shift_precompute(
            context, self.eos_token_id
        )
        shifted = [context]
        for shift in range(1, self.ngram_size):
            shifted.append(
                self._shift_apply(
                    context,
                    positions_2d,
                    position_in_segment,
                    shift,
                    self.eos_token_id,
                )
            )
        adjusted_columns = columns + self.ngram_size - 1
        id_blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for index in range(1, ngram):
                mixed = torch.bitwise_xor(
                    mixed, shifted[index] * self.layer_multipliers[index]
                )
            sizes = self.ngram_heads_vocab_sizes[start:end]
            offsets = self.ngram_heads_offsets[start:end]
            ids = torch.remainder(mixed.unsqueeze(-1), sizes) + offsets
            id_blocks.append(ids[request_indices, adjusted_columns])
        return torch.cat(id_blocks, dim=-1)

    def _disk_embedding_lookup(
        self,
        ngram_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Gather mapped FP8 shard rows in logical-ID order."""
        if output.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise RuntimeError("PLE disk lookup currently requires FP8 output")
        if any(shard is None for shard in self._disk_shards):
            raise RuntimeError("PLE disk lookup started before every shard was loaded")

        profile = envs.VLLM_PLE_DISK_OFFLOAD_PROFILE
        if profile:
            faults_before = resource.getrusage(resource.RUSAGE_SELF)
            started = time.perf_counter()
        flat_ids = ngram_ids.reshape(-1).numpy()
        if flat_ids.size == 0:
            return
        sorted_ids, inverse = np.unique(flat_ids, return_inverse=True)
        if sorted_ids[0] < 0 or sorted_ids[-1] >= self.ngram_embedding.org_vocab_size:
            raise IndexError(
                "PLE disk row id out of range: "
                f"[{sorted_ids[0]}, {sorted_ids[-1]}] for "
                f"{self.ngram_embedding.org_vocab_size} rows"
            )
        boundaries = self._disk_shard_boundaries.numpy()
        split_positions = np.searchsorted(sorted_ids, boundaries).tolist()
        starts = [0, *split_positions]
        ends = [*split_positions, sorted_ids.size]
        sorted_output = np.empty((sorted_ids.size, self.head_dim), dtype=np.uint8)

        tasks = [
            (shard_index, start, end)
            for shard_index, (start, end) in enumerate(zip(starts, ends, strict=True))
            if start != end
        ]

        def gather_shard(task: tuple[int, int, int]) -> None:
            shard_index, start, end = task
            shard = self._disk_shards[shard_index]
            assert shard is not None
            local_ids = sorted_ids[start:end] - shard_index * self._disk_shard_size
            sorted_output[start:end] = shard.view(torch.uint8).numpy()[local_ids]

        executor = getattr(self, "_disk_executor", None)
        if executor is None or len(tasks) == 1:
            for task in tasks:
                gather_shard(task)
        else:
            for _ in executor.map(gather_shard, tasks):
                pass

        output_bytes = output.view(torch.uint8).reshape(-1, self.head_dim).numpy()
        np.take(sorted_output, inverse, axis=0, out=output_bytes)
        if profile:
            faults_after = resource.getrusage(resource.RUSAGE_SELF)
            logger.info(
                "PLE disk mmap gather: tokens=%d rows=%d wall=%.3f ms "
                "major_faults=%d minor_faults=%d",
                ngram_ids.shape[0],
                flat_ids.size,
                (time.perf_counter() - started) * 1000.0,
                faults_after.ru_majflt - faults_before.ru_majflt,
                faults_after.ru_minflt - faults_before.ru_minflt,
            )

    def forward_impl(  # type: ignore[override]
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
        output_buffer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del hidden_states
        input_ids = input_ids.reshape(-1)
        num_tokens = input_ids.shape[0]
        if is_offload_process():
            ngram_ids = self.compute_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
            )
        else:
            ngram_ids = input_ids.new_empty(
                (num_tokens, self.ngram_heads),
                dtype=torch.long,
            )
            torch.ops.vllm.qwen4_exp_compute_ple_ngram_ids(
                input_ids,
                query_start_loc,
                ngram_context,
                ngram_ids,
                self.layer_name,
            )
        if output_buffer is not None:
            output = output_buffer[:num_tokens, : self.embedding_dim]
            # Cross-process FP8 results travel as raw bytes. Keeping the IPC
            # buffers uint8 prevents TorchInductor from treating their graph
            # inputs as native FP8, which SM70 Triton does not support.
            embedding_output = (
                output.view(torch.float8_e4m3fn)
                if output.dtype == torch.uint8
                else output
            )
            if getattr(self, "_disk_offload", False):
                self._disk_embedding_lookup(ngram_ids, embedding_output)
            else:
                torch.index_select(
                    self.ngram_embedding.weight,
                    0,
                    ngram_ids.reshape(-1),
                    out=embedding_output.reshape(-1, self.head_dim),
                )
            return output
        return self.ngram_embedding(ngram_ids).flatten(-2)

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Transport quantized lookup results as opaque E4M3FN bytes."""
        embedding = getattr(self, "ngram_embedding", None)
        weight = getattr(embedding, "weight", None)
        if weight is not None and weight.dtype == torch.float8_e4m3fn:
            return torch.uint8
        if hasattr(self, "_offload_weight_scale"):
            return torch.uint8
        if weight is not None:
            return weight.dtype
        return default_dtype

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load hash buffers and checkpoint-split embedding rows."""

        disk_offload = getattr(self, "_disk_offload", False)

        # GPU workers keep only the scale required to dequantize the FP8 rows
        # returned by the CPU process. The embedding weights live exclusively
        # in that process.
        if (
            envs.VLLM_PLE_CPU_OFFLOAD
            and not envs.VLLM_SM70_QWEN38_HYBRID_PLE
            and not is_offload_process()
        ):
            retained: set[str] = set()
            for name, loaded_weight in weights:
                if name != "ngram_embedding.weight_scale":
                    continue
                # Match the resident FP8 embedding contract: its global scale
                # is a graph input in the model dtype. Retaining a checkpoint
                # BF16 scale in an otherwise-FP16 graph makes Inductor reject
                # SM70 before the explicit dequantization cast is lowered.
                scale_dtype = getattr(self, "_offload_model_dtype", loaded_weight.dtype)
                self.register_buffer(
                    "_offload_weight_scale",
                    loaded_weight.to(
                        device=torch.accelerator.current_accelerator(),
                        dtype=scale_dtype,
                    ),
                    persistent=False,
                )
                retained.add(name)
            return retained

        persistent_buffers = {
            "layer_multipliers": self.layer_multipliers,
            "ngram_heads_offsets": self.ngram_heads_offsets,
            "ngram_heads_vocab_sizes": self.ngram_heads_vocab_sizes,
        }
        loaded: set[str] = set()
        regular_weights: list[tuple[str, torch.Tensor]] = []
        shard_prefix = "ngram_embedding.shard_"

        for name, loaded_weight in weights:
            leaf_name = name.rsplit(".", 1)[-1]
            if leaf_name.startswith("hashstats_") or leaf_name == "token_lookup":
                continue
            if name in persistent_buffers:
                buffer = persistent_buffers[name]
                if buffer.shape != loaded_weight.shape:
                    raise ValueError(
                        f"Shape mismatch for {name}: expected "
                        f"{tuple(buffer.shape)}, got {tuple(loaded_weight.shape)}"
                    )
                buffer.copy_(loaded_weight.to(device=buffer.device, dtype=buffer.dtype))
                loaded.add(name)
                continue
            if name.startswith(shard_prefix) and name.endswith(".weight"):
                shard_text = name[len(shard_prefix) : -len(".weight")]
                if not shard_text.isdigit():
                    regular_weights.append((name, loaded_weight))
                    continue
                shard_index = int(shard_text)
                if shard_index >= self.split_ngram_parts:
                    raise ValueError(
                        f"PLE embedding shard index {shard_index} exceeds "
                        f"split_ngram_parts={self.split_ngram_parts}"
                    )
                embedding = self.ngram_embedding
                shard_size = (
                    embedding.org_vocab_size + self.split_ngram_parts - 1
                ) // self.split_ngram_parts
                checkpoint_start = shard_index * shard_size
                expected_rows = max(
                    0,
                    min(shard_size, embedding.org_vocab_size - checkpoint_start),
                )
                expected_shape = (expected_rows, embedding.embedding_dim)
                if tuple(loaded_weight.shape) != expected_shape:
                    raise ValueError(
                        f"Shape mismatch for PLE embedding shard {shard_index}: "
                        f"expected {expected_shape}, got "
                        f"{tuple(loaded_weight.shape)}"
                    )
                if disk_offload:
                    if loaded_weight.dtype != embedding.weight.dtype:
                        raise ValueError(
                            "PLE disk shard dtype mismatch: expected "
                            f"{embedding.weight.dtype}, got {loaded_weight.dtype}"
                        )
                    mapped_path = _advise_random_file_access(loaded_weight)
                    self._disk_shards[shard_index] = loaded_weight
                    self._disk_mapped_paths.add(mapped_path)
                    loaded.add("ngram_embedding.weight")
                    continue
                copy_ple_embedding_shard_(
                    embedding.weight.data,
                    loaded_weight,
                    checkpoint_start=checkpoint_start,
                    tp_start=embedding.shard_indices.org_vocab_start_index,
                    tp_end=embedding.shard_indices.org_vocab_end_index,
                )
                loaded.add("ngram_embedding.weight")
                continue
            if disk_offload and name == "ngram_embedding.weight_scale":
                self._disk_weight_scale = loaded_weight.clone()
                loaded.add(name)
                continue
            regular_weights.append((name, loaded_weight))

        if disk_offload:
            missing_shards = [
                index for index, shard in enumerate(self._disk_shards) if shard is None
            ]
            if missing_shards:
                raise RuntimeError(
                    f"PLE disk offload did not load shards: {missing_shards}"
                )
            mapped_gib = (
                sum(
                    shard.numel() * shard.element_size()
                    for shard in self._disk_shards
                    if shard is not None
                )
                / 2**30
            )
            logger.info(
                "Qwen4Exp PLE retained %.3f GiB across %d file-backed "
                "safetensor mappings without an anonymous table copy.",
                mapped_gib,
                len(self._disk_mapped_paths),
            )
        if regular_weights:
            loaded.update(AutoWeightsLoader(self).load_weights(regular_weights))
        return loaded


class Qwen4ExpPLELayer(nn.Module, MambaBase):
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        vllm_config: VllmConfig,
        layer_idx: int = 0,
        ple_dense_layer_id: int | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config
        self.layer_idx = layer_idx
        self.ple_dense_layer_id = (
            int(ple_dense_layer_id)
            if ple_dense_layer_id is not None
            else int(layer_idx)
        )
        self.prefix = prefix
        self.hidden_size = int(config.hidden_size)
        self.hc_count = config.hc_count
        self.hc_hidden_size = self.hidden_size * self.hc_count
        self.conv_kernel_size = int(config.ple_conv_kernel_size)
        self.short_conv_dilation = int(config.ngram_size)
        self.conv_state_len = (self.conv_kernel_size - 1) * self.short_conv_dilation
        self.num_spec_tokens = vllm_config.num_speculative_tokens
        self.activation = "silu"
        # The offload subprocess constructs the surrounding model on meta, but
        # its N-gram table must own real CPU storage. With offload disabled,
        # preserve the caller's existing device context (including SM70 UVA).
        ple_device = (
            torch.device(PleOffloadLayer.get_target_device())
            if envs.VLLM_PLE_CPU_OFFLOAD
            else nullcontext()
        )
        with ple_device:
            self.ple_embedding: nn.Module = Qwen4ExpNGramEmbedding(
                config,
                int(config.ple_embed_dim),
                self.ple_dense_layer_id,
                vllm_config.scheduler_config.max_num_batched_tokens,
                vllm_config.scheduler_config.max_num_seqs,
                prefix=f"{prefix}.ple_embedding",
                layer_name=prefix,
                quant_config=quant_config,
                params_dtype=model_config.dtype,
            )
        # PleOffloadLayer skips the embedding subclass constructor in GPU
        # workers, so preserve the model-dtype scale contract explicitly for
        # its checkpoint-only load path.
        self.ple_embedding._offload_model_dtype = model_config.dtype
        self.key_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hc_hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.key_proj",
        )
        self.value_proj = ReplicatedLinear(
            int(config.ple_embed_dim),
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.value_proj",
        )
        norm_args = (
            self.hc_hidden_size,
            config.rms_norm_eps,
            self.hidden_size,
            model_config.dtype,
        )
        self.norm_key = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_query = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.norm_conv = Qwen4ExpPLEGroupedNorm(*norm_args)
        self.conv1d = nn.Conv1d(
            self.hc_hidden_size,
            self.hc_hidden_size,
            self.conv_kernel_size,
            groups=self.hc_hidden_size,
            padding=self.conv_state_len,
            dilation=self.short_conv_dilation,
            bias=False,
            dtype=model_config.dtype,
        )
        nn.init.zeros_(self.conv1d.weight)
        self.conv1d.weight._no_reinit = True
        self.kv_cache = (torch.tensor([]),)
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def _get_embedding_weight_scale(self) -> torch.Tensor | None:
        embedding = getattr(self.ple_embedding, "ngram_embedding", None)
        weight_scale = getattr(embedding, "weight_scale", None)
        if weight_scale is not None:
            return weight_scale
        return getattr(self.ple_embedding, "_offload_weight_scale", None)

    def _dequantize_embeddings(
        self,
        embeddings: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Dequantize PLE lookup output."""

        if embeddings.dtype == torch.uint8:
            weight_scale = self._get_embedding_weight_scale()
            if weight_scale is None:
                raise RuntimeError("FP8 PLE embedding is missing its global scale")
            if weight_scale.device != embeddings.device:
                raise RuntimeError(
                    "FP8 PLE embedding scale must be on the output device"
                )
            output = torch.empty_like(embeddings, dtype=output_dtype)
            torch.ops.vllm.qwen4_exp_ple_fp8_bytes_dequant(
                embeddings,
                weight_scale,
                output,
            )
            return output
        if not is_fp8(embeddings):
            return embeddings
        weight_scale = self._get_embedding_weight_scale()
        if weight_scale is None:
            raise RuntimeError("FP8 PLE embedding is missing its global scale")
        if weight_scale.device != embeddings.device:
            raise RuntimeError("FP8 PLE embedding scale must be on the output device")
        return embeddings.to(output_dtype) * weight_scale.to(output_dtype)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        return True

    def get_attn_backend(self) -> type[PleShortConvAttentionBackend]:
        return PleShortConvAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return MambaStateDtypeCalculator.short_conv_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype
        )

    def get_state_shape(self) -> Sequence[tuple[int, ...]]:
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=1,
            intermediate_size=self.hc_hidden_size,
            conv_kernel=self.conv_state_len + 1,
            num_spec=self.num_spec_tokens,
        )

    def _apply_norm(
        self, norm: Qwen4ExpPLEGroupedNorm, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        shape = hidden_states.shape
        return norm(hidden_states.flatten(-2)).reshape(shape)

    def _short_conv_fallback(self, inputs: torch.Tensor) -> torch.Tensor:
        # Profiling / CUDA graph capture only; conv state is not updated.
        inputs_t = inputs.transpose(0, 1).unsqueeze(0)
        output = self.conv1d(inputs_t)[..., : inputs_t.size(-1)]
        return F.silu(output).squeeze(0).transpose(0, 1)

    def _short_conv_dilated_decode_batched(
        self,
        x_d: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_d: torch.Tensor,
        has_initial_states_d: torch.Tensor | None,
    ) -> torch.Tensor:
        has_initial_ok = has_initial_states_d is None or (
            has_initial_states_d.numel() >= 1
            and has_initial_states_d.is_cuda
            and has_initial_states_d.is_contiguous()
        )
        if (
            current_platform.is_device_capability((7, 0))
            and x_d.shape == (1, 10240)
            and x_d.dtype == torch.float16
            and x_d.is_cuda
            and x_d.is_contiguous()
            and conv_state.ndim == 3
            and conv_state.shape[1] == 10240
            and conv_state.shape[2] == 9
            and conv_state.dtype == torch.float16
            and conv_state.is_cuda
            and conv_weights.shape == (10240, 4)
            and conv_weights.dtype == torch.float16
            and conv_weights.is_cuda
            and conv_weights.is_contiguous()
            and state_indices_tensor_d.numel() == 1
            and state_indices_tensor_d.dtype in (torch.int32, torch.int64)
            and state_indices_tensor_d.is_cuda
            and state_indices_tensor_d.is_contiguous()
            and has_initial_ok
            and x_d.device
            == conv_state.device
            == conv_weights.device
            == state_indices_tensor_d.device
            and (
                has_initial_states_d is None
                or has_initial_states_d.device == x_d.device
            )
        ):
            conv_output = torch.empty_like(x_d)
            has_initial_ptr = (
                state_indices_tensor_d
                if has_initial_states_d is None
                else has_initial_states_d
            )
            _qwen38_ple_m1_short_conv_kernel[(triton.cdiv(10240, 256),)](
                x_d,
                conv_state,
                conv_weights,
                conv_output,
                state_indices_tensor_d,
                has_initial_ptr,
                STATE_STRIDE_0=conv_state.stride(0),
                STATE_STRIDE_1=conv_state.stride(1),
                STATE_STRIDE_2=conv_state.stride(2),
                HAS_INITIAL=has_initial_states_d is not None,
                NULL_STATE_ID=NULL_BLOCK_ID,
                HIDDEN_SIZE=10240,
                BLOCK=256,
                num_warps=4,
            )
            logger.info_once(
                "SM70 Qwen3.8 fused M=1 PLE short-conv state path enabled."
            )
            return F.silu(conv_output)

        state_indices = state_indices_tensor_d.to(
            device=conv_state.device, dtype=torch.int64
        )
        # TODO: need double-check
        # FULL cudagraph padded decode rows use NULL_BLOCK_ID. Remap them to
        # slot 0 for a safe gather, then zero output and skip write-back.
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        if has_initial_states_d is None:
            has_initial_state = valid_state
        else:
            if has_initial_states_d.numel() < state_indices_tensor_d.numel():
                raise ValueError(
                    "has_initial_states_d size mismatch: "
                    f"got {has_initial_states_d.numel()}, "
                    f"need >= {state_indices_tensor_d.numel()}."
                )
            has_initial_state = has_initial_states_d[
                : state_indices_tensor_d.numel()
            ].to(device=conv_state.device, dtype=torch.bool)
            has_initial_state = has_initial_state & valid_state

        cached_state = conv_state.index_select(0, state_indices)
        state = cached_state[..., : self.conv_state_len].to(x_d.dtype)
        if self.conv_state_len > 0:
            initial_state = torch.where(
                has_initial_state.view(-1, 1, 1),
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, x_d.unsqueeze(-1)), dim=-1)
        else:
            history = x_d.unsqueeze(-1)

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        ).squeeze(-1)
        output = F.silu(conv_output)
        output = output * valid_state.view(-1, 1).to(output.dtype)

        if self.conv_state_len > 0:
            next_state = history[..., -self.conv_state_len :]
            # Padded rows are remapped to the reserved null slot. Preserve its
            # existing value while writing the new states for valid rows.
            existing_base_state = cached_state[..., : self.conv_state_len]
            safe_next_state = torch.where(
                valid_state.view(-1, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            cached_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_prefill_batched(
        self,
        x_p: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        state_indices_tensor_p: torch.Tensor,
        num_prefills: int,
        num_decode_tokens: int,
        num_prefill_tokens: int,
    ) -> torch.Tensor:
        # ``non_spec_query_start_loc`` covers the non-spec (decode + prefill)
        # requests and equals ``query_start_loc`` when spec-decode is inactive.
        non_spec_query_start_loc = metadata.non_spec_query_start_loc
        if non_spec_query_start_loc is None:
            raise ValueError("query_start_loc is required for prefill short-conv")
        query_start_loc_p = (
            non_spec_query_start_loc[-num_prefills - 1 :] - num_decode_tokens
        )
        # The metadata builder guarantees that the prefill query offsets start
        # at 0 and end at num_prefill_tokens. Avoid reading those values here,
        # since doing so would force a device-to-host synchronization.
        has_initial_states_p = metadata.has_initial_states_p
        if has_initial_states_p is None:
            raise ValueError("has_initial_states_p is required for prefill short-conv")

        output = torch.empty_like(x_p)
        q_starts = query_start_loc_p.to(torch.int64)
        if state_indices_tensor_p.numel() < num_prefills:
            raise ValueError(
                "state_indices_tensor_p size mismatch: "
                f"got {state_indices_tensor_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if has_initial_states_p.numel() < num_prefills:
            raise ValueError(
                "has_initial_states_p size mismatch: "
                f"got {has_initial_states_p.numel()}, "
                f"need >= {num_prefills}."
            )
        if num_prefills == 0 or x_p.numel() == 0:
            return output
        lengths = q_starts[1:] - q_starts[:-1]
        # Use the CPU-computed packing width from the metadata builder instead
        # of synchronizing on lengths.max().
        max_len = metadata.max_prefill_query_len
        if max_len <= 0:
            return output

        hidden_size = x_p.shape[1]
        positions = torch.arange(
            num_prefill_tokens, device=x_p.device, dtype=torch.int64
        )
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        col_indices = positions - q_starts[req_indices]

        packed_tokens = x_p.new_zeros((num_prefills, max_len, hidden_size))
        packed_tokens[req_indices, col_indices] = x_p
        packed_tokens = packed_tokens.transpose(1, 2).contiguous()

        state_indices = state_indices_tensor_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        has_initial = has_initial_states_p[:num_prefills].to(
            device=conv_state.device, dtype=torch.bool
        )
        if self.conv_state_len > 0:
            if conv_state.shape[0] == 0:
                state = conv_state.new_zeros(
                    (num_prefills, hidden_size, self.conv_state_len),
                    dtype=x_p.dtype,
                )
            else:
                state = conv_state.index_select(0, state_indices)[
                    ..., : self.conv_state_len
                ].to(x_p.dtype)
            use_initial_mask = (valid_state & has_initial).view(num_prefills, 1, 1)
            initial_state = torch.where(
                use_initial_mask,
                state,
                torch.zeros_like(state),
            )
            history = torch.cat((initial_state, packed_tokens), dim=-1)
        else:
            history = packed_tokens

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        token_positions = torch.arange(max_len, device=x_p.device, dtype=torch.int64)
        valid_tokens = token_positions.view(1, max_len) < lengths.view(num_prefills, 1)
        valid_output_mask = valid_tokens & valid_state.to(device=x_p.device).view(
            num_prefills, 1
        )
        conv_output.masked_fill_(~valid_output_mask.unsqueeze(-1), 0)
        output.copy_(conv_output[req_indices, col_indices])

        if self.conv_state_len > 0 and conv_state.shape[0] > 0:
            state_starts = lengths.to(device=history.device, dtype=torch.int64).view(
                num_prefills, 1, 1
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=history.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            next_state = history.gather(
                dim=2,
                index=(state_starts + state_offsets).expand(-1, history.size(1), -1),
            )
            # Write back without a host synchronization. Valid, non-empty rows
            # receive their new state; padding and zero-length rows keep the
            # current cache value.
            existing_state = conv_state.index_select(0, state_indices)
            existing_base_state = existing_state[..., : self.conv_state_len]
            update_mask = valid_state & (lengths.to(device=conv_state.device) > 0)
            safe_next_state = torch.where(
                update_mask.view(num_prefills, 1, 1),
                next_state.to(conv_state.dtype),
                existing_base_state,
            )
            existing_state[..., : self.conv_state_len] = safe_next_state
            conv_state.index_copy_(0, state_indices, existing_state)
        return output

    def _short_conv_dilated_spec_batched(
        self,
        x_spec: torch.Tensor,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
        spec_state_indices_tensor: torch.Tensor,
        spec_query_start_loc: torch.Tensor,
        num_accepted_tokens: torch.Tensor,
        spec_query_len: int,
    ) -> torch.Tensor:
        """Dilated short-conv for speculative-decode (MTP) requests.

        Each spec request feeds multiple (draft + 1) query tokens. The conv
        outputs are computed causally after rolling back the previous draft
        state by ``num_accepted_tokens - 1``. The current candidate inputs stay
        in the extended cache for the next forward, matching
        ``causal_conv1d_update``.

        ``spec_query_len`` (== num_speculative_tokens + 1) is the maximum query
        length and is a Python int, so no host synchronization is needed; this
        keeps the path safe for full CUDA-graph capture/replay where the buffers
        are padded at the request level.
        """
        num_reqs = spec_state_indices_tensor.numel()
        hidden_size = x_spec.size(-1)
        # Use a fixed packing width instead of synchronizing on lengths.max().
        max_len = spec_query_len
        # Full CUDA graphs can pad these buffers. Only the first num_reqs
        # accepted-token counts belong to actual speculative requests.
        num_accepted_tokens = num_accepted_tokens[:num_reqs]
        q_starts = spec_query_start_loc[: num_reqs + 1].to(torch.int64)
        # Keep the number of real speculative tokens on the device.
        total_real_tokens = q_starts[num_reqs]

        state_indices = spec_state_indices_tensor.to(
            device=conv_state.device, dtype=torch.int64
        )
        valid_state = state_indices != NULL_BLOCK_ID
        state_indices = torch.where(
            valid_state, state_indices, torch.zeros_like(state_indices)
        )
        positions = torch.arange(
            x_spec.size(0), device=x_spec.device, dtype=torch.int64
        )
        # Route graph-padded token rows to the discarded dummy request so that
        # they cannot overwrite real packed data.
        req_indices = torch.searchsorted(q_starts[1:], positions, right=True)
        valid_tokens = (positions < total_real_tokens) & (req_indices < num_reqs)
        clamped_req_indices = req_indices.clamp_max(max(num_reqs - 1, 0))
        col_indices = (positions - q_starts[clamped_req_indices]).clamp_(0, max_len - 1)
        pack_req_indices = torch.where(
            valid_tokens,
            clamped_req_indices,
            torch.full_like(req_indices, num_reqs),
        )
        pack_col_indices = torch.where(
            valid_tokens, col_indices, torch.zeros_like(col_indices)
        )

        # The last request row is the dummy sink for graph padding.
        packed = x_spec.new_zeros((num_reqs + 1, max_len, hidden_size))
        packed[pack_req_indices, pack_col_indices] = x_spec
        packed = packed.transpose(1, 2).contiguous()

        if self.conv_state_len > 0:
            cached_state = conv_state.index_select(0, state_indices)
            rollback_offsets = num_accepted_tokens.to(
                device=conv_state.device, dtype=torch.int64
            ).sub(1)
            rollback_offsets = torch.where(
                valid_state,
                rollback_offsets.clamp_(0, max_len - 1),
                torch.zeros_like(rollback_offsets),
            )
            state_offsets = torch.arange(
                self.conv_state_len, device=conv_state.device, dtype=torch.int64
            ).view(1, 1, self.conv_state_len)
            rollback_indices = rollback_offsets.view(-1, 1, 1) + state_offsets
            state = cached_state.gather(
                2, rollback_indices.expand(-1, hidden_size, -1)
            ).to(x_spec.dtype)
            state = torch.where(
                valid_state.view(num_reqs, 1, 1),
                state,
                torch.zeros_like(state),
            )
            # Append a zeroed dummy-row state to match the [num_reqs + 1] pack.
            dummy_state = state.new_zeros((1, hidden_size, self.conv_state_len))
            state_full = torch.cat((state, dummy_state), dim=0)
            history = torch.cat((state_full, packed), dim=-1)
        else:
            history = packed

        conv_output = F.conv1d(
            history,
            conv_weights.unsqueeze(1).contiguous(),
            groups=history.size(1),
            dilation=self.short_conv_dilation,
        )
        conv_output = F.silu(conv_output).transpose(1, 2).contiguous()

        output = conv_output[pack_req_indices, pack_col_indices]
        output = output * valid_tokens.view(-1, 1).to(output.dtype)

        # Keep all current candidate inputs in the extended state. On the next
        # target forward, ``num_accepted_tokens - 1`` selects the rollback
        # window before processing the newly scheduled tokens.
        if self.conv_state_len > 0:
            state_capacity = self.conv_state_len + max_len - 1
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache cannot retain speculative tokens: "
                    f"got {conv_state.size(-1)}, need {state_capacity}."
                )
            candidate_state = history[:num_reqs, :, 1 : state_capacity + 1]
            query_lengths = q_starts[1:] - q_starts[:-1]
            state_positions = torch.arange(
                state_capacity, device=history.device, dtype=torch.int64
            ).view(1, 1, state_capacity)
            update_lengths = (self.conv_state_len + query_lengths - 1).view(
                num_reqs, 1, 1
            )
            update_mask = valid_state.view(num_reqs, 1, 1) & (
                state_positions < update_lengths
            )
            existing_state = cached_state[..., :state_capacity]
            next_state = torch.where(
                update_mask,
                candidate_state.to(conv_state.dtype),
                existing_state,
            )
            cached_state[..., :state_capacity] = next_state
            conv_state.index_copy_(0, state_indices, cached_state)

        return output

    def _short_conv_dilated_dispatch(
        self,
        inputs: torch.Tensor,
        metadata: PleShortConvAttentionMetadata,
        conv_state: torch.Tensor,
        conv_weights: torch.Tensor,
    ) -> torch.Tensor:
        num_prefills = metadata.num_prefills
        num_decodes = metadata.num_decodes
        num_decode_tokens = metadata.num_decode_tokens
        num_prefill_tokens = metadata.num_prefill_tokens
        has_prefill = num_prefills > 0
        has_decode = num_decodes > 0
        has_spec = metadata.spec_sequence_masks is not None
        x = inputs[: metadata.num_actual_tokens]

        # Split spec / non-spec tokens.
        if has_spec:
            if has_prefill or has_decode:
                assert metadata.spec_token_indx is not None
                assert metadata.non_spec_token_indx is not None
                x_spec = x.index_select(0, metadata.spec_token_indx.long())
                x_non_spec = x.index_select(0, metadata.non_spec_token_indx.long())
            else:
                x_spec = x
                x_non_spec = None
        else:
            x_spec = None
            x_non_spec = x

        spec_output = None
        # 1. Run the multi-query speculative-decode part.
        if has_spec:
            assert metadata.spec_state_indices_tensor is not None
            assert metadata.spec_query_start_loc is not None
            assert metadata.num_accepted_tokens is not None
            spec_output = self._short_conv_dilated_spec_batched(
                x_spec=x_spec,
                conv_state=conv_state,
                conv_weights=conv_weights,
                spec_state_indices_tensor=metadata.spec_state_indices_tensor[
                    : metadata.num_spec_decodes
                ],
                spec_query_start_loc=metadata.spec_query_start_loc,
                num_accepted_tokens=metadata.num_accepted_tokens,
                spec_query_len=metadata.spec_query_len,
            )

        # 2. Run regular decode and prefill requests.
        conv_out_non_spec = None
        state_indices_tensor = metadata.state_indices_tensor
        if x_non_spec is not None:
            assert state_indices_tensor is not None
            if has_prefill:
                state_indices_tensor_d, state_indices_tensor_p = torch.split(
                    state_indices_tensor,
                    [num_decodes, num_prefills],
                    dim=0,
                )
                x_d, x_p = torch.split(
                    x_non_spec,
                    [num_decode_tokens, num_prefill_tokens],
                    dim=0,
                )
                non_spec_parts: list[torch.Tensor] = []
                if has_decode:
                    non_spec_parts.append(
                        self._short_conv_dilated_decode_batched(
                            x_d=x_d,
                            conv_state=conv_state,
                            conv_weights=conv_weights,
                            state_indices_tensor_d=state_indices_tensor_d,
                            has_initial_states_d=metadata.has_initial_states_d,
                        )
                    )
                non_spec_parts.append(
                    self._short_conv_dilated_prefill_batched(
                        x_p=x_p,
                        metadata=metadata,
                        conv_state=conv_state,
                        conv_weights=conv_weights,
                        state_indices_tensor_p=state_indices_tensor_p,
                        num_prefills=num_prefills,
                        num_decode_tokens=num_decode_tokens,
                        num_prefill_tokens=num_prefill_tokens,
                    )
                )
                conv_out_non_spec = torch.vstack(non_spec_parts)
            else:
                conv_out_non_spec = self._short_conv_dilated_decode_batched(
                    x_d=x_non_spec,
                    conv_state=conv_state,
                    conv_weights=conv_weights,
                    state_indices_tensor_d=state_indices_tensor[: x_non_spec.size(0)],
                    has_initial_states_d=metadata.has_initial_states_d,
                )

        # 3. Merge both parts back into the original token order.
        if has_spec and conv_out_non_spec is not None:
            assert metadata.spec_token_indx is not None
            assert metadata.non_spec_token_indx is not None
            assert spec_output is not None
            output = x.new_empty((metadata.num_actual_tokens, x.size(-1)))
            output.index_copy_(0, metadata.spec_token_indx, spec_output)
            output.index_copy_(0, metadata.non_spec_token_indx, conv_out_non_spec)
            return output
        elif has_spec:
            assert spec_output is not None
            return spec_output
        if conv_out_non_spec is None:
            return x
        return conv_out_non_spec

    def _short_conv(self, inputs: torch.Tensor) -> torch.Tensor:
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            return self._short_conv_fallback(inputs)

        if not isinstance(attn_metadata, dict):
            raise RuntimeError(
                "PLE short-conv expects per-layer attention metadata dict "
                f"during inference, got {type(attn_metadata).__name__}."
            )

        layer_attn_metadata = attn_metadata.get(self.prefix)
        if layer_attn_metadata is None:
            raise RuntimeError(
                f"Missing short-conv metadata for layer '{self.prefix}'. "
                "This would bypass conv-state updates and is not allowed."
            )
        if not isinstance(layer_attn_metadata, PleShortConvAttentionMetadata):
            raise TypeError(
                "Expected PleShortConvAttentionMetadata for layer "
                f"'{self.prefix}', got "
                f"{type(layer_attn_metadata).__name__}."
            )

        conv_state = self.kv_cache[0]
        if not is_conv_state_dim_first():
            conv_state = conv_state.transpose(-1, -2)
        conv_weights = self.conv1d.weight.squeeze(1)

        state_capacity = self.conv_state_len + self.num_spec_tokens
        if state_capacity > 0:
            if conv_state.size(-1) < state_capacity:
                raise RuntimeError(
                    "PLE short-conv cache is smaller than expected for "
                    f"dilated convolution: got {conv_state.size(-1)}, "
                    f"expect at least {state_capacity}."
                )
            conv_state = conv_state[..., -state_capacity:]
        return self._short_conv_dilated_dispatch(
            inputs,
            layer_attn_metadata,
            conv_state,
            conv_weights.to(dtype=inputs.dtype),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        query_start_loc: torch.Tensor,
        ngram_context: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.reshape(-1)
        if input_ids.shape[0] != hidden_states.shape[0]:
            raise ValueError(
                "PLE expects input_ids and hidden_states to have the same "
                f"token length, got {input_ids.shape[0]} and "
                f"{hidden_states.shape[0]}"
            )
        embeddings = self.ple_embedding(
            hidden_states,
            input_ids,
            query_start_loc,
            ngram_context,
        )
        embeddings = self._dequantize_embeddings(embeddings, hidden_states.dtype)
        key, _ = self.key_proj(embeddings)
        value, _ = self.value_proj(embeddings)
        token_count = hidden_states.shape[0]
        key = key.reshape(token_count, self.hc_count, self.hidden_size)
        query = hidden_states.reshape(token_count, self.hc_count, self.hidden_size)
        key = self._apply_norm(self.norm_key, key)
        query = self._apply_norm(self.norm_query, query)
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = torch.sigmoid(gate.sign() * gate.abs().clamp_min(1e-6).sqrt())
        gated_value = gate * value.unsqueeze(-2)
        normalized = self._apply_norm(self.norm_conv, gated_value).flatten(-2)
        conv_output = torch.zeros_like(normalized)
        torch.ops.vllm.qwen4_exp_ple_short_conv(
            normalized,
            conv_output,
            self.prefix,
        )
        return gated_value.flatten(-2) + conv_output


def qwen4_exp_ple_short_conv(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    result = layer._short_conv(inputs)
    output[: result.shape[0]].copy_(result)


def qwen4_exp_ple_short_conv_fake(
    inputs: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def qwen4_exp_compute_ple_ngram_ids(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Compute request-dependent PLE IDs outside PIECEWISE CUDA graphs."""
    layer = get_forward_context().no_compile_layers[layer_name]
    ngram_ids = layer.ple_embedding.compute_ngram_ids(
        input_ids,
        query_start_loc,
        ngram_context,
    )
    output.copy_(ngram_ids)


def qwen4_exp_compute_ple_ngram_ids_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def qwen4_exp_ple_pinned_gather(
    input_ids: torch.Tensor,
    output: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_ptr: int,
    embedding_dim: int,
) -> None:
    if input_ids.numel() == 0:
        return
    block_d = triton.next_power_of_2(embedding_dim)
    _gather_ple_fp8_from_pinned_kernel[(input_ids.numel(),)](
        weight_ptr,
        input_ids,
        weight_scale,
        output,
        embedding_dim=embedding_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


def qwen4_exp_ple_pinned_gather_fake(
    input_ids: torch.Tensor,
    output: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_ptr: int,
    embedding_dim: int,
) -> None:
    return


def qwen4_exp_ple_fp8_bytes_dequant(
    input_bytes: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    if input_bytes.dtype != torch.uint8:
        raise TypeError(
            f"Qwen4Exp PLE byte dequant expects uint8 input, got {input_bytes.dtype}"
        )
    if input_bytes.numel() == 0:
        return
    block = 1024
    _dequantize_ple_fp8_bytes_kernel[(triton.cdiv(input_bytes.numel(), block),)](
        input_bytes,
        weight_scale,
        output,
        input_bytes.numel(),
        BLOCK=block,
        num_warps=4,
    )


def qwen4_exp_ple_fp8_bytes_dequant_fake(
    input_bytes: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="qwen4_exp_compute_ple_ngram_ids",
    op_func=qwen4_exp_compute_ple_ngram_ids,
    mutates_args=["output"],
    fake_impl=qwen4_exp_compute_ple_ngram_ids_fake,
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_pinned_gather",
    op_func=qwen4_exp_ple_pinned_gather,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ple_pinned_gather_fake,
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_fp8_bytes_dequant",
    op_func=qwen4_exp_ple_fp8_bytes_dequant,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ple_fp8_bytes_dequant_fake,
)


direct_register_custom_op(
    op_name="qwen4_exp_ple_short_conv",
    op_func=qwen4_exp_ple_short_conv,
    mutates_args=["output"],
    fake_impl=qwen4_exp_ple_short_conv_fake,
)


__all__ = [
    "Qwen4ExpNGramEmbedding",
    "Qwen4ExpPLEGroupedNorm",
    "Qwen4ExpPLELayer",
]
