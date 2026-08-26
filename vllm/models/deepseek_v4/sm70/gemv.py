# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Single-token FP16 GEMV for fixed DeepSeek V4 SM70 projections."""

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_SUPPORTED_N = frozenset((64, 256, 512, 1024, 2048))
_K = 4096
_BLOCK_K = 1024
_NUM_WARPS = 4
_FP13_GROUP_VALUES = 32
_FP13_GROUP_WORDS = 13
_FP13_GROUPS_PER_ROW = _K // _FP13_GROUP_VALUES
_FP13_WORDS_PER_ROW = _FP13_GROUPS_PER_ROW * _FP13_GROUP_WORDS
_FP13_BUFFER = "_sm70_dsv4_fp13_weight"


@triton.jit
def _sm70_dsv4_fp16_gemv_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    acc = 0.0
    for block_start in tl.static_range(0, K, BLOCK_K):
        x = tl.load(x_ptr + block_start + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + row * K + block_start + offsets).to(tl.float32)
        acc += tl.sum(x * weight, axis=0)
    tl.store(out_ptr + row, acc)


@triton.jit
def _load_fp16x8(ptr):
    word0, word1, word2, word3 = tl.inline_asm_elementwise(
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        constraints="=r,=r,=r,=r,l,~{memory}",
        args=[ptr],
        dtype=(tl.uint32, tl.uint32, tl.uint32, tl.uint32),
        is_pure=False,
        pack=1,
    )
    return (
        (word0 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word0 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word1 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word1 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word2 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word2 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word3 & 0xFFFF).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
        (word3 >> 16).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32),
    )


@triton.jit
def _select_fp13_word(
    word0,
    word1,
    word2,
    word3,
    word4,
    word5,
    word6,
    word7,
    word8,
    word9,
    word10,
    word11,
    word12,
    INDEX: tl.constexpr,
):
    if INDEX == 0:  # noqa: SIM116 - Triton constexpr dispatch needs static branches.
        return word0
    elif INDEX == 1:
        return word1
    elif INDEX == 2:
        return word2
    elif INDEX == 3:
        return word3
    elif INDEX == 4:
        return word4
    elif INDEX == 5:
        return word5
    elif INDEX == 6:
        return word6
    elif INDEX == 7:
        return word7
    elif INDEX == 8:
        return word8
    elif INDEX == 9:
        return word9
    elif INDEX == 10:
        return word10
    elif INDEX == 11:
        return word11
    else:
        return word12


@triton.jit
def _sm70_dsv4_fp13_gemv_kernel(
    x_ptr,
    packed_ptr,
    out_ptr,
    GROUP_VALUES: tl.constexpr,
    GROUP_WORDS: tl.constexpr,
    GROUPS_PER_ROW: tl.constexpr,
    WORDS_PER_ROW: tl.constexpr,
):
    row = tl.program_id(0)
    groups = tl.arange(0, GROUPS_PER_ROW)
    word_base = row * WORDS_PER_ROW + groups * GROUP_WORDS
    word0 = tl.load(packed_ptr + word_base).to(tl.uint32, bitcast=True)
    word1 = tl.load(packed_ptr + word_base + 1).to(tl.uint32, bitcast=True)
    word2 = tl.load(packed_ptr + word_base + 2).to(tl.uint32, bitcast=True)
    word3 = tl.load(packed_ptr + word_base + 3).to(tl.uint32, bitcast=True)
    word4 = tl.load(packed_ptr + word_base + 4).to(tl.uint32, bitcast=True)
    word5 = tl.load(packed_ptr + word_base + 5).to(tl.uint32, bitcast=True)
    word6 = tl.load(packed_ptr + word_base + 6).to(tl.uint32, bitcast=True)
    word7 = tl.load(packed_ptr + word_base + 7).to(tl.uint32, bitcast=True)
    word8 = tl.load(packed_ptr + word_base + 8).to(tl.uint32, bitcast=True)
    word9 = tl.load(packed_ptr + word_base + 9).to(tl.uint32, bitcast=True)
    word10 = tl.load(packed_ptr + word_base + 10).to(tl.uint32, bitcast=True)
    word11 = tl.load(packed_ptr + word_base + 11).to(tl.uint32, bitcast=True)
    word12 = tl.load(packed_ptr + word_base + 12).to(tl.uint32, bitcast=True)
    x_values_0 = _load_fp16x8(x_ptr + groups * GROUP_VALUES)
    x_values_1 = _load_fp16x8(x_ptr + groups * GROUP_VALUES + 8)
    x_values_2 = _load_fp16x8(x_ptr + groups * GROUP_VALUES + 16)
    x_values_3 = _load_fp16x8(x_ptr + groups * GROUP_VALUES + 24)
    partial = tl.zeros((GROUPS_PER_ROW,), tl.float32)
    for value_index in tl.static_range(0, GROUP_VALUES):
        bit_offset = value_index * 13
        word_index = bit_offset // 32
        shift = bit_offset % 32
        code = (
            _select_fp13_word(
                word0,
                word1,
                word2,
                word3,
                word4,
                word5,
                word6,
                word7,
                word8,
                word9,
                word10,
                word11,
                word12,
                INDEX=word_index,
            )
            >> shift
        )
        if shift > 19:
            code |= _select_fp13_word(
                word0,
                word1,
                word2,
                word3,
                word4,
                word5,
                word6,
                word7,
                word8,
                word9,
                word10,
                word11,
                word12,
                INDEX=word_index + 1,
            ) << (32 - shift)
        code &= 0x1FFF
        bits = (code << 3).to(tl.uint16)
        weight = bits.to(tl.float16, bitcast=True).to(tl.float32)
        if value_index < 8:
            x = x_values_0[value_index]
        elif value_index < 16:
            x = x_values_1[value_index - 8]
        elif value_index < 24:
            x = x_values_2[value_index - 16]
        else:
            x = x_values_3[value_index - 24]
        partial += x * weight
    result = tl.sum(partial, axis=0)
    tl.store(out_ptr + row, result)


@torch.no_grad()
def _pack_sm70_dsv4_fp13_weight(weight: torch.Tensor) -> torch.Tensor:
    if weight.dtype != torch.float16 or weight.ndim != 2 or weight.shape[1] != _K:
        raise ValueError(
            f"unsupported FP13 weight contract: {weight.dtype=} {weight.shape=}"
        )
    raw = weight.view(torch.int16).to(torch.int32) & 0xFFFF
    codes = (raw >> 3).reshape(
        weight.shape[0], _FP13_GROUPS_PER_ROW, _FP13_GROUP_VALUES
    )
    words = torch.zeros(
        (weight.shape[0], _FP13_GROUPS_PER_ROW, _FP13_GROUP_WORDS),
        dtype=torch.int32,
        device=weight.device,
    )
    for value_index in range(_FP13_GROUP_VALUES):
        bit_offset = value_index * 13
        word_index = bit_offset // 32
        shift = bit_offset % 32
        value = codes[..., value_index]
        words[..., word_index].bitwise_or_(value << shift)
        if shift > 19:
            words[..., word_index + 1].bitwise_or_(value >> (32 - shift))
    return words.contiguous()


@torch.no_grad()
def _has_sm70_dsv4_fp13_weight_contract(weight: torch.Tensor) -> bool:
    """Accept exact BF16-derived normals and bounded FP16 subnormal tails."""
    bits = weight.view(torch.int16).to(torch.int32) & 0xFFFF
    exponent = (bits >> 10) & 0x1F
    discarded = bits & 0x7
    incompatible = (exponent == 0x1F) | ((discarded != 0) & (exponent != 0))
    return not bool(torch.any(incompatible).item())


def prepare_sm70_dsv4_fp13_gemv(layer: torch.nn.Module) -> torch.Tensor | None:
    if not envs.VLLM_SM70_DSV4_FP13_GEMV or not getattr(
        layer, "_sm70_dsv4_fp13_gemv", False
    ):
        return None
    weight = getattr(layer, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    if (
        not current_platform.is_cuda()
        or not current_platform.is_device_capability((7, 0))
        or not weight.is_cuda
        or weight.dtype != torch.float16
        or weight.ndim != 2
        or weight.shape[0] not in _SUPPORTED_N
        or weight.shape[1] != _K
        or not weight.is_contiguous()
    ):
        return None
    if not _has_sm70_dsv4_fp13_weight_contract(weight):
        logger.warning_once(
            "SM70 packed-FP13 GEMV rejected a weight outside its tensor-value "
            "contract; retaining the FP16 GEMV fallback."
        )
        return None
    packed = _pack_sm70_dsv4_fp13_weight(weight)
    if _FP13_BUFFER in layer._buffers:
        layer._buffers[_FP13_BUFFER] = packed
    else:
        layer.register_buffer(_FP13_BUFFER, packed, persistent=False)
    return packed


def _has_sm70_dsv4_gemv_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> bool:
    return (
        current_platform.is_cuda()
        and current_platform.is_device_capability((7, 0))
        and x.is_cuda
        and weight.is_cuda
        and x.device == weight.device
        and x.dtype == torch.float16
        and weight.dtype == torch.float16
        and output_dtype in (torch.float16, torch.float32)
        and x.ndim == 2
        and x.shape == (1, _K)
        and weight.ndim == 2
        and weight.shape[0] in _SUPPORTED_N
        and weight.shape[1] == _K
        and (output_dtype == torch.float32 or weight.shape[0] == 64)
        and x.is_contiguous()
        and weight.is_contiguous()
    )


def can_use_sm70_dsv4_fp16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
) -> bool:
    enabled = envs.VLLM_SM70_DSV4_FP16_GEMV or envs.VLLM_SM70_DSV4_FP13_GEMV
    return enabled and _has_sm70_dsv4_gemv_contract(x, weight, output_dtype)


def can_use_sm70_dsv4_fp13_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    packed_weight: torch.Tensor | None,
    output_dtype: torch.dtype,
) -> bool:
    return (
        envs.VLLM_SM70_DSV4_FP13_GEMV
        and _has_sm70_dsv4_gemv_contract(x, weight, output_dtype)
        and packed_weight is not None
        and packed_weight.dtype == torch.int32
        and packed_weight.device == weight.device
        and packed_weight.shape
        == (weight.shape[0], _FP13_GROUPS_PER_ROW, _FP13_GROUP_WORDS)
        and packed_weight.is_contiguous()
    )


def maybe_sm70_dsv4_fp16_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    output_dtype: torch.dtype,
    packed_weight: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if can_use_sm70_dsv4_fp13_gemv(x, weight, packed_weight, output_dtype):
        assert packed_weight is not None
        logger.info_once(
            "DeepSeek V4 SM70 packed-FP13 GEMV enabled for batch-one decode."
        )
        out = torch.empty((1, weight.shape[0]), device=x.device, dtype=output_dtype)
        _sm70_dsv4_fp13_gemv_kernel[(weight.shape[0],)](
            x,
            packed_weight,
            out,
            GROUP_VALUES=_FP13_GROUP_VALUES,
            GROUP_WORDS=_FP13_GROUP_WORDS,
            GROUPS_PER_ROW=_FP13_GROUPS_PER_ROW,
            WORDS_PER_ROW=_FP13_WORDS_PER_ROW,
            num_warps=_NUM_WARPS,
        )
        return out

    if not can_use_sm70_dsv4_fp16_gemv(x, weight, output_dtype):
        return None

    logger.info_once(
        "DeepSeek V4 SM70 fixed-shape FP16 GEMV enabled for batch-one decode."
    )
    out = torch.empty((1, weight.shape[0]), device=x.device, dtype=output_dtype)
    _sm70_dsv4_fp16_gemv_kernel[(weight.shape[0],)](
        x,
        weight,
        out,
        K=_K,
        BLOCK_K=_BLOCK_K,
        num_warps=_NUM_WARPS,
    )
    return out
