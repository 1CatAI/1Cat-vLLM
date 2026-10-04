# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF decoder families and prepared fallback operator capabilities."""

from dataclasses import dataclass
from enum import Enum

import torch

from vllm.transformers_utils.gguf_tensor_reader import quant_size, quant_type_name


class GGUFDecoderFamily(str, Enum):
    AFFINE = "affine_integer"
    LUT4 = "lut4"
    LATTICE = "lattice_codebook"
    TERNARY = "ternary"
    FLOAT = "float"


FAMILY_FORMATS = {
    GGUFDecoderFamily.AFFINE: frozenset(
        (
            "Q1_0",
            "Q2_0",
            "Q4_0",
            "Q4_1",
            "Q5_0",
            "Q5_1",
            "Q8_0",
            "Q2_K",
            "Q3_K",
            "Q4_K",
            "Q5_K",
            "Q6_K",
        )
    ),
    GGUFDecoderFamily.LUT4: frozenset(("IQ4_NL", "IQ4_XS", "MXFP4", "NVFP4")),
    GGUFDecoderFamily.LATTICE: frozenset(
        (
            "IQ1_S",
            "IQ1_M",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ2_S",
            "IQ3_XXS",
            "IQ3_S",
        )
    ),
    GGUFDecoderFamily.TERNARY: frozenset(("TQ1_0", "TQ2_0")),
    GGUFDecoderFamily.FLOAT: frozenset(("F32", "F16", "BF16")),
}


def decoder_family(weight_type: int) -> GGUFDecoderFamily:
    name = quant_type_name(weight_type)
    for family, formats in FAMILY_FORMATS.items():
        if name in formats:
            return family
    raise ValueError(f"No GGUF decoder family declared for {name}")


@dataclass(frozen=True)
class GGUFOperatorCapability:
    family: GGUFDecoderFamily
    source_type: str
    operator: str
    graph_safe: bool
    min_m: int = 1
    max_m: int | None = None
    reason: str | None = None

    def supports_m(self, m: int) -> bool:
        return m >= self.min_m and (self.max_m is None or m <= self.max_m)


def dense_fp16_cache_capabilities(
    source_type: int, k: int, n: int, dtype: torch.dtype, enabled: bool = True
) -> tuple[GGUFOperatorCapability, ...]:
    """Measured small-projection cache; unmeasured M retains packed MMA."""
    reason = None
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif dtype != torch.float16:
        reason = "requires_fp16_activations"
    elif (source_type, k, n) not in ((8, 5120, 12), (8, 5120, 24)):
        reason = "small_projection_cache_shape_has_no_calibration"
    elif (
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        or torch.backends.cuda.matmul.allow_fp16_accumulation
    ):
        reason = "requires_fp32_matmul_policy"
    return tuple(
        GGUFOperatorCapability(
            decoder_family(source_type),
            quant_type_name(source_type),
            "aten.mm",
            True,
            min_m=minimum,
            max_m=maximum,
            reason=reason,
        )
        for minimum, maximum in (((1, 8192),) if n == 24 else ((1, 1), (32, 8192)))
    )


# Actual TP4 expert sweeps in docs/design/gguf_turbomind_lattice_decode.md.
# (source, K, N, experts) -> measured vector intervals. Unknown descriptors
# retain grouped GEMM. The unmeasured gap is deliberately not interpolated.
_LATTICE_GROUPED_VECTOR_BANDS = {
    (source, 2560, 160, experts): bands
    for source in (17, 18)
    for experts, bands in (
        (4, ((1, 8),)),
        (512, ((1, 128), (512, 512))),
    )
}


def lattice_grouped_capabilities(
    source_type: int, k: int, n: int, num_experts: int, dtype, enabled: bool = True
) -> tuple[GGUFOperatorCapability, ...]:
    """Declare prepared lattice schedules without synchronizing routed rows."""
    family = decoder_family(source_type)
    if family != GGUFDecoderFamily.LATTICE:
        raise ValueError("Canonical lattice storage is required")
    source = quant_type_name(source_type)
    group = 16 if source_type in (17, 22, 29) else 32
    shared_reason = None
    if not enabled:
        shared_reason = "disabled_by_kernel_config"
    elif dtype != torch.float16:
        shared_reason = "requires_fp16_activations"
    elif k <= 0 or n <= 0 or k % group or n % 32 or num_experts <= 0:
        shared_reason = "local_shape_cuts_canonical_group_or_output_pack"
    gemm_name = "gguf_lattice_grouped_gemm_sm70_out"
    vec_name = "gguf_lattice_grouped_vec_sm70_out"
    gemm = GGUFOperatorCapability(
        family,
        source,
        gemm_name,
        True,
        reason=shared_reason
        or (
            None
            if hasattr(torch.ops._C, gemm_name)
            else f"operator_missing:{gemm_name}"
        ),
    )
    bands = _LATTICE_GROUPED_VECTOR_BANDS.get((source_type, k, n, num_experts))
    reason = shared_reason
    if reason is None and not hasattr(torch.ops._C, vec_name):
        reason = f"operator_missing:{vec_name}"
    elif reason is None and bands is None:
        reason = "grouped_vector_shape_has_no_calibration"
    vector = tuple(
        GGUFOperatorCapability(
            family,
            source,
            vec_name,
            True,
            min_m=minimum,
            max_m=maximum,
            reason=reason,
        )
        for minimum, maximum in (bands or ((1, None),))
    )
    return (gemm, *vector)


def select_lattice_grouped_capability(capabilities, m: int) -> GGUFOperatorCapability:
    """Prefer calibrated decode, then the admitted grouped GEMM schedule."""
    for capability in (*capabilities[1:], capabilities[0]):
        if capability.reason is None and capability.supports_m(m):
            return capability
    raise ValueError("No prepared lattice grouped operator admits this descriptor")


def admit_moe_fallback(weight, weight_type: int, dtype) -> GGUFOperatorCapability:
    """Inspect the installed operator at preparation time, not on every token.

    This declares capability only. A measured TurboMind route takes precedence
    once its canonical family is available. MMVQ's explicit operator handles
    larger route counts in chunks, so selection does not require a token-count
    threshold or the raw auto selector's host-sorted grouped fallback.
    """
    family = decoder_family(weight_type)
    matrix = weight[0]
    block, size = quant_size(weight_type)
    k = (
        matrix.shape[-1] // size * block
        if weight.dtype == torch.uint8
        else matrix.shape[-1]
    )
    probe = torch.empty((1, k), dtype=dtype, device=weight.device)
    bits = torch.ops._C_gguf.ggml_dense_upstream_capabilities(
        matrix, probe, weight_type, matrix.shape[0]
    )
    if bits & 4:
        operator, graph_safe, reason = "ggml_moe_mmvq", True, None
    elif bits & 3:
        operator, graph_safe, reason = "ggml_moe_upstream", True, None
    elif bits & 8:
        operator, graph_safe, reason = "ggml_moe_mmq", True, None
    elif bits & 16:
        operator, graph_safe, reason = (
            "ggml_moe_grouped_dense",
            False,
            "graph_safe_moe_operator_unavailable",
        )
    else:
        raise ValueError(
            f"No packaged GGUF MoE fallback for {quant_type_name(weight_type)}"
        )
    return GGUFOperatorCapability(
        family, quant_type_name(weight_type), operator, graph_safe, reason=reason
    )
