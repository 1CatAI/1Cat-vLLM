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
