# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility exports for model-independent SM70 diffusion operators."""

from vllm.model_executor.layers.sm70_attention import (
    flashattn_extension as flashattn_extension,
)
from vllm.model_executor.layers.sm70_attention import (
    flashinfer_extension as flashinfer_extension,
)
from vllm.model_executor.layers.sm70_diffusion import (
    _column_major_plan as _column_major_plan,
)
from vllm.model_executor.layers.sm70_diffusion import (
    fp16_gemm as fp16_gemm,
)
from vllm.model_executor.layers.sm70_diffusion import (
    sm70_extension,
)

w8a16_extension = sm70_extension
