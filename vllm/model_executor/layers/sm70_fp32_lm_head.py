# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility imports for the indexed FP32 LM-head provider."""

from vllm.model_executor.kernels.lm_head import fp32 as _provider


def __getattr__(name):
    return getattr(_provider, name)
