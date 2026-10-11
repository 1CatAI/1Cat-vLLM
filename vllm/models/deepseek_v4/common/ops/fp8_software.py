# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compatibility exports for software E4M3FN cache conversion."""

from vllm.v1.attention.ops.kv_codec import (
    _round_nearest_even_i32 as _round_nearest_even_i32,
)
from vllm.v1.attention.ops.kv_codec import (
    fp8_e4m3fn_bits_to_fp32 as fp8_e4m3fn_bits_to_fp32,
)
from vllm.v1.attention.ops.kv_codec import (
    fp8_e4m3fn_bits_to_fp32_bitcast as fp8_e4m3fn_bits_to_fp32_bitcast,
)
from vllm.v1.attention.ops.kv_codec import (
    fp32_to_fp8_e4m3fn_bits as fp32_to_fp8_e4m3fn_bits,
)
