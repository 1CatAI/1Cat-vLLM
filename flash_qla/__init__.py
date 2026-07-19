# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.0"

try:
    import tilelang.language as _T

    if not hasattr(_T, "gemm_v1") and hasattr(_T, "gemm"):
        _T.gemm_v1 = _T.gemm
except ImportError:
    pass

from flash_qla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
]
