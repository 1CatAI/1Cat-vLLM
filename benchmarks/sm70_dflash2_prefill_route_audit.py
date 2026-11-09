# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inspect native prefill availability and route counts between requests.

Compose this worker extension with the frozen performance extension. It does
not load missing libraries or instrument model execution; the explicit launch
configuration must resolve native dependencies before model initialization.
"""

import hashlib
import os
from pathlib import Path


class PrefillRouteAuditExtension:
    def dflash2_prefill_route_snapshot(self):
        import torch

        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
        from vllm.v1.attention.backends import flash_attn_v100 as backend

        required = (
            "sm70_d256_splitd_n32_dense_fwd",
            "sm70_d256_splitd_n32_paged_fwd",
            "sm70_d256_gqa_v37_fwd",
            "sm70_v37_e4m3_bridge",
        )
        available = {name: hasattr(torch.ops._vllm_fa2_C, name) for name in required}
        flags = {
            name: os.getenv(name, default)
            for name, default in (
                ("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "1"),
                ("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "1"),
                ("VLLM_FLASH_V100_FA2_D256_PREFILL", "1"),
            )
        }
        libraries = {}
        for line in Path("/proc/self/maps").read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) == 6 and "_vllm_fa2_C" in fields[5]:
                libraries[fields[5]] = None
        for name in libraries:
            libraries[name] = hashlib.sha256(Path(name).read_bytes()).hexdigest()
        return {
            "rank": torch.distributed.get_rank(),
            "pid": os.getpid(),
            "operators": available,
            "flags": flags,
            "libraries": libraries,
            "native_prefill_available": (
                all(available.values())
                and all(value == "1" for value in flags.values())
                and bool(libraries)
            ),
            "routes": dict(backend._route_counts),
            "gdn_prefill": {
                "original_tilelang": gdn._sm70_flashqla_original_prefill_enabled(),
                "indexed_state": gdn._sm70_flashqla_indexed_prefill_enabled(),
                "direct_output": gdn._sm70_flashqla_direct_output_enabled(),
                "explicit_original_flag": os.getenv(
                    "VLLM_SM70_FLASHQLA_ORIGINAL_PREFILL"
                ),
                "legacy_original_flag": os.getenv(
                    "FLASH_QLA_SM70_USE_ORIGINAL_TILELANG"
                ),
                "configuration_only_verify_actual_hit_in_worker_log": True,
            },
            "graph_route_counts_are_capture_counts": True,
        }
