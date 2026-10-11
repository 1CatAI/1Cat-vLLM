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


def prefill_observations(config):
    """Read resolved policy and initialized owners, without creating counters."""
    options = config.attention_config.flash_v100.options
    resources = getattr(config, "_runtime_resources", {})
    owner = resources.get("diagnostics")
    flags = {
        legacy: str(int(options.value(field)))
        for legacy, field in (
            ("VLLM_FLASH_V100_FP8_PREFILL_BRIDGE", "fp8_prefill_bridge"),
            ("VLLM_FLASH_V100_PREFILL_D256_GQA_V37", "prefill_d256_gqa_v37"),
            ("VLLM_FLASH_V100_FA2_D256_PREFILL", "fa2_d256_prefill"),
        )
    }
    architecture = (
        ("sm70_d256_gqa_v37_fwd",)
        if options.value("prefill_d256_gqa_v37")
        else (
            "sm70_d256_gqa_architecture_fwd",
            "sm70_d256_gqa_architecture_q8192_fwd",
        )
    )
    required = (
        "sm70_d256_splitd_n32_dense_fwd",
        "sm70_d256_splitd_n32_paged_fwd",
        *architecture,
    )
    prefill_owner = resources.get("sm70_prefill")
    return {
        "flags": flags,
        "required_operators": required,
        "routes": dict(owner.histories.get("flash_v100_routes", {}))
        if owner is not None
        else {},
        "route_counting_enabled": owner is not None
        and owner.trace.flash_v100.value("route_summary"),
        "prefill_owner": prefill_owner.explain() if prefill_owner is not None else None,
        "observation_scope": "host dispatches: capture included; graph replay excluded",
    }


class PrefillRouteAuditExtension:
    def cache_layout_snapshot(self):
        """Report actual scheduler-block geometry, without reading cache data."""
        import math

        import torch

        config = self.model_runner.kv_cache_config
        groups = [
            {
                "kind": type(group.kv_cache_spec).__name__,
                "block_tokens": group.kv_cache_spec.block_size,
                "physical_page_bytes": group.kv_cache_spec.page_size_bytes,
                "layers": list(group.layer_names),
            }
            for group in config.kv_cache_groups
        ]
        return {
            "rank": torch.distributed.get_rank(),
            "groups": groups,
            "block_alignment_tokens": math.lcm(
                *(row["block_tokens"] for row in groups)
            ),
            "mamba_cache_mode": self.vllm_config.cache_config.mamba_cache_mode,
            "scope": "allocation and configuration; does not establish prefix restore",
        }

    def kv_scale_snapshot(self):
        """Read the live scales without changing calibration or captured graphs."""
        import torch

        rows = []
        context = self.vllm_config.compilation_config.static_forward_context
        for name, layer in sorted(context.items()):
            codec = getattr(layer, "kv_cache_dtype", None)
            if codec is None:
                continue
            scales = {}
            for field in ("_q_scale", "_k_scale", "_v_scale"):
                value = getattr(layer, field, None)
                if isinstance(value, torch.Tensor):
                    value = value.detach().float().cpu()
                    scales[field] = {
                        "values": value.reshape(-1).tolist(),
                        "finite_positive": bool(torch.isfinite(value).all())
                        and bool((value > 0).all()),
                    }
            rows.append(
                {
                    "layer": name,
                    "codec": codec,
                    "calculation_pending": bool(
                        getattr(layer, "calculate_kv_scales", False)
                    ),
                    "scales": scales,
                }
            )
        return {
            "rank": torch.distributed.get_rank(),
            "layers": rows,
            "scope": "live scale values; does not establish calibration quality",
        }

    def dflash2_prefill_route_snapshot(self):
        import torch

        from vllm.config import set_current_vllm_config
        from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

        observations = prefill_observations(self.vllm_config)
        required = observations["required_operators"]
        available = {name: hasattr(torch.ops._vllm_fa2_C, name) for name in required}
        flags = observations["flags"]
        with set_current_vllm_config(self.vllm_config):
            gdn_prefill = {
                "original_tilelang": gdn._sm70_flashqla_original_prefill_enabled(),
                "indexed_state": gdn._sm70_flashqla_indexed_prefill_enabled(),
                "direct_output": gdn._sm70_flashqla_direct_output_enabled(),
                "configuration_only_verify_actual_hit_in_worker_log": True,
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
            **observations,
            "libraries": libraries,
            "native_prefill_available": (
                all(available.values())
                and flags["VLLM_FLASH_V100_FP8_PREFILL_BRIDGE"] == "1"
                and flags["VLLM_FLASH_V100_FA2_D256_PREFILL"] == "1"
                and bool(libraries)
            ),
            "gdn_prefill": gdn_prefill,
            "graph_route_counts_are_capture_counts": True,
        }
