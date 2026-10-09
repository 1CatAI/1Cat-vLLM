# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Initialization-only bridge from ordered legacy defaults to typed owners.

The bridge never modifies the process environment. Workers receive the resolved
owner objects; execution code does not use this alias-indexed adapter.
"""

import os
from typing import Any

from vllm import envs
from vllm.config.execution_policy import POLICY_OWNERS, read_execution_legacy
from vllm.config.sm70_dflash2 import SM70_DFLASH2_LEGACY_FIELDS


def _owner(cfg, path):
    for part in path.split("."):
        cfg = getattr(cfg, part, None)
        if cfg is None:
            break
    return cfg


class PolicyDefaults:
    def __init__(self, cfg):
        self.cfg = cfg
        self.phase = "platform.runtime"
        self.raw = dict(os.environ)
        self.bindings: dict[str, list[tuple[Any, str]]] = {}
        for path in POLICY_OWNERS:
            policy = _owner(cfg, path)
            for field, alias in policy.aliases.items():
                self.bindings[alias] = [(policy, field)]
        spec = _owner(cfg, "speculative_config.sm70_dflash2")
        if spec is not None:
            aliases = dict(SM70_DFLASH2_LEGACY_FIELDS)
            aliases.update(
                {
                    "VLLM_SM70_DFLASH2_BF16_EMULATION": "bf16_emulation",
                    "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE": (
                        "proposal_temperature_scale"
                    ),
                    "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P": "proposal_top_p",
                }
            )
            self.bindings.update(
                {name: [(spec, field)] for name, field in aliases.items()}
            )
        for name, paths in EXTRA_BINDINGS.items():
            self.bindings[name] = [
                (_owner(cfg, path.rsplit(".", 1)[0]), path.rsplit(".", 1)[1])
                for path in paths
            ]

    def _source(self, policy, field):
        source = getattr(policy, "sources", {}).get(field)
        if source is not None:
            return source
        return "typed" if getattr(policy, field) is not None else "default"

    def __contains__(self, name):
        bindings = self.bindings.get(name)
        if not bindings:
            return name in self.raw
        return self._source(*bindings[0]) != "default" or name in self.raw

    def value(self, name):
        bindings = self.bindings.get(name)
        if not bindings:
            return envs.environment_variables[name]()
        policy, field = bindings[0]
        value = getattr(policy, field)
        if value is None:
            value = read_execution_legacy(name)
            setattr(policy, field, value)
            if hasattr(policy, "sources"):
                policy.sources[field] = name if name in self.raw else "default"
        return value

    def __getitem__(self, name):
        if name not in self.bindings:
            return self.raw[name]
        value = self.value(name)
        return str(int(value)) if isinstance(value, bool) else str(value)

    def get(self, name, default=None):
        if name in self:
            return self[name]
        return default

    def __setitem__(self, name, raw):
        for policy, field in self.bindings[name]:
            # A typed choice can differ between providers sharing a legacy alias.
            if self._source(policy, field) == "typed":
                continue
            annotation = type(policy).__dataclass_fields__[field].type
            annotation = str(annotation)
            value: Any
            if "bool" in annotation:
                value = bool(int(raw))
            elif "int" in annotation:
                value = int(raw)
            elif "float" in annotation:
                value = float(raw)
            else:
                value = raw
            setattr(policy, field, value)
            if hasattr(policy, "sources"):
                policy.sources[field] = f"default:{self.phase}"
            self.cfg.runtime_default_sources.setdefault(name, []).append(
                {
                    "source": f"default:{self.phase}",
                    "value": value,
                }
            )

    def setdefault(self, name, value):
        if name not in self:
            self[name] = value
        return self[name]

    def finish(self):
        from vllm.config.sm70_runtime import resolve_legacy_fields

        # Two legacy defaults depended on the compile-graph flag. Typed inputs
        # follow the same order without writing that flag into os.environ.
        if self.value("VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH"):
            self.setdefault("VLLM_USE_AOT_COMPILE", "1")
            self.setdefault("VLLM_SM70_LM_HEAD_TOP1", "0")
        for path in POLICY_OWNERS:
            policy = _owner(self.cfg, path)
            policy.resolve()
        graph = self.cfg.compilation_config.runtime
        if graph.sources.get("mega_aot") == "default":
            from vllm.utils.torch_utils import is_torch_equal_or_newer

            graph.mega_aot = bool(
                graph.aot_compile and is_torch_equal_or_newer("2.12.0.dev")
            )
        for name in EXTRA_BINDINGS:
            for policy, field in self.bindings[name]:
                resolve_legacy_fields(
                    policy, {field: name}, reader=read_execution_legacy
                )
        for name, bindings in self.bindings.items():
            policy, field = bindings[0]
            self.cfg.runtime_default_sources.setdefault(name, []).append(
                {
                    "source": self._source(policy, field),
                    "value": getattr(policy, field),
                }
            )


EXTRA_BINDINGS = {
    "VLLM_SM70_AWQ_WARMUP_MAX_M": ("kernel_config.sm70_runtime.awq_warmup_max_m",),
    "VLLM_SM70_FP8_DENSE_TUNE_MAX_M": (
        "kernel_config.sm70_fp8.native.fp8_dense_tune_max_m",
        "kernel_config.sm70_moe.fp8.native.fp8_dense_tune_max_m",
    ),
    "VLLM_SM70_NVFP4_DENSE_TUNE_MAX_M": (
        "kernel_config.sm70_nvfp4.native.nvfp4_dense_tune_max_m",
        "kernel_config.sm70_moe.nvfp4.native.nvfp4_dense_tune_max_m",
    ),
    "VLLM_SM70_GLM53_MOE_QPN_W13_Q8": ("kernel_config.sm70_moe.nvfp4.glm53_qpn_w13",),
    "VLLM_SM70_NVFP4_MOE_GROUPED_EXPERT_ROWS": (
        "kernel_config.sm70_moe.nvfp4.grouped_expert_rows",
        "kernel_config.sm70_nvfp4.native.nvfp4_moe_grouped_expert_rows",
    ),
}


def finalize_runtime_policy_hashes(cfg):
    """Fingerprint computation, excluding diagnostics and unused feature knobs."""
    from vllm.model_executor.models.runtime_defaults import (
        _is_sm70_qwen38_decode_compile_contract,
    )
    from vllm.platforms.runtime_defaults import _any_participating_device_is_pre_ampere

    graph = cfg.compilation_config.runtime
    pre_ampere = _any_participating_device_is_pre_ampere(cfg)
    spec = cfg.speculative_config
    graph_fields = ["aot_compile", "breakable", "mega_aot"]
    if pre_ampere:
        graph_fields.append("compile_graph")
        if _is_sm70_qwen38_decode_compile_contract(
            cfg.model_config, spec, cfg.parallel_config
        ):
            graph_fields.append("dual_compile")
        if getattr(spec, "method", None) == "mtp":
            graph_fields.append("split_draft_graphs")
    graph.hash_fields = tuple(graph_fields)
    layers = cfg.kernel_config.layer_execution
    layers.hash_fields = (
        tuple(layers.aliases) if pre_ampere else ("shared_moe_overlap",)
    )
    tp = cfg.parallel_config.tensor_parallel_size
    comm = cfg.parallel_config.communication
    comm.hash_fields = tuple(
        field
        for field in comm.aliases
        if (
            (
                field == "pp_layer_partition"
                and cfg.parallel_config.pipeline_parallel_size > 1
            )
            or (pre_ampere and field == "tp4_push" and tp == 4)
            or (pre_ampere and field in ("tp8_hierarchical", "tp8_push") and tp == 8)
            or (pre_ampere and field == "moe_add_allreduce" and tp > 1)
        )
    )
    text = getattr(cfg.model_config, "hf_text_config", None)
    cfg.offload_config.ple.active = bool(getattr(text, "ple_layer_ids", None))
    attention = cfg.attention_config
    backend_name = getattr(attention.backend, "name", attention.backend)
    attention.flash_v100.active = pre_ampere and backend_name in (
        None,
        "FLASH_ATTN_V100",
        "FLASHINFER_SM70",
    )


def runtime_policy_report(cfg):
    """Explain typed values and their initialization trace; not a native hit log."""
    return {
        "evidence": "resolved_configuration",
        "owners": {
            path: {
                "values": {
                    field: getattr(_owner(cfg, path), field)
                    for field in _owner(cfg, path).aliases
                },
                "sources": dict(_owner(cfg, path).sources),
                "hash_fields": (
                    list(_owner(cfg, path).hash_fields)
                    if _owner(cfg, path).hash_fields is not None
                    else None
                ),
                "active": _owner(cfg, path).active,
            }
            for path in POLICY_OWNERS
        },
        "default_resolution": cfg.runtime_default_sources,
    }


def effective_runtime_values(cfg):
    values = {}
    for path in POLICY_OWNERS:
        policy = _owner(cfg, path)
        values.update(
            {alias: getattr(policy, field) for field, alias in policy.aliases.items()}
        )
    for alias, paths in EXTRA_BINDINGS.items():
        values[alias] = _owner(cfg, paths[0])
    return values


def runtime_compile_ignored_aliases(cfg) -> set[str]:
    """Resolved owner hashes replace their legacy inputs, including provenance.

    Standalone callers retain environment hashing. Do not drop aliases for an
    unresolved owner: a plugin may compile before the normal init checkpoint.
    """
    ignored: set[str] = set()
    for path in POLICY_OWNERS:
        policy = _owner(cfg, path)
        ignored.update(
            alias for field, alias in policy.aliases.items() if field in policy.sources
        )
    ignored.update(
        alias
        for alias, paths in EXTRA_BINDINGS.items()
        if all(_owner(cfg, path) is not None for path in paths)
    )
    spec = _owner(cfg, "speculative_config.sm70_dflash2")
    if spec is not None and spec.resolved:
        ignored.update(SM70_DFLASH2_LEGACY_FIELDS)
        ignored.update(
            (
                "VLLM_SM70_DFLASH2_BF16_EMULATION",
                "VLLM_SM70_DFLASH2_PROPOSAL_TEMPERATURE_SCALE",
                "VLLM_SM70_DFLASH2_PROPOSAL_TOP_P",
            )
        )
    return ignored
