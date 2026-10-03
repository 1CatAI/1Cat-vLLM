# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Historical DFlash2 policy and actual GDN admission predicates, on CPU."""

import ast
import itertools
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any
from unittest.mock import patch

import torch

from vllm import envs
from vllm.config.sm70_dflash2 import (
    SM70_DFLASH2_LEGACY_FIELDS,
    SM70_DFLASH2_VERIFIER_DEFAULTS,
    Sm70DFlash2Config,
    capture_sm70_dflash2_config,
    sm70_dflash2_enabled,
)
from vllm.model_executor.models.config import sm70_dflash2_verifier_qualified
from vllm.v1.worker.gpu.spec_decode import uses_dflash_selector_engine

GDN_SOURCE = "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
GUARDS = (
    "enable_sm70_dflash2_fused_gdn_verify",
    "enable_sm70_dflash2_fused_gdn_norm",
    "enable_sm70_dflash2_fused_gdn_split",
    "enable_sm70_dflash2_fused_gdn_combined_split",
)


def _read(ref, path):
    return (
        subprocess.check_output(["git", "show", f"{ref}:{path}"], text=True)
        if ref
        else Path(path).read_text()
    )


def _assignment(source, name):
    return next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        )
    )


def _guard_expressions(source):
    tree = ast.parse(source)
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "QwenGatedDeltaNetAttention"
    )
    init = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    statements = []
    for statement in init.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr in GUARDS
                for target in statement.targets
            )
            or (
                isinstance(statement, ast.If)
                and isinstance(statement.test, ast.Attribute)
                and statement.test.attr in GUARDS
                and any(isinstance(node, ast.Assign) for node in ast.walk(statement))
            )
        ):
            statements.append(statement)
    return compile(
        ast.Module(body=statements, type_ignores=[]), "<GDN admission>", "exec"
    )


def load_baseline(ref, directory=None):
    source = _read(ref, "vllm/config/vllm.py")
    defaults_node = _assignment(source, "_SM70_DFLASH2_VERIFIER_DEFAULTS")
    policy_source = None
    if not isinstance(defaults_node, ast.Dict):
        policy_source = _read(ref, "vllm/config/sm70_dflash2.py")
        defaults_node = _assignment(policy_source, "SM70_DFLASH2_VERIFIER_DEFAULTS")
        source = _read(ref, "vllm/model_executor/models/config.py")
        function_name = "sm70_dflash2_verifier_qualified"
    else:
        function_name = "_is_sm70_dflash2_verifier_contract"
    defaults = ast.literal_eval(defaults_node)
    qualification = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == function_name
    )
    namespace = {"torch": torch, "Mapping": Mapping, "Any": Any}
    exec(
        compile(
            ast.Module(body=[qualification], type_ignores=[]),
            "<historical qualification>",
            "exec",
        ),
        namespace,
    )
    from tools.pre_commit.check_env_metadata import registrations

    getters = {
        name: eval(
            compile(ast.Expression(value.args[0]), "<historical getter>", "eval"),
            {"os": os},
        )
        for name, value in registrations(_read(ref, "vllm/envs.py")).items()
        if name in defaults
    }
    policy_class = None
    if policy_source is not None:
        import sys
        from types import ModuleType

        module = ModuleType("_historical_dflash_policy_" + ref[:12])
        sys.modules[module.__name__] = module
        exec(compile(policy_source, "<historical policy>", "exec"), vars(module))
        module.envs = NS(environment_variables=getters)
        policy_class = module.Sm70DFlash2Config
    vocab = ast.parse(
        _read(ref, "vllm/model_executor/layers/vocab_parallel_embedding.py")
    )
    order = next(
        n
        for n in vocab.body
        if isinstance(n, ast.FunctionDef) and n.name == "_sm70_dflash2_use_dense_order"
    )
    return NS(
        defaults=defaults,
        getters=getters,
        qualify=namespace[function_name],
        policy_class=policy_class,
        order=compile(
            ast.Module(body=[order], type_ignores=[]), "<historical tie order>", "exec"
        ),
        guards=_guard_expressions(_read(ref, GDN_SOURCE)),
    )


def make_config(model, kv, tp, spec, concurrency, budget):
    hidden = {
        "27b_dflash2_nvfp4": 5120,
        "flash_next_mtp4_nvfp4": 2560,
        "35b_a3b_awq": 2048,
    }[model]
    config = NS(
        model_config=NS(
            architectures=[
                "Qwen4ExpForConditionalGeneration"
                if "flash_next" in model
                else "Qwen3_5ForConditionalGeneration"
            ],
            dtype=torch.float16,
            hf_text_config=NS(
                hidden_size=hidden,
                num_attention_heads=24,
                num_key_value_heads=4,
                head_dim=256,
            ),
        ),
        parallel_config=NS(
            tensor_parallel_size=tp,
            pipeline_parallel_size=1,
            enable_dbo=False,
            ubatch_size=0,
        ),
        speculative_config=None
        if spec == "none"
        else NS(
            method=spec,
            num_speculative_tokens=4 if spec == "mtp" else 7,
            draft_model_config=NS(hf_config=NS(dflash_config={"selector_top_k": 16})),
            sm70_dflash2=Sm70DFlash2Config(),
        ),
        cache_config=NS(cache_dtype=kv),
        scheduler_config=NS(max_num_seqs=concurrency, max_num_batched_tokens=budget),
    )
    return config


def snapshot(baseline=None):
    guards = (
        baseline.guards if baseline else _guard_expressions(_read(None, GDN_SOURCE))
    )
    defaults = baseline.defaults if baseline else SM70_DFLASH2_VERIFIER_DEFAULTS
    clean = {
        key: value for key, value in os.environ.items() if not key.startswith("VLLM_")
    }

    def row(values, overrides=None, gdn_heads=(16, 48)):
        model, kv, tp, spec, concurrency, budget = values
        cfg = make_config(*values)
        with patch.dict(os.environ, {**clean, **(overrides or {})}, clear=True):
            envs.disable_envs_cache()
            qualify = baseline.qualify if baseline else sm70_dflash2_verifier_qualified
            qualified = qualify(
                cfg.model_config, cfg.speculative_config, cfg.parallel_config
            )
            if baseline and baseline.policy_class is not None:
                policy = baseline.policy_class()
                policy.resolve(qualified=qualified)
                if cfg.speculative_config is not None:
                    cfg.speculative_config.sm70_dflash2 = policy
                # Unconfigured layers retain the original standalone getter.
                states = {
                    name: bool(getattr(policy, field))
                    if cfg.speculative_config is not None
                    else getter()
                    for name, getter in baseline.getters.items()
                    for field in [
                        name.removeprefix("VLLM_SM70_DFLASH2_").lower()
                        if name != "VLLM_SM70_FP8_QPN8"
                        else "target_fp8_qpn8"
                    ]
                }
                legacy_env = NS(**states)
            elif baseline:
                if qualified:
                    for name, value in defaults.items():
                        os.environ.setdefault(name, value)
                states = {name: getter() for name, getter in baseline.getters.items()}
                legacy_env = NS(**states)
            else:
                before = dict(os.environ)
                policy = capture_sm70_dflash2_config(cfg)
                if policy is not None:
                    policy.resolve(qualified=qualified)
                assert before == dict(os.environ), (
                    "Resolution mutated process environment"
                )
                states = {
                    name: sm70_dflash2_enabled(field, policy)
                    for name, field in SM70_DFLASH2_LEGACY_FIELDS.items()
                }
                legacy_env = envs
            namespace = dict(
                envs=legacy_env,
                current_platform=NS(is_device_capability=lambda cc: True),
                _is_dflash2_spec_config=uses_dflash_selector_engine,
                capture_sm70_dflash2_config=capture_sm70_dflash2_config,
                sm70_dflash2_enabled=sm70_dflash2_enabled,
                vllm_config=cfg,
                torch=torch,
                self=NS(
                    key_dim=gdn_heads[0] * 128,
                    value_dim=gdn_heads[1] * 128,
                    num_v_heads=gdn_heads[1],
                    tp_size=tp,
                    hidden_size=cfg.model_config.hf_text_config.hidden_size,
                ),
            )
            exec(guards, namespace)
            admissions = {
                name: bool(getattr(namespace["self"], name)) for name in GUARDS
            }
            if baseline:
                from vllm.logger import init_logger

                order_namespace = {
                    "_sm70_dflash2_option": lambda field, layer=None: states[
                        "VLLM_SM70_DFLASH2_" + field.upper()
                    ],
                    "logger": init_logger(__name__),
                }
                exec(baseline.order, order_namespace)
                dense_order = order_namespace["_sm70_dflash2_use_dense_order"]()
            else:
                from vllm.model_executor.layers.vocab_parallel_embedding import (
                    _sm70_dflash2_use_dense_order,
                )

                dense_order = _sm70_dflash2_use_dense_order()
            states = {
                name: value
                for name, value in states.items()
                if name in SM70_DFLASH2_LEGACY_FIELDS
            }
            label = f"{model}/{kv}/tp{tp}/{spec}/c{concurrency}/budget{budget}"
            if gdn_heads != (16, 48):
                label += "/gdn-k8-v24"
            if overrides:
                label += "/" + str(sorted(overrides.items()))
            return {
                "config": label,
                "qualified": qualified,
                "policy": states,
                "dense_tie_order": dense_order,
                "gdn_layer_admission": admissions,
            }

    matrix = itertools.product(
        ("27b_dflash2_nvfp4", "flash_next_mtp4_nvfp4", "35b_a3b_awq"),
        ("float16", "fp8_e4m3", "fp8_e5m2"),
        (2, 4),
        ("none", "mtp", "dflash"),
        (1, 4, 8),
        (4096, 8192),
    )
    cases = [row(values) for values in matrix]
    selected = ("27b_dflash2_nvfp4", "fp8_e4m3", 4, "dflash", 4, 8192)
    edges = [
        row(selected, {name: value})
        for name in (
            *SM70_DFLASH2_VERIFIER_DEFAULTS,
            "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER",
            "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER",
        )
        for value in ("0", "1", "2", "-1")
    ]
    edges.append(
        row(("27b_dflash2_nvfp4", "fp8_e4m3", 2, "dflash", 1, 8192), gdn_heads=(8, 24))
    )
    edges.append(
        row(
            selected,
            {
                "VLLM_SM70_DFLASH2_QPN8_DENSE_ORDER": "0",
                "VLLM_SM70_DFLASH2_QPN8_ALLOW_CANDIDATE_ORDER": "1",
            },
        )
    )
    envs.disable_envs_cache()
    return {"cases": cases, "edge_cases": edges}
