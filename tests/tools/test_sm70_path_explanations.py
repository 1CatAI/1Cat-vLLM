# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import pytest
import regex as re
import torch

from tools.sm70.explain import explain_linear, explain_moe_plan
from tools.sm70.native_trace import NativeDispatchTrace
from vllm.config.kernel import KernelConfig
from vllm.config.sm70_moe import Sm70MxFp4MoEConfig, Sm70NvFp4MoEConfig
from vllm.config.sm70_native import NATIVE_FIELDS
from vllm.model_executor.layers.fused_moe.sm70.declarations import (
    FP4_STAGE_BINDINGS,
)
from vllm.model_executor.layers.quantization.sm70_moe_router import (
    Sm70MoeStageRoute,
    select_fp4_stage_plan,
)

pytestmark = pytest.mark.cpu_test


def test_python_cpp_native_policy_abi_order_and_calculation_fields_match():
    path = Path(__file__).resolve().parents[2] / "csrc/sm70_policy_fields.inc"
    entries = re.findall(
        r'SM70_POLICY_FIELD\(\s*(\w+),\s*"([^"]+)",\s*(true|false)\s*\)',
        path.read_text(),
    )
    assert entries == [
        (field, alias, "false" if diagnostic else "true")
        for field, alias, _, diagnostic in NATIVE_FIELDS
    ]


@pytest.mark.parametrize("family,stage,mode", list(FP4_STAGE_BINDINGS))
def test_fp4_explanation_uses_the_executors_binding(family, stage, mode):
    effective = mode.removesuffix("_raw").removesuffix("_mtp")
    plan = select_fp4_stage_plan(
        Sm70MoeStageRoute(effective if stage == "w13" else "dense"),
        Sm70MoeStageRoute(effective if stage == "w2" else "dense"),
        qpn_mtp=mode.endswith("_mtp"),
    )
    policy = Sm70NvFp4MoEConfig() if family == "nvfp4" else Sm70MxFp4MoEConfig()
    policy.resolve()
    report = explain_moe_plan(family, plan, policy, raw_scale=mode.endswith("_raw"))
    selected = next(row for row in report["stages"] if row["stage"] == stage)
    assert selected["predicted_operator"] == FP4_STAGE_BINDINGS[family, stage, mode][0]
    assert selected["layout"] == FP4_STAGE_BINDINGS[family, stage, mode][2]
    assert report["observed_execution"] is None


def test_linear_explanation_has_sources_and_observations_do_not_salt_hash(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_AWQ_PREFILL_EXACT_DENSE", "0")
    config = KernelConfig()
    config.sm70_awq.prefill_exact_dense = True
    config.sm70_awq.resolve()
    config.sm70_awq.native.resolve("awq")
    before = config.compute_hash()
    config.linear_kernel_selections["layer"] = {"selected": "TurboMind"}
    report = explain_linear(config, "awq", observed={"records": []})
    record = next(
        r for r in report["parameters"] if r["parameter"] == "prefill_exact_dense"
    )
    assert record["value"] is True and record["source"] == "configuration"
    config.sm70_awq.sources["prefill_exact_dense"] = "diagnostic annotation"
    assert config.compute_hash() == before
    assert report["admission_and_fallback"]["layer"]["selected"] == "TurboMind"


def test_native_trace_does_not_present_meta_dispatch_as_gpu_execution():
    library = torch.library.Library("_C", "FRAGMENT")
    library.define("phase_b_trace_probe(Tensor x) -> Tensor")
    library.impl(
        "phase_b_trace_probe", lambda x: x.clone(), "CompositeExplicitAutograd"
    )
    try:
        with NativeDispatchTrace("capture") as trace:
            torch.ops._C.phase_b_trace_probe(torch.empty(2, 3, device="meta"))
        assert trace.records[0]["evidence"] == "fake_dispatch"
        assert trace.records[0]["phase"] == "capture"
        assert "not replay" in trace.report()["completion"]
    finally:
        library._destroy()
