# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.config import KernelConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.platforms import current_platform
from vllm.runtime_resources import release_runtime_resources
from vllm.scalar_type import scalar_types

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70), reason="SM70 native Marlin"
)


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("rows", [1, 8, 33])
def test_marlin_engine_policy_matches_legacy_and_survives_replay(
    monkeypatch, moe, rows
):
    torch.manual_seed(319)
    n, k = 512, 1024
    x = torch.randn(rows, k, device="cuda", dtype=torch.float16)
    _, weight, scales = rand_marlin_weight_mxfp4_like(
        torch.empty(n, k, device="cuda", dtype=torch.float16), 32
    )
    ids = torch.zeros(rows, 1, device="cuda", dtype=torch.long)
    sorted_ids, expert_ids, padded = moe_align_block_size(ids, 8, 1)
    scores = torch.ones(rows, 1, device="cuda", dtype=torch.float32)
    output = torch.empty(rows, n, device="cuda", dtype=torch.float16)
    legacy_workspace = marlin_make_workspace_new(x.device)

    def run(workspace):
        if moe:
            return ops.moe_wna16_marlin_gemm(
                x,
                output,
                weight.unsqueeze(0),
                None,
                scales.unsqueeze(0),
                None,
                None,
                None,
                None,
                None,
                workspace,
                sorted_ids,
                expert_ids,
                padded,
                scores,
                8,
                1,
                False,
                scalar_types.float4_e2m1f,
                rows,
                n,
                k,
                True,
                False,
                True,
                False,
            )
        return ops.marlin_gemm(
            x,
            output,
            weight,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            scalar_types.float4_e2m1f,
            rows,
            n,
            k,
            True,
            False,
            True,
            False,
        )

    prefix = "SM70_MARLIN_" + ("MOE" if moe else "DENSE")
    prepared = []
    for split in (1, 8):
        monkeypatch.setenv(prefix + "_CTA_GEOMETRY", "")
        monkeypatch.setenv(prefix + "_SPLIT_K", str(split))
        monkeypatch.setenv(prefix + "_METADATA_CACHE", "lane_vectors")
        reference = run(legacy_workspace).clone()
        cfg = SimpleNamespace(kernel_config=KernelConfig())
        cfg.kernel_config.capture_provider_inputs()
        with set_current_vllm_config(cfg):
            workspace = marlin_make_workspace_new(x.device)
        owner = cfg._runtime_resources["sm70_marlin_binding"].owner
        prepared.append((cfg, owner, workspace, reference))
    for suffix in ("_CTA_GEOMETRY", "_SPLIT_K", "_METADATA_CACHE"):
        monkeypatch.setenv(prefix + suffix, "invalid-after-init")
    for cfg, owner, workspace, reference in prepared + prepared[::-1]:
        with owner.activate():
            assert torch.equal(run(workspace), reference)
    cfg, owner, workspace, _ = prepared[1]
    with owner.activate():
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(workspace)
        x.mul_(0.5)
        expected = run(workspace).clone()
        graph.replay()
        assert torch.equal(captured, expected)
    release_runtime_resources(prepared[0][0])
    with owner.activate():
        graph.replay()
        assert torch.equal(captured, expected)
    graph.reset()
    release_runtime_resources(cfg)


def test_marlin_invalid_bound_override_preserves_native_error(monkeypatch):
    from tests.kernels.quantization.test_sm70_mxfp4_e8m0 import _make_fixed_e8m0_case

    x, weight, scales, _ = _make_fixed_e8m0_case(monkeypatch, 112)
    workspace = marlin_make_workspace_new(x.device)

    def run():
        return ops.marlin_gemm(
            x,
            None,
            weight,
            None,
            scales,
            None,
            None,
            None,
            None,
            None,
            workspace,
            scalar_types.float4_e2m1f,
            1,
            512,
            1024,
        )

    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", "3")
    with pytest.raises(RuntimeError) as old:
        run()
    cfg = SimpleNamespace(kernel_config=KernelConfig())
    cfg.kernel_config.capture_provider_inputs()
    with set_current_vllm_config(cfg):
        workspace = marlin_make_workspace_new(x.device)
    owner = cfg._runtime_resources["sm70_marlin_binding"].owner
    monkeypatch.setenv("SM70_MARLIN_DENSE_SPLIT_K", "1")
    with owner.activate(), pytest.raises(RuntimeError) as new:
        run()
    assert str(new.value) == str(old.value)
    release_runtime_resources(cfg)
