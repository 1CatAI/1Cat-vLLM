# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.linear.mixed_precision import sm70_silu_down as op
from vllm.scalar_type import scalar_types


def config(**kwargs):
    values = dict(
        full_weight_shape=(160, 2560),
        partition_weight_shape=(160, 2560),
        weight_type=scalar_types.float16,
        act_type=torch.float16,
        group_size=-1,
        zero_points=False,
        has_g_idx=False,
        out_type=torch.float16,
    )
    values.update(kwargs)
    return op.Sm70SiluDownConfig(**values)


def test_quantized_and_other_precision_configs_are_rejected(monkeypatch):
    monkeypatch.setattr(op.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", False
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    assert op.Sm70SiluDownKernel.can_implement(config())[0]
    for changed in (
        dict(act_type=torch.bfloat16),
        dict(out_type=torch.float32),
        dict(zero_points=True),
        dict(group_size=32),
    ):
        assert not op.Sm70SiluDownKernel.can_implement(config(**changed))[0]


def test_measured_m5_admitted_unmeasured_width_keeps_visible_fallback(monkeypatch):
    import vllm.config

    cfg = SimpleNamespace(
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[1, 5, 17]),
        kernel_config=SimpleNamespace(linear_kernel_selections={}),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: cfg)
    monkeypatch.setattr(op, "_measure", lambda weight, width: dict(accepted=width == 5))
    kernel = op.Sm70SiluDownKernel(config(), "weight", "")
    layer = SimpleNamespace(
        weight=torch.zeros(97, 17, dtype=torch.float16), prefix="operator"
    )
    kernel.process_weights_after_loading(layer)
    assert kernel.widths == [5]
    assert (
        kernel.apply_silu_down(layer, torch.zeros(7, 34, dtype=torch.float16)) is None
    )
    calls = []

    def fused(x, weight):
        calls.append(tuple(x.shape))
        return x.new_zeros(x.shape[0], weight.shape[0])

    monkeypatch.setattr(torch.ops.vllm, "sm70_fp16_silu_down", fused)
    assert kernel.apply_silu_down(
        layer, torch.zeros(5, 34, dtype=torch.float16)
    ).shape == (5, 97)
    assert calls == [(5, 34)]
