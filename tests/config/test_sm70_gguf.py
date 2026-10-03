# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config.kernel import KernelConfig, Sm70GgufConfig
from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail
from vllm.sm70_profiles.acceleration import linear_policy_report, loaded_linear_kernels


def test_gguf_policy_is_enabled_and_part_of_the_graph_contract():
    default = KernelConfig()
    assert default.sm70_gguf.enabled
    assert default.sm70_gguf.prefill_min_m == 8
    disabled = KernelConfig(sm70_gguf=Sm70GgufConfig(enabled=False))
    assert default.compute_hash() != disabled.compute_hash()
    assert linear_policy_report(default)["sm70_gguf"]["configuration"]["enabled"]
    with pytest.raises(ValueError, match="positive"):
        Sm70GgufConfig(prefill_min_m=0)


def test_storage_tail_does_not_alias_the_next_projection():
    # K=640 Q2_0 rows consume 180 bytes; the upstream allocation guard
    # needs another 108 bytes. Those bytes may belong to a merged neighbour.
    backing = torch.full((16 * 180 + 108,), 255, dtype=torch.uint8)
    weight = backing[: 16 * 180].reshape(16, 180)
    padded = pad_weight_tail(weight, 42)
    assert padded.data_ptr() != weight.data_ptr()
    assert torch.equal(padded, weight)
    assert torch.all(backing == 255)
    storage = torch.empty(0, dtype=torch.uint8).set_(padded.untyped_storage())
    assert torch.all(storage[padded.numel() :] == 0)


def test_prepared_gguf_capability_is_exposed_by_existing_report():
    admission = {"enabled": False, "reason": "disabled_by_kernel_config"}
    layer = SimpleNamespace(quant_method=SimpleNamespace(native_admission=admission))
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.mlp", layer)])
    rows = loaded_linear_kernels(model)
    row = rows["GGUF:model.layers.0.mlp"]
    assert row["operator_admission"] == admission
    assert row["scope"] == "prepared_gguf_operator_capability"
