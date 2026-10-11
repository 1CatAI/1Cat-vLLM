# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref
from types import SimpleNamespace

import torch

from vllm.model_executor.kernels.linear.sm70_dense import DenseLinearState
from vllm.models.qwen4_exp.nvidia import sm70_fp16_hc as hc
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual


def test_hc_sharding_releases_borrowed_dense_checkpoint(monkeypatch):
    module = GatedResidual.__new__(GatedResidual)
    torch.nn.Module.__init__(module)
    module.hc_count, module.hidden_size, module.lora_rank = 4, 2560, 320
    module.use_combine = True
    references = []
    for name, shape in (
        ("input_mix_weight_down_block_inject", (336, 10240)),
        ("input_mix_weight_up", (10240, 320)),
    ):
        layer = torch.nn.Module()
        layer.register_parameter(
            "weight", torch.nn.Parameter(torch.ones(shape, dtype=torch.float16), False)
        )
        layer._hc_cpu_staging = True
        layer._sm70_dense_state = DenseLinearState(
            layer.weight, prefix=name, policy=None, trace=None
        )
        references.append(weakref.ref(layer.weight))
        module.add_module(name, layer)
    communicator = SimpleNamespace(
        status={"enabled": True}, logical_rank=0, device=torch.device("cpu")
    )
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group",
        lambda: SimpleNamespace(
            device_communicator=SimpleNamespace(hc_ll_comm=communicator)
        ),
    )
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(hc_weight_storage="sharded"), lora_config=None
    )
    hc.prepare_sharded_hc_storage(module, config)
    gc.collect()
    assert all(reference() is None for reference in references)
    assert not hasattr(layer, "_sm70_dense_state")
    down, up = hc._unpack_hc_storage(module.hc_shard_down, module.hc_shard_up)
    torch.testing.assert_close(down, torch.ones_like(down), atol=0, rtol=0)
    torch.testing.assert_close(up, torch.ones_like(up), atol=0, rtol=0)
