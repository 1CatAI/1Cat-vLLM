# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from vllm import envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.kernels import linear
from vllm.model_executor.kernels.linear import pre_ampere_qpn as qpn
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
)
from vllm.platforms import PlatformEnum
from vllm.platforms.interface import DeviceCapability

pytestmark = pytest.mark.cpu_test


@pytest.fixture(autouse=True)
def engine(monkeypatch):
    for name in (
        "VLLM_SM70_QUANT_BACKEND",
        "VLLM_LINEAR_BACKEND",
        "VLLM_SM70_FP8_TURBOMIND",
        "VLLM_SM70_NVFP4_TURBOMIND",
        "VLLM_DISABLED_KERNELS",
        "VLLM_NVFP4_GEMM_BACKEND",
    ):
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    platform = NS(
        _enum=PlatformEnum.CUDA,
        is_cuda=lambda: True,
        get_device_capability=lambda *args: DeviceCapability(7, 5),
    )
    monkeypatch.setattr(linear, "current_platform", platform)
    monkeypatch.setattr(qpn, "current_platform", platform)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 2)
    monkeypatch.setattr(
        torch.ops,
        "_C",
        NS(
            nvfp4_qpn2_prepare_sm70=True,
            nvfp4_qpn2_gemm_sm70_out=True,
            fp8_qpn8_prepare_sm70=True,
            fp8_qpn8_dispatch_sm70_out=True,
            fp8_qpn8_prefill_sm70_out=True,
        ),
    )
    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config
    envs.disable_envs_cache()


def nvfp4_config(n=40, k=384):
    return qpn.TuringNvFp4LinearLayerConfig(
        input_dtype=torch.float16,
        weight_shape=(n, k // 2),
        scale_shape=(n, k // 16),
        weight_dtype=torch.uint8,
        scale_dtype=torch.float8_e4m3fn,
    )


def fp8_config(n=96, k=384):
    return linear.FP8ScaledMMLinearLayerConfig(
        weight_quant_key=kFp8StaticTensorSym,
        activation_quant_key=kFp8StaticTensorSym,
        weight_shape=(n, k),
        input_dtype=torch.float16,
        out_dtype=torch.float16,
    )


def test_real_selectors_prefer_native_turing_kernels():
    assert isinstance(
        linear.init_nvfp4_linear_kernel(nvfp4_config()),
        qpn.TuringQpn2NvFp4LinearKernel,
    )


def test_modelopt_selects_with_metadata_for_explicit_provider(engine, monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers.quantization import modelopt

    engine.kernel_config.linear_backend = "turbomind"
    engine.model_config = NS(dtype=torch.float16)
    monkeypatch.setattr(modelopt.sm70_tm, "is_turing_cuda_platform", lambda: True)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    method = modelopt.ModelOptNvFp4LinearMethod(
        NS(is_checkpoint_nvfp4_serialized=True, group_size=16)
    )
    assert method.kernel is None
    layer = torch.nn.Module()
    method.create_weights(layer, 128, [40], 128, 40, torch.float16)
    assert isinstance(method.kernel, qpn.TuringQpn2NvFp4LinearKernel)


def test_fp8_real_selector_prefers_native_turing_kernel():
    assert (
        linear.choose_scaled_mm_linear_kernel(
            fp8_config(),
            linear._POSSIBLE_FP8_KERNELS,
            compute_capability=75,
        )
        is qpn.TuringQpn8Fp8LinearKernel
    )
    assert isinstance(
        linear.init_fp8_linear_kernel(
            activation_quant_key=kFp8StaticTensorSym,
            weight_quant_key=kFp8StaticTensorSym,
            weight_shape=(96, 384),
            input_dtype=torch.float16,
            out_dtype=torch.float16,
        ),
        qpn.TuringQpn8Fp8LinearKernel,
    )


@pytest.mark.parametrize("n,k", [(1, 128), (40, 384), (97, 768), (1024, 2048)])
def test_nvfp4_admits_padding_and_aligned_shapes(n, k):
    assert qpn.TuringQpn2NvFp4LinearKernel.can_implement(nvfp4_config(n, k))[0]


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"input_dtype": torch.bfloat16}, "FP16"),
        ({"weight_shape": (40, 24)}, "128"),
        ({"scale_shape": (40, 1)}, "16 weight"),
        ({"scale_dtype": torch.float32}, "E4M3"),
        ({"weight_shape": (40, 24, 1)}, "rank"),
    ],
)
def test_nvfp4_rejections_explain_fallback(change, reason):
    accepted, why = qpn.TuringQpn2NvFp4LinearKernel.can_implement(
        replace(nvfp4_config(), **change)
    )
    assert not accepted and reason in why


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"out_dtype": torch.bfloat16}, "FP16"),
        ({"weight_shape": (31, 128)}, "32"),
        ({"weight_shape": (32, 16)}, "128"),
        ({"weight_quant_key": kFp8Static128BlockSym}, "per-tensor"),
    ],
)
def test_fp8_rejections_explain_fallback(change, reason):
    accepted, why = qpn.TuringQpn8Fp8LinearKernel.can_implement(
        replace(fp8_config(), **change)
    )
    assert not accepted and reason in why


def test_kernelconfig_and_legacy_disable(engine, monkeypatch):
    engine.kernel_config.sm70_nvfp4.dense_qpn2 = False
    assert (
        "KernelConfig"
        in qpn.TuringQpn2NvFp4LinearKernel.can_implement(nvfp4_config())[1]
    )
    engine.kernel_config.sm70_fp8.enabled = False
    assert not qpn.TuringQpn8Fp8LinearKernel.can_implement(fp8_config())[0]
    engine.kernel_config.sm70_nvfp4.dense_qpn2 = True
    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "0")
    assert (
        "override" in qpn.TuringQpn2NvFp4LinearKernel.can_implement(nvfp4_config())[1]
    )


def test_native_missing_and_worker_device(monkeypatch):
    asked = []

    def worker_capability(device):
        asked.append(device)
        return DeviceCapability(7, 5)

    monkeypatch.setattr(
        qpn.current_platform,
        "get_device_capability",
        worker_capability,
    )
    assert qpn.TuringQpn2NvFp4LinearKernel.is_supported()[0]
    assert asked == [2]
    assert not qpn.TuringQpn2NvFp4LinearKernel.is_supported(70)[0]
    assert not qpn.TuringQpn8Fp8LinearKernel.is_supported(72)[0]
    monkeypatch.setattr(torch.ops, "_C", NS())
    assert (
        "missing native" in qpn.TuringQpn8Fp8LinearKernel.can_implement(fp8_config())[1]
    )


def test_fp8_scratch_is_owned_by_each_invocation(monkeypatch):
    from vllm import _sm70_ops as ops

    allocations = []
    dispatched = []
    original = torch.Tensor.new_empty

    def allocate(tensor, shape, **kwargs):
        result = original(tensor, shape, **kwargs)
        allocations.append(result)
        return result

    monkeypatch.setattr(torch.Tensor, "new_empty", allocate)
    monkeypatch.setattr(
        ops,
        "fp8_qpn8_dispatch_sm70_out",
        lambda out, ptr, *args: dispatched.append(ptr),
    )
    x = torch.empty(9, 128, dtype=torch.float16)
    for _ in range(2):
        qpn._turing_fp8_qpn8_linear(x, torch.empty(0), torch.empty(0), 32, 8, 2, False)
    workspaces = [tensor for tensor in allocations if tensor.shape == (128, 32)]
    assert len(workspaces) == 2
    assert dispatched == [tensor.data_ptr() for tensor in workspaces]
    assert dispatched[0] != dispatched[1]
    dispatched.clear()
    assert qpn._turing_fp8_qpn8_linear(
        x[:0], torch.empty(0), torch.empty(0), 32, 8, 2, False
    ).shape == (0, 32)
    assert dispatched == []
