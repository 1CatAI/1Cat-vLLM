# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact-SM70 ModelOpt NVFP4: real kernel path vs an independent fp32 reference.

Runs only on a Volta (7,0) CUDA device. Builds a synthetic linear in the exact
on-disk ModelOpt layout (``weight`` uint8 E2M1 pairs, ``weight_scale`` fp8 e4m3
per 16-group, ``weight_scale_2`` fp32 = amax/(6*448)), loads it through the real
ModelOpt W4A4 / W4A16 quant methods (so the SM70 TurboMind prepare/apply route is
exercised end to end), and compares against a pure-torch dequant + fp32 GEMM that
shares no code with vLLM's scale handling. A convention error (reciprocated or
dropped global, PR 166 class) shows up as an output-norm ratio of ~2688x or
~1/2688x and fails loudly; kernel arithmetic error shows up as low cosine.
"""

import os

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import sm70_turbomind as sm70_tm
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _is_exact_sm70() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (7, 0)


pytestmark = pytest.mark.skipif(
    not _is_exact_sm70(), reason="requires an exact SM70 (Volta) CUDA device"
)


def _decode_e2m1(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    nib = torch.stack([lo, hi], dim=-1).reshape(packed.shape[0], -1).long()
    return E2M1[nib & 0x7] * torch.where((nib & 0x8) != 0, -1.0, 1.0)


def _make_modelopt_layer(n: int, k: int, seed: int):
    """Synthetic ModelOpt NVFP4 tensors: nibbles, fp8 block scales, amax global."""
    g = torch.Generator().manual_seed(seed)
    packed = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8)
    amax = 0.35
    weight_scale_2 = torch.tensor(amax / (6.0 * 448.0), dtype=torch.float32)
    # block scales in fp8: values so that block*global lands in a realistic weight range
    block = torch.rand(n, k // 16, generator=g) * 400.0 + 20.0
    weight_scale = block.to(torch.float8_e4m3fn)
    w_ref = (
        _decode_e2m1(packed)
        * weight_scale.to(torch.float32).repeat_interleave(16, dim=1)
        * weight_scale_2
    )
    return packed, weight_scale, weight_scale_2, w_ref


def _build_layer(quant_method: str, n: int, k: int, tensors, device):
    packed, weight_scale, weight_scale_2, _ = tensors
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29581")
    with set_current_vllm_config(VllmConfig()):
        try:
            init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method="env://",
                backend="gloo",
            )
            ensure_model_parallel_initialized(1, 1)
        except Exception:
            pass  # already initialised by an earlier test in this process
        qc = ModelOptNvFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            kv_cache_quant_algo=None,
            exclude_modules=[],
            group_size=16,
            quant_method=quant_method,
        )
        lin = ReplicatedLinear(
            k,
            n,
            bias=False,
            quant_config=qc,
            prefix="oracle",
            params_dtype=torch.float16,
        ).to(device)
        for pname, param in lin.named_parameters():
            src = {
                "weight": packed,
                "weight_scale": weight_scale,
                "weight_scale_2": weight_scale_2.reshape(1),
                "input_scale": torch.tensor([1.0]),
            }.get(pname)
            if src is None:
                continue
            loader = getattr(param, "weight_loader", None)
            if loader is not None:
                loader(
                    param, src if param.dtype == torch.uint8 else src.to(param.dtype)
                )
            else:
                param.data.copy_(src.reshape(param.shape))
        lin.quant_method.process_weights_after_loading(lin)
        return lin


@pytest.mark.parametrize("quant_method", ["NVFP4", "W4A16_NVFP4"])
@pytest.mark.parametrize("n,k", [(512, 1024), (1024, 512)])
def test_sm70_modelopt_nvfp4_kernel_matches_independent_reference(
    monkeypatch, quant_method, n, k
):
    monkeypatch.setenv("VLLM_SM70_NVFP4_TURBOMIND", "1")
    monkeypatch.delenv("VLLM_SM70_QUANT_BACKEND", raising=False)
    assert ModelOptNvFp4Config.get_min_capability() == 70

    device = torch.device("cuda:0")
    tensors = _make_modelopt_layer(n, k, seed=n + k)
    w_ref = tensors[3]
    lin = _build_layer(quant_method, n, k, tensors, device)
    assert sm70_tm.has_prepared_linear(lin), "expected the exact-SM70 TurboMind route"

    torch.manual_seed(0)
    x = (torch.randn(64, k) * 0.5).to(torch.float16)
    y_ref = x.float() @ w_ref.T  # independent fp32 reference
    y_floor = (x @ w_ref.to(torch.float16).T).float()  # fp16 accumulation floor
    with torch.no_grad():
        y_dut, _ = lin(x.to(device))
    y_dut = y_dut.float().cpu()

    ratio = float(y_dut.norm() / y_ref.norm())
    cos = float(
        torch.nn.functional.cosine_similarity(y_dut.flatten(), y_ref.flatten(), dim=0)
    )
    rel = float((y_dut - y_ref).norm() / y_ref.norm())
    rel_floor = float((y_floor - y_ref).norm() / y_ref.norm())
    # A reciprocated/dropped global would put ratio at ~2688 or ~1/2688.
    assert abs(ratio - 1.0) < 0.02, f"output scale ratio {ratio} (convention error?)"
    assert cos > 0.999, f"cosine {cos} (kernel arithmetic error?)"
    assert rel < max(5.0 * rel_floor, 5e-3), f"rel err {rel} vs fp16 floor {rel_floor}"
