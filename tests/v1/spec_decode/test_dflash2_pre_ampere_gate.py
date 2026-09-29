# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2's BF16 emulation must follow the worker's own device.

``_use_sm70_bf16_emulation`` decides whether a BF16 draft checkpoint runs on
the range-preserving FP16 path. Every card without native BF16 arithmetic
needs it -- Volta and Turing alike -- and on a node that mixes generations,
device 0 of the visibility list belongs to a different worker.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models import qwen3_dflash2 as dflash2

SM70 = (7, 0)
SM75 = (7, 5)
SM80 = (8, 0)

BF16_CONFIG = SimpleNamespace(dtype=torch.bfloat16)


@dataclass
class Node:
    """Per-index capabilities of a node and the index this worker runs on."""

    capabilities: dict[int, tuple[int, int]] = field(
        default_factory=lambda: {0: SM75, 1: SM70, 2: SM80}
    )
    index: int = 0


def _as_tuple(capability: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(capability, int):
        return divmod(capability, 10)
    return capability


@pytest.fixture
def node(monkeypatch) -> Node:
    state = Node()

    def fake_has_device_capability(capability, device_id=0):
        return state.capabilities[device_id] >= _as_tuple(capability)

    def fake_is_device_capability(capability, device_id=0):
        return state.capabilities[device_id] == _as_tuple(capability)

    monkeypatch.delenv("VLLM_SM70_DFLASH2_BF16_EMULATION", raising=False)
    monkeypatch.setattr(dflash2.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash2.current_platform,
        "has_device_capability",
        fake_has_device_capability,
    )
    monkeypatch.setattr(
        dflash2.current_platform,
        "is_device_capability",
        fake_is_device_capability,
    )
    monkeypatch.setattr(
        dflash2.torch.accelerator,
        "current_device_index",
        lambda: state.index,
    )
    return state


def test_volta_worker_takes_the_emulation(node: Node) -> None:
    node.index = 1
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_turing_worker_takes_the_emulation(node: Node) -> None:
    """The regression: Turing has no BF16 arithmetic either, but the old
    exact-SM70 check sent it down the plain FP16 path."""
    node.index = 0
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_ampere_worker_keeps_native_bf16(node: Node) -> None:
    node.index = 2
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_gate_does_not_answer_for_device_zero(node: Node) -> None:
    """Device 0 is Ampere here; the Turing worker must still emulate."""
    node.capabilities = {0: SM80, 1: SM75}
    node.index = 1
    assert dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_fp16_checkpoint_needs_no_emulation(node: Node) -> None:
    node.index = 0
    fp16_config = SimpleNamespace(dtype=torch.float16)
    assert not dflash2._use_sm70_bf16_emulation(fp16_config)


def test_switch_disables_the_emulation(monkeypatch, node: Node) -> None:
    monkeypatch.setenv("VLLM_SM70_DFLASH2_BF16_EMULATION", "0")
    node.index = 0
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_non_cuda_platform_never_emulates(monkeypatch) -> None:
    monkeypatch.setattr(dflash2.current_platform, "is_cuda", lambda: False)
    assert not dflash2._use_sm70_bf16_emulation(BF16_CONFIG)


def test_flashinfer_topk_gate_follows_the_worker_device(
    monkeypatch, node: Node
) -> None:
    """The selector's FlashInfer top-k has no pre-SM80 kernel. Device 0 is
    Ampere here; the Turing worker must still fall back to torch.topk."""
    node.capabilities = {0: SM80, 1: SM75}
    node.index = 1
    monkeypatch.setattr(dflash2, "has_flashinfer", lambda: True)
    dflash2._flashinfer_topk.cache_clear()
    try:
        assert dflash2._flashinfer_topk() is None
    finally:
        dflash2._flashinfer_topk.cache_clear()
