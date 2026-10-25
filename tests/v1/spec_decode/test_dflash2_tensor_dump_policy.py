# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import VllmConfig
from vllm.config.diagnostic_dump import TensorDumpConfig
from vllm.config.sm70_dflash2 import Sm70DFlash2Config
from vllm.diagnostics import DiagnosticChannel
from vllm.v1.worker.gpu.spec_decode.dflash2 import speculator as dflash2

pytestmark = pytest.mark.cpu_test

_DEBUG_BUFFERS = {
    "_debug_backbone_hidden_states": ((2, 7, 4), torch.float16),
    "_debug_candidate_ids": ((2, 7, 16), torch.int64),
    "_debug_unary_logits": ((2, 7, 16), torch.float32),
    "_debug_lattice_scores": ((2, 7, 16, 16), torch.float32),
}


def _make_speculator(monkeypatch, device, enabled):
    channel = DiagnosticChannel(
        TensorDumpConfig(directory="/diagnostic" if enabled else "", max_dumps=2)
    )

    def base_init(self, config, device):
        # Skip model/cache allocation, retaining the real DFlash2 constructor
        # and its engine-owned diagnostic policy.
        self.max_num_reqs = 2
        self.draft_block = self.num_speculative_steps = 7
        self.num_query_per_req = 8
        self.hidden_size = 4
        self.dtype = torch.float16
        self.draft_model_config = SimpleNamespace(
            hf_config=SimpleNamespace(dflash_config={"selector_top_k": 16})
        )
        self.draft_tokens = torch.empty((2, 7), dtype=torch.int64, device=device)
        self._draft_logits_init = self.draft_logits = None
        self.speculative_config = None
        self._tensor_dump = channel
        # A stale legacy attribute must neither enable nor disable capture.
        self._debug_tensor_dump_dir = "" if enabled else "/ignored/legacy"

    monkeypatch.setattr(dflash2.DFlashSpeculator, "__init__", base_init)
    monkeypatch.setattr(
        dflash2, "capture_sm70_dflash2_config", lambda _: Sm70DFlash2Config()
    )
    monkeypatch.setattr(
        dflash2,
        "bind_diagnostics",
        lambda _: SimpleNamespace(sampling=SimpleNamespace(value=lambda _: False)),
    )
    return dflash2.DFlash2Speculator(object.__new__(VllmConfig), torch.device(device))


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("enabled", [False, True])
def test_dflash2_capture_buffers_follow_tensor_dump_policy(
    monkeypatch, device, enabled
):
    speculator = _make_speculator(monkeypatch, device, enabled)
    for name, (shape, dtype) in _DEBUG_BUFFERS.items():
        buffer = getattr(speculator, name)
        if enabled:
            assert buffer.shape == shape
            assert buffer.dtype == dtype
            assert buffer.device == torch.device(device)
        else:
            assert buffer is None


@pytest.mark.parametrize("enabled", [False, True])
def test_dflash2_proposal_captures_current_tensors(monkeypatch, enabled):
    speculator = _make_speculator(monkeypatch, "cpu", enabled)
    hidden = torch.arange(8 * 4, dtype=torch.float16).view(8, 4)
    candidates = torch.arange(7 * 16, dtype=torch.int64).view(7, 16)
    unary = candidates.to(torch.float32) / 10
    scores = torch.arange(7 * 16 * 16, dtype=torch.float32).view(1, 7, 16, 16)
    monkeypatch.setattr(speculator, "sample_indices", torch.arange(1, 8), raising=False)
    monkeypatch.setattr(
        speculator,
        "input_buffers",
        SimpleNamespace(input_ids=torch.arange(8)),
        raising=False,
    )
    monkeypatch.setattr(speculator, "_run_model", lambda *args: hidden)
    monkeypatch.setattr(
        speculator,
        "model",
        SimpleNamespace(
            compute_candidates=lambda _: (candidates, unary),
            model=SimpleNamespace(candidate_selector=lambda *args: scores),
        ),
        raising=False,
    )

    def sample_path(candidate_ids, scores, num_reqs):
        speculator._selector_tokens[:num_reqs].copy_(candidate_ids[..., 0])

    monkeypatch.setattr(speculator, "_sample_path", sample_path)
    # Exercise the actual proposal copy path with deterministic CPU model output.
    speculator._generate_draft(1, 8, None, None, None)
    expected = {
        "_debug_backbone_hidden_states": hidden[1:].view(1, 7, 4),
        "_debug_candidate_ids": candidates.view(1, 7, 16),
        "_debug_unary_logits": unary.view(1, 7, 16),
        "_debug_lattice_scores": scores,
    }
    for name, values in expected.items():
        buffer = getattr(speculator, name)
        if enabled:
            torch.testing.assert_close(buffer[:1], values)
            assert buffer.data_ptr() != values.data_ptr()
        else:
            assert buffer is None
    torch.testing.assert_close(speculator.draft_tokens[:1], candidates[:, 0].view(1, 7))
