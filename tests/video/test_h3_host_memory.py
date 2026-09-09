# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import argparse

import pytest
import torch
from torch import nn

from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError


@pytest.mark.parametrize("mode", ["generate", "serve"])
def test_pageable_weight_masters_are_an_explicit_deployment_option(mode):
    from vllm.entrypoints.cli.video import VideoSubcommand

    parser = argparse.ArgumentParser()
    VideoSubcommand().subparser_init(parser.add_subparsers())
    assert parser.parse_args(["video", mode]).host_weight_pin_memory
    assert not parser.parse_args(
        ["video", mode, "--disable-host-weight-pinning"]
    ).host_weight_pin_memory
    assert not H3Config(host_weight_pin_memory=False).host_weight_pin_memory
    with pytest.raises(H3InputError, match="boolean"):
        H3Config(host_weight_pin_memory="false")


@pytest.mark.parametrize("pin_memory", [True, False])
@pytest.mark.parametrize("kind", ["MiniMaxH3VideoVAE", "MiniMaxH3AudioVAE"])
def test_both_vae_stagers_honor_the_host_policy(monkeypatch, pin_memory, kind):
    from vllm.model_executor.models.minimax_h3 import vae

    remote = nn.Module()
    remote.model = nn.Linear(2, 2)
    expected = remote.model.weight.clone()
    calls = []
    monkeypatch.setattr(
        vae, "_load_component_config", lambda path: {"sample_rate": 44100}
    )
    monkeypatch.setattr(vae, "_load_remote_component", lambda *args: remote)
    monkeypatch.setattr(
        vae,
        "PinnedModuleStager",
        lambda module, device, **kwargs: calls.append((module, device, kwargs)),
    )
    wrapper = getattr(vae, kind)(
        "test",
        device=torch.device("cuda"),
        load_device=torch.device("cpu"),
        pin_memory=pin_memory,
    )
    assert calls == [(remote, torch.device("cuda"), {"pin_memory": pin_memory})]
    assert wrapper.model.weight.dtype == torch.float32
    torch.testing.assert_close(wrapper.model.weight, expected, rtol=0, atol=0)
