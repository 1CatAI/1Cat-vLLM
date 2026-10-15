# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import ple_layer as ple


@pytest.mark.parametrize("active", [0, 3, 5])
def test_owned_conv_result_preserves_input_and_initialized_padding(monkeypatch, active):
    source = torch.full((5, 4), 7, dtype=torch.half)
    before = source.clone()
    result = torch.arange(active * 4).reshape(active, 4).half()
    owners = {
        "first": NS(_short_conv=lambda x: result),
        "second": NS(_short_conv=lambda x: result + 10),
    }
    monkeypatch.setattr(
        ple, "get_forward_context", lambda: NS(no_compile_layers=owners)
    )
    for name, offset in (("first", 0), ("second", 10)):
        output = ple.qwen4_exp_ple_short_conv_result(source, name)
        assert output.shape == source.shape and output.dtype == source.dtype
        assert torch.equal(output[:active], result + offset)
        assert bool((output[active:] == 0).all())
        assert torch.equal(source, before)
        assert output.data_ptr() != source.data_ptr()
    if active == 5:
        assert (
            ple.qwen4_exp_ple_short_conv_result(source, "first").data_ptr()
            == result.data_ptr()
        )
