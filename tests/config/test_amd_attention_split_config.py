# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMD opaque-op names participate in default splitting and cache identity."""

from unittest.mock import patch

import pytest

from vllm.config.compilation import CompilationConfig, CompilationMode, PassConfig

AMD_OPS = {
    "vllm::qwen4_exp_ple_short_conv_amd",
    "vllm::qwen4_exp_qsa_with_output_amd",
}


@pytest.mark.parametrize(
    ("mode", "splitting_ops", "uses_default"),
    [
        (CompilationMode.VLLM_COMPILE, None, True),
        (CompilationMode.VLLM_COMPILE, [], False),
        (CompilationMode.NONE, None, False),
    ],
)
def test_amd_split_names_and_hash(mode, splitting_ops, uses_default):
    def make_config():
        config = CompilationConfig(
            mode=mode,
            splitting_ops=None if splitting_ops is None else list(splitting_ops),
            use_inductor_graph_partition=True,
            pass_config=PassConfig(fuse_attn_quant=False),
        )
        config.set_splitting_ops_for_v1("naive")
        return config

    config = make_config()
    original_names = [
        name for name in CompilationConfig._attention_ops if name not in AMD_OPS
    ]
    with patch.object(CompilationConfig, "_attention_ops", original_names):
        before = make_config()

    if uses_default:
        assert AMD_OPS.issubset(config.splitting_ops)
        assert "vllm::qwen4_exp_ple_short_conv" in config.splitting_ops
        assert "vllm::qwen4_exp_qsa_with_output" in config.splitting_ops
        assert config.compute_hash() != before.compute_hash()
    else:
        assert config.splitting_ops == before.splitting_ops == []
        assert config.compute_hash() == before.compute_hash()
