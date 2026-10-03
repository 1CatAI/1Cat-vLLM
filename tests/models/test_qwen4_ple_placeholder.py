# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding


def test_whole_table_placeholder_has_no_remote_row_placement(monkeypatch):
    monkeypatch.setenv("VLLM_PLE_CPU_OFFLOAD", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    with patch.object(
        Qwen4ExpNGramEmbedding, "offload_keeps_local_tables", return_value=False
    ):
        # The guarded GPU constructor deliberately skips every model argument.
        layer = Qwen4ExpNGramEmbedding()
    assert not hasattr(layer, "_cascade")
    assert layer.remote_placement() is None
