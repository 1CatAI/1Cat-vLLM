# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import patch

from tests.utils import set_lazy_env
from vllm.model_executor.layers import ple_offload_layer
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding


def test_whole_table_placeholder_has_no_remote_row_placement(monkeypatch):
    set_lazy_env(monkeypatch, "VLLM_PLE_CPU_OFFLOAD", "1")
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    monkeypatch.setattr(ple_offload_layer, "_offload_worker_flag", False)
    with patch.object(
        Qwen4ExpNGramEmbedding, "offload_keeps_local_tables", return_value=False
    ):
        # The guarded GPU constructor deliberately skips every model argument.
        layer = Qwen4ExpNGramEmbedding()
    assert not hasattr(layer, "_cascade")
    assert layer.remote_placement() is None
