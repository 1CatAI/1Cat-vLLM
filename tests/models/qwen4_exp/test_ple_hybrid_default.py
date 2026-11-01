# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json

import vllm.models.qwen4_exp.common.ple as ple

GIB = 1 << 30


def _checkpoint(tmp_path, table_gib: int):
    weight_map = {
        "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight": (
            "model-plefp8-00000.safetensors"
        ),
        "model.language_model.layers.0.mlp.gate.weight": "model-bf16-00001.safetensors",
    }
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    with open(tmp_path / "model-plefp8-00000.safetensors", "wb") as f:
        f.truncate(table_gib * GIB)  # sparse: only the size matters
    return str(tmp_path)


def test_table_bytes_counts_only_ple_shards(tmp_path):
    path = _checkpoint(tmp_path, 3)
    assert ple.ple_table_bytes(path) == 3 * GIB
    assert ple.ple_table_bytes("org/remote-model") is None


def test_hybrid_default_uses_physical_memory(monkeypatch, tmp_path):
    path = _checkpoint(tmp_path, 40)
    # MemAvailable is never consulted: processes resolving the defaults after
    # others pinned their shares must reach the same placement.
    monkeypatch.setattr(ple, "available_host_bytes", lambda: 0)
    # Reserve 64/4 = 16 GiB, fair share (64 - 16) / 4 - 2 = 10 GiB per rank,
    # table share 40 / 4 = 10 GiB.
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 64 * GIB)
    assert ple.default_hybrid_host_gib(path, 4) == 10.0
    # 0.75 * 40 GiB + 8 GiB reserve = 38 GiB is the smallest host for hybrid.
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 37 * GIB)
    assert ple.default_hybrid_host_gib(path, 4) is None
    assert ple.default_hybrid_host_gib("org/remote-model", 4) is None


def test_apply_keeps_an_explicit_host_budget(monkeypatch, tmp_path):
    path = _checkpoint(tmp_path, 40)
    monkeypatch.setattr(ple, "total_host_bytes", lambda: 64 * GIB)
    defaults = {}
    assert ple.apply_hybrid_ple_defaults(defaults, path, 4) == 10.0
    assert defaults == {
        "VLLM_SM70_QWEN38_HYBRID_PLE": "1",
        "VLLM_QWEN4EXP_PLE_HOST_GIB": "10.0",
    }
    defaults = {"VLLM_QWEN4EXP_PLE_HOST_GIB": "7"}
    ple.apply_hybrid_ple_defaults(defaults, path, 4)
    assert defaults["VLLM_QWEN4EXP_PLE_HOST_GIB"] == "7"
    defaults = {}
    assert ple.apply_hybrid_ple_defaults(defaults, None, 4) is None
    assert defaults == {"VLLM_SM70_QWEN38_HYBRID_PLE": "0"}
