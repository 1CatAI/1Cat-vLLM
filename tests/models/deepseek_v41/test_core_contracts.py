# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""C0 checks: contracts.py is frozen (PORT_DESIGN §3.1) and the package skeleton imports cleanly."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

# sha256 of common/contracts.py as written verbatim from PORT_DESIGN §3.1 in C0. Changing contracts.py
# requires an orchestrator-approved design change (PORT_DESIGN §2.1 rule 1); update this hash in that commit.
CONTRACTS_SHA256 = "47454e7899d9aeb075b27ac084e86aa78906bfe572d9e05bc6c886063520ddf8"


def test_contracts_frozen(ds41_package_dir: Path) -> None:
    digest = hashlib.sha256((ds41_package_dir / "common" / "contracts.py").read_bytes()).hexdigest()
    assert digest == CONTRACTS_SHA256, "contracts.py changed without an approved PORT_DESIGN §3.1 amendment"


def test_contracts_constants_consistent() -> None:
    from vllm.models.deepseek_v41.common import contracts as c

    assert c.NOPE_DIM + c.ROPE_DIM == c.HEAD_DIM
    assert c.ENGRAM_SUBTABLES == 3 * 8
    assert c.ENGRAM_ROW_BYTES == c.ENGRAM_HEAD_DIM + c.ENGRAM_HEAD_DIM // 32
    assert set(c.KV_SOURCES) <= set(c.INDEX_SOURCES)
    assert c.CAND_SOURCE in c.KV_SOURCES
    assert set(c.LEGAL_STAGE_STARTS) == {1} | set(c.INDEX_SOURCES)
    assert c.STREAM_DTYPE is torch.bfloat16 and c.ACT_DTYPE is torch.float16
    assert c.KV_RECORD_DTYPE is torch.float16


def test_allocate_shared_attn_buffers_cpu() -> None:
    from vllm.models.deepseek_v41.common.contracts import (
        CAND_TOPK_BLOCKS, CKV_RECORD_DIM, IDX_TOPK, IK_RECORD_DIM, StagePlan, allocate_shared_attn_buffers)

    exporting = StagePlan(pp_rank=1, pp_size=3, first_layer=14, last_layer=27, mirrored_kv_sources=(),
                          exports_kv_sources=(20,), engram_layers=(14,))
    buf = allocate_shared_attn_buffers(16, exporting, torch.device("cpu"))
    assert buf.topk_indices.shape == (16, IDX_TOPK) and buf.topk_indices.dtype == torch.int32
    assert bool((buf.topk_indices == -1).all()) and bool((buf.candidate_blocks == -1).all())
    assert buf.candidate_blocks.shape == (16, CAND_TOPK_BLOCKS)
    assert buf.export_ckv is not None and buf.export_ckv.shape == (16, CKV_RECORD_DIM)
    assert buf.export_ik is not None and buf.export_ik.shape == (16, IK_RECORD_DIM)
    plain = StagePlan(pp_rank=0, pp_size=1, first_layer=0, last_layer=39, mirrored_kv_sources=(),
                      exports_kv_sources=(), engram_layers=(1, 14))
    buf = allocate_shared_attn_buffers(4, plain, torch.device("cpu"))
    assert buf.export_ckv is None and buf.export_ik is None


def test_package_import_is_lazy() -> None:
    import vllm.models.deepseek_v41 as pkg

    assert "DeepseekV41ForCausalLM" in pkg.__all__
    with pytest.raises(AttributeError):
        pkg.does_not_exist  # noqa: B018


def test_knobs_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.models.deepseek_v41 import knobs

    monkeypatch.delenv("VLLM_DS41_CORE_TESTFLAG", raising=False)
    assert knobs.env_bool("VLLM_DS41_CORE_TESTFLAG", True) is True
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "off")
    assert knobs.env_bool("VLLM_DS41_CORE_TESTFLAG", True) is False
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "maybe")
    with pytest.raises(ValueError, match="VLLM_DS41_CORE_TESTFLAG"):
        knobs.env_bool("VLLM_DS41_CORE_TESTFLAG", True)
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "12")
    assert knobs.env_int("VLLM_DS41_CORE_TESTFLAG", 0, minimum=1, maximum=12) == 12
    with pytest.raises(ValueError):
        knobs.env_int("VLLM_DS41_CORE_TESTFLAG", 0, maximum=11)
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "x1")
    with pytest.raises(ValueError):
        knobs.env_int("VLLM_DS41_CORE_TESTFLAG", 0)
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "b")
    assert knobs.env_str("VLLM_DS41_CORE_TESTFLAG", "a", choices=("a", "b")) == "b"
    monkeypatch.setenv("VLLM_DS41_CORE_TESTFLAG", "c")
    with pytest.raises(ValueError):
        knobs.env_str("VLLM_DS41_CORE_TESTFLAG", "a", choices=("a", "b"))
    with pytest.raises(ValueError):
        knobs.env_bool("NOT_DS41_FLAG", False)
