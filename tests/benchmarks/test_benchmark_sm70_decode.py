# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import subprocess
import types

from benchmarks import (
    benchmark_sm70_decode,
    benchmark_sm70_dflash2_gsm8k,
    benchmark_sm70_model_tokens,
    benchmark_sm70_serving_quality,
)


def test_dflash_quality_git_provenance_is_optional(monkeypatch, tmp_path):
    def fail(*_args, **_kwargs):
        raise subprocess.CalledProcessError(128, ["git", "rev-parse", "HEAD"])

    monkeypatch.setattr(benchmark_sm70_dflash2_gsm8k.subprocess, "check_output", fail)

    assert benchmark_sm70_dflash2_gsm8k._git_sha(tmp_path) is None


def test_dflash_quality_persists_completed_case(tmp_path):
    path = tmp_path / "quality.partial.jsonl"
    output = types.SimpleNamespace(
        outputs=[
            types.SimpleNamespace(
                token_ids=[3, 5, 8],
                finish_reason="stop",
                stop_reason=None,
                text="answer",
            )
        ]
    )

    benchmark_sm70_dflash2_gsm8k._append_partial_case(
        path,
        dataset_index=7,
        request_seed=42,
        output=output,
    )

    assert json.loads(path.read_text()) == {
        "dataset_index": 7,
        "finish_reason": "stop",
        "output_tokens": 3,
        "request_seed": 42,
        "stop_reason": None,
        "text": "answer",
        "token_ids": [3, 5, 8],
    }


def test_sm70_fa2_d256_prefill_status_reports_import_error(monkeypatch):
    def fail_import(_name):
        raise ImportError("missing vendored extension")

    monkeypatch.setattr(benchmark_sm70_decode.importlib, "import_module", fail_import)

    status = benchmark_sm70_decode._sm70_fa2_d256_prefill_status(
        types.SimpleNamespace(ops=types.SimpleNamespace())
    )

    assert status["available"] is False
    assert status["error"] == "ImportError: missing vendored extension"
    assert not any(status["required_ops"].values())


def test_sm70_fa2_d256_prefill_status_requires_dense_and_paged_ops(monkeypatch):
    monkeypatch.setattr(
        benchmark_sm70_decode.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(),
    )
    namespace = types.SimpleNamespace(
        sm70_d256_splitd_n32_dense_fwd=object(),
        sm70_d256_splitd_n32_paged_fwd=object(),
        sm70_d256_splitd_n32_dense_splitkv3_fwd=object(),
    )
    fake_torch = types.SimpleNamespace(ops=types.SimpleNamespace(_vllm_fa2_C=namespace))
    fake_extension = types.SimpleNamespace(__file__="/tmp/_vllm_fa2_C.abi3.so")
    monkeypatch.setitem(
        benchmark_sm70_decode.sys.modules,
        "vllm.vllm_flash_attn._vllm_fa2_C",
        fake_extension,
    )

    status = benchmark_sm70_decode._sm70_fa2_d256_prefill_status(fake_torch)

    assert status["available"] is True
    assert status["error"] is None
    assert all(status["required_ops"].values())
    assert all(status["optional_ops"].values())


def test_sm70_fa2_d256_prefill_status_accepts_explicit_sidecar(monkeypatch):
    namespace = types.SimpleNamespace()
    loaded: list[str] = []

    def load_library(path: str) -> None:
        loaded.append(path)
        namespace.sm70_d256_splitd_n32_dense_fwd = object()
        namespace.sm70_d256_splitd_n32_paged_fwd = object()

    fake_ops = types.SimpleNamespace(
        _vllm_fa2_C=namespace,
        load_library=load_library,
    )
    fake_torch = types.SimpleNamespace(ops=fake_ops)
    monkeypatch.setattr(
        benchmark_sm70_decode.importlib,
        "import_module",
        lambda _name: types.SimpleNamespace(),
    )
    monkeypatch.setenv("VLLM_SM70_FA2_D256_LIBRARY", "/tmp/stable-fa2.so")

    status = benchmark_sm70_decode._sm70_fa2_d256_prefill_status(fake_torch)

    assert status["available"] is True
    assert status["extension_file"] == "/tmp/stable-fa2.so"
    assert loaded == ["/tmp/stable-fa2.so"]


def test_spec_decoding_delta_reports_one_sequential_prompt():
    before = [
        {"name": "vllm:spec_decode_num_drafts", "value": 10},
        {"name": "vllm:spec_decode_num_draft_tokens", "value": 30},
        {"name": "vllm:spec_decode_num_accepted_tokens", "value": 20},
        {
            "name": "vllm:spec_decode_num_accepted_tokens_per_pos",
            "values": [8, 7, 5],
        },
    ]
    after = [
        {"name": "vllm:spec_decode_num_drafts", "value": 14},
        {"name": "vllm:spec_decode_num_draft_tokens", "value": 42},
        {"name": "vllm:spec_decode_num_accepted_tokens", "value": 27},
        {
            "name": "vllm:spec_decode_num_accepted_tokens_per_pos",
            "values": [11, 9, 7],
        },
    ]

    metrics = benchmark_sm70_model_tokens._spec_decoding_delta(before, after)

    assert metrics is not None
    assert metrics["num_drafts"] == 4
    assert metrics["num_draft_tokens"] == 12
    assert metrics["num_accepted_tokens"] == 7
    assert metrics["accepted_tokens_per_pos"] == [3, 2, 2]
    assert metrics["mean_acceptance_length"] == 2.75


def test_serving_quality_preserves_all_dflash_positions():
    prefix = "vllm:spec_decode_num_accepted_tokens_per_pos_total:position_"
    before = {
        "vllm:spec_decode_num_drafts_total": 10.0,
        "vllm:spec_decode_num_draft_tokens_total": 70.0,
        "vllm:spec_decode_num_accepted_tokens_total": 35.0,
        **{f"{prefix}{pos}": float(10 - pos) for pos in range(7)},
    }
    after = {
        "vllm:spec_decode_num_drafts_total": 12.0,
        "vllm:spec_decode_num_draft_tokens_total": 84.0,
        "vllm:spec_decode_num_accepted_tokens_total": 42.0,
        **{f"{prefix}{pos}": float(12 - pos) for pos in range(7)},
    }

    delta = benchmark_sm70_serving_quality._counter_delta(before, after)

    assert delta is not None
    assert delta["accepted_tokens_per_pos"] == [2.0] * 7
    summary = benchmark_sm70_serving_quality._summarize_spec_metrics(
        [
            {"serving_metrics_delta": delta},
            {
                "serving_metrics_delta": {
                    **delta,
                    "accepted_tokens_per_pos": [1.0, 1.0, 1.0],
                }
            },
        ]
    )
    assert summary is not None
    assert summary["accepted_tokens_per_pos"] == [3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0]


def test_sm70_policy_records_generic_mtp_moe_tuning(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QWEN36_MTP_MOE_TUNED_CONFIG", "1")
    monkeypatch.delenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", raising=False)

    policy = benchmark_sm70_model_tokens._sm70_turbomind_policy()

    assert policy["VLLM_SM70_QWEN36_MTP_MOE_TUNED_CONFIG"] == "1"
    assert policy["VLLM_SM70_MTP_MOE_TUNED_CONFIG"] is None
    assert policy["mtp_moe_tuned_config_effective"] is True
    assert policy["qwen36_mtp_moe_tuned_config_effective"] is True

    monkeypatch.setenv("VLLM_SM70_MTP_MOE_TUNED_CONFIG", "0")
    policy = benchmark_sm70_model_tokens._sm70_turbomind_policy()
    assert policy["mtp_moe_tuned_config_effective"] is False
    assert policy["qwen36_mtp_moe_tuned_config_effective"] is False
