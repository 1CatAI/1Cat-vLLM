# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import pytest
from safetensors import safe_open


def _load_tool():
    path = (
        Path(__file__).resolve().parents[3]
        / "tools"
        / "qwen4_exp"
        / "qsa_kv_calibration.py"
    )
    spec = importlib.util.spec_from_file_location("qsa_kv_calibration_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qsa_kv_overlay_uses_merged_standard_index(tmp_path: Path) -> None:
    tool = _load_tool()
    base = tmp_path / "base"
    base.mkdir()
    shard = base / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"base checkpoint remains untouched")
    base_weights = {"model.embed_tokens.weight": shard.name}
    base_weights.update(
        {
            f"model.language_model.layers.{layer}.self_attn.k_proj.weight": shard.name
            for layer in range(12)
        }
    )
    base_weights["mtp.layers.0.self_attn.k_proj.weight"] = shard.name
    base_index = {
        "metadata": {"total_size": shard.stat().st_size},
        "weight_map": base_weights,
    }
    (base / tool.INDEX_FILENAME).write_text(json.dumps(base_index), encoding="utf-8")
    (base / "config.json").write_text("{}\n", encoding="utf-8")
    tensors = {
        f"model.layers.{layer}.self_attn.{kind}_scale": {"scale": 0.01 + layer / 1000}
        for layer in range(12)
        for kind in ("k", "v")
    }
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"tensor_count": 24, "tensors": tensors}), encoding="utf-8"
    )
    output = tmp_path / "overlay"
    tool.build_overlay(
        Namespace(
            base_checkpoint=str(base),
            report=str(report),
            output_dir=str(output),
            expected_layers=12,
            unit_scale_negative_control=False,
        )
    )

    merged = json.loads((output / tool.INDEX_FILENAME).read_text(encoding="utf-8"))
    checkpoint_tensors = {
        name.replace("model.layers.", "model.language_model.layers.", 1)
        for name in tensors
    }
    assert len(merged["weight_map"]) == 38
    assert checkpoint_tensors <= set(merged["weight_map"])
    assert all(
        merged["weight_map"][name] == tool.SCALE_FILENAME for name in checkpoint_tensors
    )
    assert (output / shard.name).is_symlink()
    assert shard.read_bytes() == b"base checkpoint remains untouched"
    with safe_open(output / tool.SCALE_FILENAME, framework="pt") as scales:
        assert set(scales.keys()) == checkpoint_tensors


def test_qsa_kv_summary_reports_scales_percentiles_and_shards(tmp_path: Path) -> None:
    tool = _load_tool()
    traces = tmp_path / "traces"
    traces.mkdir()
    stat = {
        "count": 5,
        "finite_count": 5,
        "nonzero_count": 4,
        "max_abs": 224.0,
        "histogram": [0, 1, 2, 1],
    }
    record = {
        "schema_version": 1,
        "layer_id": 3,
        "corpus_shard": "code-16k",
        "histogram": {"bins": 4, "log2_min": -2.0, "log2_max": 2.0},
        "k": stat,
        "v": {**stat, "max_abs": 112.0},
    }
    (traces / "qsa-kv-rank0-pid1.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    output = tmp_path / "summary.json"
    tool.summarize(
        Namespace(input_dir=str(traces), output=str(output), expected_layers=1)
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["tensor_count"] == 2
    key = report["tensors"]["model.layers.3.self_attn.k_scale"]
    value = report["tensors"]["model.layers.3.self_attn.v_scale"]
    assert key["scale"] == 0.5
    assert value["scale"] == 0.25
    assert key["saturation_ratio"] == 0.0
    assert key["shard_scales"] == {"code-16k": 0.5}
    assert key["p99_9_abs_upper_bound"] > 0.0
    assert key["p99_99_abs_upper_bound"] > 0.0


def test_scale_rounds_up_to_runtime_float32_without_saturating_max() -> None:
    tool = _load_tool()
    max_abs = 37.03125

    scale = tool._ceil_float32(max_abs / tool.E4M3_MAX)

    assert scale >= max_abs / tool.E4M3_MAX
    assert 448.0 * scale >= max_abs


def test_summary_combines_directories_and_excludes_superseded_shards(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    histogram = {"bins": 1, "log2_min": -2.0, "log2_max": 2.0}

    def write_record(directory: Path, shard: str, max_abs: float) -> None:
        directory.mkdir()
        stat = {
            "count": 1,
            "finite_count": 1,
            "nonzero_count": 1,
            "max_abs": max_abs,
            "histogram": [1],
        }
        record = {
            "schema_version": 1,
            "layer_id": 3,
            "corpus_shard": shard,
            "histogram": histogram,
            "k": stat,
            "v": stat,
        }
        (directory / "qsa-kv-rank0-pid1.jsonl").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )

    generic = tmp_path / "generic"
    page4 = tmp_path / "page4"
    write_record(generic, "english-128k-generic", 448.0)
    write_record(page4, "english-128k-page4", 224.0)
    output = tmp_path / "summary.json"

    tool.summarize(
        Namespace(
            input_dir=[str(generic), str(page4)],
            output=str(output),
            expected_layers=1,
            exclude_corpus_shard=["english-128k-generic"],
        )
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    tensor = report["tensors"]["model.layers.3.self_attn.k_scale"]
    assert tensor["max_abs"] == 224.0
    assert tensor["shard_scales"] == {"english-128k-page4": 0.5}
    assert report["excluded_corpus_shards"] == ["english-128k-generic"]


def test_scale_envelope_keeps_base_percentiles_and_covers_other_runs(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    tensor_name = "model.layers.3.self_attn.k_scale"
    base = tmp_path / "base.json"
    candidate = tmp_path / "candidate.json"
    base.write_text(
        json.dumps(
            {
                "e4m3_max": 448.0,
                "tensor_count": 1,
                "tensors": {
                    tensor_name: {
                        "max_abs": 224.0,
                        "scale": 0.5,
                        "saturation_ratio": 0.0,
                        "p99_9_abs_upper_bound": 8.0,
                        "p99_99_abs_upper_bound": 16.0,
                        "shard_scales": {"base": 0.5},
                        "shard_scale_min_over_global": 1.0,
                        "shard_scale_max_over_global": 1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            {
                "e4m3_max": 448.0,
                "tensor_count": 1,
                "tensors": {
                    tensor_name: {
                        "max_abs": 336.0,
                        "scale": 0.75,
                        "shard_scales": {"candidate": 0.75},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "envelope.json"

    tool.build_scale_envelope(
        Namespace(
            base_report=str(base),
            scale_report=[str(candidate)],
            output=str(output),
        )
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    tensor = report["tensors"][tensor_name]
    assert tensor["distribution_max_abs"] == 224.0
    assert tensor["max_abs"] == 336.0
    assert tensor["scale"] == 0.75
    assert tensor["p99_9_abs_upper_bound"] == 8.0
    assert tensor["shard_scales"] == {"base": 0.5, "candidate": 0.75}
    assert len(report["scale_envelope_reports"]) == 2


def test_prepare_uses_dynamic_generation_prompt_and_writes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tool = _load_tool()

    class FakeTokenizer:
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, tools=None
        ):
            assert tokenize
            if tools:
                return [10, 12, 13]
            return [10, 11] if add_generation_prompt else [10]

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps(
                    {"id": "assistant-final", "messages": [{"role": "assistant"}]}
                ),
                json.dumps({"id": "user-final", "messages": [{"role": "user"}]}),
                json.dumps(
                    {
                        "id": "tool-final",
                        "messages": [{"role": "assistant"}],
                        "tools": [{"type": "function", "function": {"name": "f"}}],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "prepared.jsonl"
    tool.prepare(
        Namespace(
            model=str(tmp_path),
            input_jsonl=str(source),
            output_jsonl=str(output),
            max_prompt_tokens=None,
            processor_audit_jsonl=None,
            corpus_manifest=None,
        )
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["prompt_token_ids"] for row in rows] == [[10], [10, 11], [10, 12, 13]]
    assert [row["calibration_metadata"]["has_tools"] for row in rows] == [
        False,
        False,
        True,
    ]
    manifest = json.loads(output.with_name(output.name + ".manifest.json").read_text())
    assert manifest["record_count"] == 3
    assert manifest["active_tokens"] == 6
    assert manifest["processor_audit_verified"] is False


def test_prepare_requires_audit_and_manifest_together(tmp_path: Path) -> None:
    tool = _load_tool()
    source = tmp_path / "source.jsonl"
    source.write_text('{"text":"hello"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="must be supplied together"):
        tool.prepare(
            Namespace(
                model=str(tmp_path),
                input_jsonl=str(source),
                output_jsonl=str(tmp_path / "prepared.jsonl"),
                max_prompt_tokens=None,
                processor_audit_jsonl=str(tmp_path / "audit.jsonl"),
                corpus_manifest=None,
            )
        )
