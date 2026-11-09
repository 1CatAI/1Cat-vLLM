# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import sm70_qwen38_baseline as baseline


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    source = root / "kernel.cu"
    source.write_text("source fixture")
    monkeypatch.setattr(baseline, "ROOT", root)
    monkeypatch.setattr(baseline, "production_fingerprint", lambda: "test-source-tree")
    output = tmp_path / "bundle"
    output.mkdir()
    libraries = {}
    for name in (*baseline.LIBRARY_ENVS, "flashv100", "paged_kv"):
        path = output / f"{name}.so"
        path.write_bytes(name.encode())
        libraries[name] = {
            "path": str(path),
            "sha256": baseline.digest(path),
            "sources": {"kernel.cu": baseline.digest(source)},
        }
    manifest = {
        "complete": True,
        "libraries": libraries,
        "flashv100_sources": {},
        "production_fingerprint": "test-source-tree",
    }
    (output / "manifest.json").write_text(json.dumps(manifest))
    return output, root, manifest


def test_native_state_contract():
    args = baseline.engine_args("model")
    assert args["mamba_ssm_cache_dtype"] == "auto"
    assert args["dtype"] == "half" and args["kv_cache_dtype"] == "float16"
    assert args["tensor_parallel_size"] == 4 and args["max_num_seqs"] == 1
    assert args["max_model_len"] == 262144 and args["max_num_batched_tokens"] == 8192
    assert args["enable_prefix_caching"] is False
    assert args["disable_log_stats"] is False
    assert "speculative_config" not in args
    assert baseline.BASELINE_ENV["VLLM_SM70_NVFP4_QWEN38_MOE_QPN_BATCH_DECODE"] == "0"
    assert baseline.BASELINE_ENV["VLLM_SM70_QWEN4_EXP_ONLINE_QPN8"] == "0"
    assert baseline.BASELINE_ENV["VLLM_SM70_LM_HEAD_TOP1"] == "0"


def test_bundle_validation(bundle):
    output, _, expected = bundle
    assert baseline.validate_bundle(output) == expected


def test_reject_modified_binary(bundle):
    output, _, _ = bundle
    (output / "hc.so").write_bytes(b"changed")
    with pytest.raises(ValueError, match="binary"):
        baseline.validate_bundle(output)


def test_reject_modified_source(bundle):
    output, root, _ = bundle
    (root / "kernel.cu").write_text("changed")
    with pytest.raises(ValueError, match="Source changed"):
        baseline.validate_bundle(output)


def test_reject_modified_production_tree(bundle, monkeypatch):
    output, _, _ = bundle
    monkeypatch.setattr(baseline, "production_fingerprint", lambda: "changed-tree")
    with pytest.raises(ValueError, match="Production source tree"):
        baseline.validate_bundle(output)


def test_reject_incomplete_bundle(bundle):
    output, _, manifest = bundle
    manifest["complete"] = False
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete"):
        baseline.validate_bundle(output)


def test_reject_old_artifact_pointer(bundle, tmp_path):
    output, _, manifest = bundle
    old = tmp_path / "old.so"
    old.write_bytes(b"hc")
    manifest["libraries"]["hc"]["path"] = str(old)
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="outside"):
        baseline.validate_bundle(output)


def test_environment_is_explicit(bundle, tmp_path, monkeypatch):
    output, root, _ = bundle
    original = dict(os.environ)
    # configure_environment changes only this benchmark process, not user shell
    # state. Restore the complete mapping at the end of this unit test.
    monkeypatch.setattr(os, "environ", dict(original))
    os.environ["VLLM_UNRELATED_EXPERIMENT"] = "1"
    os.environ["VLLM_SM70_QWEN4_EXP_ONLINE_QPN8"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    baseline.configure_environment(output, tmp_path / "cache", None)
    assert "VLLM_UNRELATED_EXPERIMENT" not in os.environ
    assert os.environ["VLLM_SM70_QWEN4_EXP_ONLINE_QPN8"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "4,5,6,7"
    assert str(root) in os.environ["PYTHONPATH"]
    for component, name in baseline.LIBRARY_ENVS.items():
        assert os.environ[name] == str(output / f"{component}.so")


def test_default_route_does_not_inject_optimization_flags(
    bundle, tmp_path, monkeypatch
):
    output, _, _ = bundle
    monkeypatch.setattr(os, "environ", dict(os.environ))
    os.environ["VLLM_SM70_QWEN38_FP16_GEMV"] = "0"
    baseline.configure_environment(output, tmp_path / "cache", None, use_defaults=True)
    assert set(name for name in os.environ if name.startswith("VLLM_")) == set(
        baseline.LIBRARY_ENVS.values()
    ) - {"FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH"} | {"VLLM_CACHE_ROOT"}
    args = baseline.engine_args("model", use_defaults=True)
    assert (
        not {"compilation_config", "kernel_config", "attention_backend", "quantization"}
        & args.keys()
    )
    assert args["tensor_parallel_size"] == 4
    assert args["mamba_ssm_cache_dtype"] == "auto"
    assert args["kv_cache_dtype"] == "float16"


@pytest.mark.parametrize("failure", [None, "health", "reference"])
def test_driver_records_failure_and_releases_workers(
    bundle, tmp_path, monkeypatch, failure
):
    from benchmarks import benchmark_sm70_qwen38_baseline as driver

    monkeypatch.setattr(driver, "ROOT", Path(__file__).resolve().parents[2])
    output, _, manifest = bundle
    shutdown = []

    class FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda **kw: shutdown.append(kw))
            )

        def collective_rpc(self, *args, **kwargs):
            return [
                {
                    "rank": rank,
                    "ssm_dtype": "float32",
                    "mtp": False,
                    "prefix_cache": False,
                    "max_model_len": 262144,
                    "graph_mode": "FULL_DECODE_ONLY",
                    "qsa_specialization_version": 1,
                    "native_dependencies": {"vllm._C": {}},
                    "vllm_file": str(driver.ROOT / "vllm/__init__.py"),
                    "binaries": {
                        key: {"mapped": True, "sha256": entry["sha256"]}
                        for key, entry in manifest["libraries"].items()
                        if key != "paged_kv"
                    },
                }
                for rank in range(4)
            ]

    def fake_run(llm, prompt, sampling):
        return {
            "finish_reason": "length" if failure == "health" else "stop",
            "text": "RESULT=437 CEDAR-47|8261",
            "output_tokens": 513,
            "token_ids": [1] * 513,
            "request_metrics": {"prefill_time": 1.0, "raw": {"is_corrupted": False}},
        }

    tokenizer = SimpleNamespace(
        encode=lambda *a, **kw: [7, 8],
        apply_chat_template=lambda *a, **kw: [9],
    )
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(LLM=FakeLLM, SamplingParams=SimpleNamespace),
    )
    monkeypatch.setitem(
        sys.modules,
        "benchmarks.benchmark_sm70_decode",
        SimpleNamespace(
            _hash_ids=lambda ids: "hash",
            _run_once=fake_run,
            _summarize=lambda runs: {"count": len(runs)},
        ),
    )
    model = tmp_path / "model"
    model.mkdir()
    (model / "generation_config.json").write_text(
        json.dumps({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    )
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "name": "fixed_8192",
                        "runs": [
                            {"token_ids": [2 if failure == "reference" else 1] * 513}
                        ],
                    }
                ]
            }
        )
    )
    args = SimpleNamespace(
        model=str(model),
        runtime_dir=output,
        long_context=False,
        repeats=2,
        reference_json=reference,
        output=tmp_path / "result.json",
    )
    if failure:
        with pytest.raises(RuntimeError, match="health failed|differs from frozen"):
            driver.run(args)
    else:
        driver.run(args)
    result = json.loads(args.output.read_text())
    assert result["complete"] is (failure is None)
    assert ("error" in result) is (failure is not None)
    assert shutdown == [{"timeout": 30.0}]
