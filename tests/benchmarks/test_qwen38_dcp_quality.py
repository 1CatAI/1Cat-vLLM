# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only validation of the offline qualification transport."""

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.benchmark_qwen38_dcp_quality import (
    build_prompts,
    run,
    token_difference,
    validate_manifest_transport,
    worker_manifest,
)
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

pytestmark = pytest.mark.cpu_test


def test_manifest_transport_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    with pytest.raises(TypeError, match="VLLM_ALLOW_INSECURE_SERIALIZATION"):
        validate_manifest_transport()


def test_manifest_callable_and_result_roundtrip(monkeypatch):
    monkeypatch.setattr(
        "benchmarks.benchmark_qwen38_dcp_quality.runtime_sources",
        lambda: {"version": "test"},
    )
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    validate_manifest_transport()
    for name in ("memory_allocated", "memory_reserved", "max_memory_allocated"):
        monkeypatch.setattr(torch.accelerator, name, lambda: 1234)
    worker = SimpleNamespace(
        rank=0,
        vllm_config=SimpleNamespace(
            cache_config=SimpleNamespace(
                cache_dtype="fp8_e4m3",
                mamba_ssm_cache_dtype="float32",
                enable_prefix_caching=False,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=2, dcp_comm_backend="a2a"
            ),
            speculative_config=None,
            compilation_config=SimpleNamespace(cudagraph_mode="FULL_AND_PIECEWISE"),
        ),
        model_runner=SimpleNamespace(
            sampler=SimpleNamespace(compute_nans=True),
            kv_cache_config=SimpleNamespace(
                num_blocks=32,
                kv_cache_tensors=[SimpleNamespace(size=4096)],
                kv_cache_groups=[
                    SimpleNamespace(
                        layer_names=["target"],
                        kv_cache_spec=SimpleNamespace(
                            block_size=32, page_size_bytes=128, dcp_sharded=True
                        ),
                    )
                ],
            ),
        ),
    )
    encoder, decoder = MsgpackEncoder(), MsgpackDecoder()
    callback = decoder.decode(encoder.encode(worker_manifest))
    manifest = callback(worker)
    decoded = decoder.decode(encoder.encode(manifest))
    assert decoded == manifest
    assert decoded["physical_kv_bytes"] == 4096
    assert decoded["dcp"] == 2
    assert decoded["sampler_checks_nans"] is True


class TemplateTokenizer:
    def apply_chat_template(self, messages, *, tokenize, return_dict=True, **kwargs):
        if not tokenize:
            return "START ARCHIVE_BODY END"
        return {"input_ids": [1, 2, 3]} if return_dict else [1, 2, 3]

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]


@pytest.mark.parametrize("long_context", [False, True])
def test_all_prompt_inputs_are_ready_before_model_load(long_context):
    cases, boundary = build_prompts(TemplateTokenizer(), long_context)
    assert [len(ids) for _, ids, _ in cases[:4]] == [3, 3, 3, 3]
    assert [len(ids) for _, ids, _ in cases[4:]] == (
        [8192, 32768, 261632] if long_context else [8192]
    )
    assert (len(boundary) if boundary is not None else 0) == (
        262143 if long_context else 0
    )


def test_bad_tokenizer_output_fails_before_model_load():
    class BrokenTokenizer(TemplateTokenizer):
        def apply_chat_template(self, *args, tokenize, **kwargs):
            if tokenize:
                return {"input_ids": [1, 2, 3]}
            return super().apply_chat_template(*args, tokenize=False, **kwargs)

    with pytest.raises(TypeError, match="integer token IDs"):
        build_prompts(BrokenTokenizer(), False)


@pytest.mark.parametrize("healthy", [False, True])
def test_driver_saves_result_and_always_shuts_down(monkeypatch, tmp_path, healthy):
    import transformers

    import vllm

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_COMPUTE_NANS_IN_LOGITS", "0")
    monkeypatch.setattr(
        "benchmarks.benchmark_qwen38_dcp_quality.validate_spawn_runtime",
        lambda: {"version": "test"},
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: TemplateTokenizer(),
    )
    shutdowns = []

    class FakeLLM:
        def __init__(self, **kwargs):
            import os

            assert os.environ["VLLM_COMPUTE_NANS_IN_LOGITS"] == "1"
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(
                    shutdown=lambda **kwargs: shutdowns.append(True)
                )
            )

        def collective_rpc(self, *args, **kwargs):
            return [
                dict(
                    mtp=False,
                    prefix_cache=False,
                    ssm_dtype="float32",
                    fp16_reduced_reduction=False,
                    bf16_reduced_reduction=False,
                    fp16_accumulation=False,
                    sampler_checks_nans=True,
                    runtime={"version": "test"},
                    dcp=1,
                    kv_dtype="fp8_e4m3",
                    ple_environment={"VLLM_PLE_DISK_OFFLOAD": "1"},
                )
            ]

        def generate(self, prompts, sampling, **kwargs):
            assert all(isinstance(t, int) for t in prompts[0]["prompt_token_ids"])
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text="437 CEDAR-47|8261 MAPLE-8261" if healthy else "bad",
                            token_ids=[7],
                            finish_reason="stop",
                            logprobs=[{7: SimpleNamespace(logprob=-0.5)}],
                        )
                    ],
                    metrics=SimpleNamespace(
                        is_corrupted=False,
                        scheduled_ts=1.0,
                        first_token_ts=2.0,
                        last_token_ts=3.0,
                    ),
                )
            ]

    monkeypatch.setattr(vllm, "LLM", FakeLLM)
    (tmp_path / "generation_config.json").write_text(
        json.dumps(dict(temperature=0.7, top_p=0.8, top_k=20))
    )
    args = SimpleNamespace(
        model=str(tmp_path),
        dcp=1,
        kv_gib=4.0,
        kv_dtype="auto",
        gpu_memory_utilization=0.90,
        long_context=True,
        preflight_only=False,
        require_token_parity=False,
        dataset_spec=None,
        reference=None,
        output=tmp_path / "report.json",
    )
    if healthy:
        run(args)
    else:
        with pytest.raises(RuntimeError, match="Quality gate failed"):
            run(args)
    report = json.loads(args.output.read_text())
    assert report["complete"] is healthy
    assert len(report["cases"]) == 7
    assert report["checks_finished"]
    assert len(report["quality_failures"]) == (0 if healthy else 7)
    if healthy:
        assert report["exact_256k_boundary"]["finite"]
    assert shutdowns == [True]


@pytest.mark.parametrize(
    ("control", "candidate", "position"),
    [([1, 2], [1, 2], None), ([1, 2], [1, 3], 1), ([1, 2], [1], 1)],
)
def test_token_difference(control, candidate, position):
    result = token_difference(control, candidate)
    assert result["matches"] == (position is None)
    assert result["first_differing_token_0based"] == position
