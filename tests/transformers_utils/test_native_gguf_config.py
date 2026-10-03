# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gguf
import numpy as np
import pytest
import torch
from transformers import PretrainedConfig

from vllm.transformers_utils.gguf_config import (
    gguf_config_dict,
    load_gguf_config,
    read_gguf_metadata,
)
from vllm.transformers_utils.gguf_files import gguf_shard_paths, gguf_tensor_index


@pytest.mark.parametrize("model", ["org/repo/model.gguf", "org/repo:IQ3_XXS"])
def test_speculators_probe_does_not_use_transformers_for_gguf(monkeypatch, model):
    from vllm.transformers_utils.config import maybe_override_with_speculators

    def forbidden(*args, **kwargs):
        raise AssertionError("GGUF must not enter the Transformers config parser")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", forbidden)
    speculative = {"method": "mtp", "num_speculative_tokens": 1}
    assert maybe_override_with_speculators(
        model, None, False, vllm_speculative_config=speculative
    ) == (model, None, speculative)


def write_fixture(path, arch="qwen35", *, extra=None, tensor="token_embd.weight"):
    w = gguf.GGUFWriter(path, arch)
    w.add_embedding_length(256)
    w.add_context_length(8192)
    w.add_head_count(4)
    w.add_head_count_kv(2)
    w.add_key_length(64)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_block_count(5)
    w.add_uint32(f"{arch}.nextn_predict_layers", 1)
    w.add_feed_forward_length(512)
    w.add_ssm_conv_kernel(4)
    w.add_ssm_state_size(16)
    w.add_ssm_group_count(4)
    w.add_ssm_time_step_rank(8)
    w.add_ssm_inner_size(256)
    w.add_rope_dimension_count(16)
    w.add_rope_dimension_sections([3, 3, 2, 0])
    w.add_rope_freq_base(10000000.0)
    w.add_array(f"{arch}.attention.recurrent_layers", [True, False, True, False, False])
    w.add_token_list(["a", "b", "c", "d"])
    w.add_eos_token_id(3)
    if arch == "qwen35moe":
        w.add_expert_count(8)
        w.add_expert_used_count(2)
        w.add_expert_feed_forward_length(640)
        w.add_expert_shared_feed_forward_length(512)
    for key, value in (extra or {}).items():
        w.add_uint32(key, value)
    w.add_tensor(tensor, np.ones((4, 256), dtype=np.float16))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


@pytest.mark.parametrize("arch", ["qwen35", "qwen35moe", "llama", "qwen2", "qwen3"])
def test_standalone_file_config(tmp_path, monkeypatch, arch):
    path = tmp_path / "standalone.gguf"
    write_fixture(path, arch)

    def forbid_hf_config(*args, **kwargs):
        pytest.fail("standalone GGUF must not read an HF config")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", forbid_hf_config)
    config = load_gguf_config(path)
    assert config.num_hidden_layers == 4
    assert config.head_dim == 64
    assert config.hidden_size == 256
    assert config.vocab_size == 4
    assert config.eos_token_id == 3
    assert config.dtype == torch.float16
    assert config.rope_parameters["rope_theta"] == 10000000.0
    if arch.startswith("qwen35"):
        assert config.linear_value_head_dim == 32
        assert config.layer_types == [
            "linear_attention",
            "full_attention",
            "linear_attention",
            "full_attention",
        ]
        assert config.rope_parameters["mrope_section"] == [3, 3, 2]
    if arch == "qwen35moe":
        assert config.moe_intermediate_size == 640
        assert config.num_experts == 8
        assert config.num_experts_per_tok == 2
        assert config.norm_topk_prob


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("qwen35.ssm.inner_size", 255, "not divisible"),
        ("qwen35.attention.recurrent_layers", [True], "length"),
        ("qwen35.rope.scaling.type", "llama3", "explicit --hf-config-path"),
        ("qwen35.embedding_length", 0, "positive integer"),
        ("qwen35.ssm.state_size", None, "Missing required"),
    ],
)
def test_reject_incomplete_or_unsupported_metadata(tmp_path, key, value, error):
    path = tmp_path / "model.gguf"
    write_fixture(path)
    metadata = read_gguf_metadata(path)
    if value is None:
        del metadata[key]
    else:
        metadata[key] = value
    with pytest.raises(ValueError, match=error):
        gguf_config_dict(metadata)


def test_missing_shard_fails_before_reading(tmp_path):
    path = tmp_path / "model-00001-of-00002.gguf"
    write_fixture(path)
    with pytest.raises(FileNotFoundError, match="model-00002-of-00002.gguf"):
        gguf_shard_paths(path)


def test_metadata_does_not_decode_tensor_payloads(tmp_path, monkeypatch):
    path = tmp_path / "new-quant-type.gguf"
    write_fixture(path)

    def forbidden(*args, **kwargs):
        pytest.fail("configuration reader attempted to decode tensor payloads")

    monkeypatch.setattr(gguf.GGUFReader, "_build_tensors", forbidden)
    assert load_gguf_config(path).hidden_size == 256


def test_hub_subfolder_resolves_all_checkpoint_shards(tmp_path, monkeypatch):
    from vllm.transformers_utils import gguf_files

    calls = []

    def download(repo, filename, **kwargs):
        calls.append((repo, filename, kwargs))
        return str(tmp_path / filename)

    monkeypatch.setattr(gguf_files, "hf_hub_download", download)
    path = gguf_files.resolve_gguf_file(
        "owner/model/IQ3_XXS/weights-00002-of-00002.gguf",
        revision="pinned",
        cache_dir="cache",
        token=True,
    )
    assert path == tmp_path / "IQ3_XXS/weights-00002-of-00002.gguf"
    assert [call[:2] for call in calls] == [
        ("owner/model", "IQ3_XXS/weights-00001-of-00002.gguf"),
        ("owner/model", "IQ3_XXS/weights-00002-of-00002.gguf"),
    ]
    assert all(
        kwargs == {"revision": "pinned", "cache_dir": "cache", "token": True}
        for _, _, kwargs in calls
    )


def test_split_indices_and_duplicate_tensors(tmp_path):
    first = tmp_path / "model-01-of-02.gguf"
    second = tmp_path / "model-02-of-02.gguf"
    write_fixture(first, extra={"split.no": 0, "split.count": 2})
    write_fixture(second, extra={"split.no": 1, "split.count": 2})
    paths = gguf_shard_paths(second)
    assert paths == [first, second]
    with pytest.raises(ValueError, match="Duplicate GGUF tensor"):
        gguf_tensor_index(paths)


def test_split_tensor_count_validation(tmp_path):
    path = tmp_path / "model.gguf"
    write_fixture(path, extra={"split.tensors.count": 2})
    with pytest.raises(ValueError, match="declares 2 tensors, found 1"):
        gguf_tensor_index([path])


def test_integrated_config_does_not_use_transformers_gguf_parser(tmp_path, monkeypatch):
    from vllm.transformers_utils.config import get_config

    path = tmp_path / "standalone.gguf"
    write_fixture(path)

    def forbidden(*args, **kwargs):
        pytest.fail("native GGUF called Transformers GGUF/config parser")

    monkeypatch.setattr(PretrainedConfig, "get_config_dict", forbidden)
    cfg = get_config(
        path, trust_remote_code=False, hf_overrides_kw={"max_position_embeddings": 4096}
    )
    assert cfg.architectures == ["Qwen3_5ForCausalLM"]
    assert cfg.max_position_embeddings == 4096


def test_native_qwen_tokenizer_nfc_and_special_token_policy():
    from tokenizers import pre_tokenizers

    from vllm.tokenizers.gguf import tokenizer_from_gguf_metadata

    tokens = sorted(pre_tokenizers.ByteLevel.alphabet()) + ["<|im_end|>", "<think>"]
    metadata = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen35",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.token_type": [1] * 256 + [3, 4],
        "tokenizer.ggml.eos_token_id": 256,
        "tokenizer.chat_template": "{{ messages[0]['content'] }}",
    }
    tok = tokenizer_from_gguf_metadata(metadata)
    assert tok.encode("é") == tok.encode("e\u0301")
    ids = tok.encode("<think>你好<|im_end|>")
    assert ids[0] == 257 and ids[-1] == 256
    assert tok.decode(ids, skip_special_tokens=True) == "<think>你好"
    assert (
        tok.apply_chat_template([{"role": "user", "content": "Hi"}], tokenize=False)
        == "Hi"
    )
