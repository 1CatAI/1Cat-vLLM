# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

from vllm.config import CUDAGraphMode
from vllm.models.qwen4_exp.nvidia import mtp


@pytest.mark.parametrize(
    "rows,accepted",
    [
        (5, False),
        (4095, False),
        (4096, True),
        (4097, True),
        (8192, True),
        (16384, True),
        (16385, False),
    ],
)
def test_mtp_row_admission_retains_decode_precision(monkeypatch, rows, accepted):
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction", True
    )
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_fp16_accumulation", False)
    monkeypatch.setattr(mtp, "is_forward_context_available", lambda: True)
    context = NS(cudagraph_runtime_mode=CUDAGraphMode.NONE)
    monkeypatch.setattr(mtp, "get_forward_context", lambda: context)
    owner = NS(_hc_prefill_sequence_qualified=True)
    hidden = torch.empty(rows, 10240, dtype=torch.half, device="meta")
    positions = torch.empty(rows, device="meta")
    select = mtp.Qwen4ExpMTP._use_hc_prefill_sequence_rows
    assert select(owner, positions, hidden, None) is accepted
    assert torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
    for mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
        context.cudagraph_runtime_mode = mode
        assert not select(owner, positions, hidden, None)
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    assert not select(owner, positions, None, None)
    assert not select(owner, positions, hidden.float(), None)
    assert not select(owner, positions, hidden[:, :-1], None)
    assert not select(owner, positions, hidden, object())
    owner._hc_prefill_sequence_qualified = False
    assert not select(owner, positions, hidden, None)


def test_mtp_prefill_view_borrows_weights_and_state():
    class Owner(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.register_buffer("state", torch.zeros(1))

        def forward(self, ids, positions, hidden, *args, hc_sequence_parallel):
            assert hc_sequence_parallel
            self.state.add_(1)
            return ids * self.weight, hidden.clone()

    owner = Owner()
    cfg = NS(compilation_config=NS(mode=0))
    view = mtp._Qwen4ExpMTPPrefillSequenceModel(target_model=owner, vllm_config=cfg)
    assert not list(view.parameters()) and not list(view.buffers())
    hidden = torch.ones(2, 4)
    sample, multi = view(torch.ones(2), torch.arange(2), hidden)
    assert torch.equal(sample, torch.ones(2)) and torch.equal(multi, hidden)
    assert multi.data_ptr() != hidden.data_ptr()
    assert torch.equal(owner.state, torch.ones(1))
    assert view._target_model is owner


@pytest.mark.parametrize("aot", [False, True])
def test_mtp_singleton_first_compilation_preserves_token_axes(
    monkeypatch, tmp_path, aot
):
    import json

    from vllm.compilation.decorators import support_torch_compile
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        ModelConfig,
        VllmConfig,
        set_current_vllm_config,
    )
    from vllm.config.compilation import (
        DynamicShapesConfig,
        DynamicShapesType,
        PassConfig,
    )
    from vllm.forward_context import set_forward_context

    monkeypatch.setenv("VLLM_USE_AOT_COMPILE", "1" if aot else "0")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    config_dir = tmp_path / "model"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gpt2",
                "architectures": ["GPT2LMHeadModel"],
                "n_embd": 8,
                "n_head": 2,
                "n_layer": 1,
                "n_positions": 128,
                "vocab_size": 32,
                "bos_token_id": 1,
                "eos_token_id": 2,
            }
        )
    )

    class Layer(nn.Module):
        def forward(self, *, hidden_states, positions, **kwargs):
            block = hidden_states + positions.to(hidden_states.dtype)[:, None]
            return hidden_states, block, torch.zeros_like(hidden_states)

    class Mixer(nn.Module):
        def combine_and_mix(self, hidden, block, injection):
            multi = hidden + block
            return multi, multi[:, :4], None

    @support_torch_compile(
        dynamic_arg_dims={**mtp._MTP_TOKEN_AXES, "intermediate_tensors": 0},
    )
    class Predictor(nn.Module):
        forward = mtp.Qwen4ExpMultiTokenPredictor.forward

        def __init__(self):
            super().__init__()
            self.hc_count, self.hidden_size, self.num_mtp_layers = 2, 4, 1
            self.layers = nn.ModuleList([Layer()])
            self.hyper_connection_mixer = Mixer()
            for name in (
                "pre_fc_norm_embedding",
                "pre_fc_norm_hidden",
                "fc_embedding",
                "fc_hidden",
            ):
                setattr(self, name, nn.Identity())

    cfg = VllmConfig(
        model_config=ModelConfig(
            model=str(config_dir),
            dtype="float32",
            max_model_len=128,
            skip_tokenizer_init=True,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            compile_sizes=[],
            cudagraph_mode=CUDAGraphMode.NONE,
            dynamic_shapes_config=DynamicShapesConfig(type=DynamicShapesType.UNBACKED),
            pass_config=PassConfig(
                fuse_norm_quant=False,
                fuse_act_quant=False,
                fuse_attn_quant=False,
                fuse_allreduce_rms=False,
                fuse_minimax_qk_norm=False,
                enable_qk_norm_rope_fusion=False,
                fuse_rope_kvcache=False,
                fuse_rope_kvcache_cat_mla=False,
                fuse_mla_dual_rms_norm=False,
            ),
        ),
    )
    with torch.no_grad(), set_current_vllm_config(cfg), set_forward_context({}, cfg):
        owner = Predictor()
        for rows in (1, 6, 16, 2, 1, 64):
            ids = torch.arange(rows)
            positions = torch.arange(rows) + 7
            hidden = torch.arange(rows * 8).float().reshape(rows, 8) / 128
            embeds = torch.arange(rows * 4).float().reshape(rows, 4) / 64
            reference = owner.forward(ids, positions, hidden, inputs_embeds=embeds)
            actual = owner(ids, positions, hidden, inputs_embeds=embeds)
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

            assert actual[0].shape == (rows, 4) and actual[1].shape == (rows, 8)
        if aot:
            restored = Predictor()
            ids, positions = torch.arange(6), torch.arange(6) + 81
            hidden, embeds = torch.ones(6, 8), torch.ones(6, 4)
            actual = restored(ids, positions, hidden, inputs_embeds=embeds)
            reference = restored.forward(ids, positions, hidden, inputs_embeds=embeds)
            assert restored.was_aot_compile_fn_loaded_from_disk
            for a, b in zip(actual, reference):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_mtp_fallback_compiler_copy_preserves_other_owners():
    from vllm.config import CompilationConfig
    from vllm.config.compilation import DynamicShapesType

    root = NS(compilation_config=CompilationConfig(), kernel_config=object())
    root.compilation_config.cache_dir = "original"
    root.compilation_config.traced_files = {"original.py"}
    assert mtp._make_mtp_prefill_fallback_config(root, qualified=False) is root
    copied = mtp._make_mtp_prefill_fallback_config(root, qualified=True)
    assert copied.kernel_config is root.kernel_config
    assert (
        copied.compilation_config.dynamic_shapes_config.type
        == DynamicShapesType.UNBACKED
    )
    assert (
        root.compilation_config.dynamic_shapes_config.type == DynamicShapesType.BACKED
    )
    assert copied.compilation_config.cache_dir == ""
    assert root.compilation_config.cache_dir == "original"
    copied.compilation_config.traced_files.add("new.py")
    assert root.compilation_config.traced_files == {"original.py"}


def test_mtp_constructor_binds_the_explicit_fallback_config(monkeypatch):
    from vllm.config import (
        CompilationConfig,
        VllmConfig,
        get_current_vllm_config,
        set_current_vllm_config,
    )
    from vllm.config.compilation import DynamicShapesType

    seen = []

    class Predictor(nn.Module):
        def __init__(self, *, vllm_config, **kwargs):
            super().__init__()
            seen.append((vllm_config, get_current_vllm_config()))
            self.layers = nn.ModuleList()
            self.make_empty_intermediate_tensors = lambda *args: None

    monkeypatch.setattr(mtp, "Qwen4ExpMultiTokenPredictor", Predictor)
    monkeypatch.setattr(mtp, "_hc_mtp_prefill_sequence_qualified", lambda *args: True)
    monkeypatch.setattr(
        mtp, "get_pp_group", lambda: NS(world_size=1, is_last_rank=True)
    )
    monkeypatch.setattr(mtp, "ParallelLMHead", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(mtp, "LogitsProcessor", lambda *args: nn.Identity())
    monkeypatch.setattr(mtp.Qwen4ExpMTP, "set_moe_parameters", lambda *args: None)
    for name in (
        "enable_qwen4_exp_low_latency_gemm",
        "enable_qwen38_sm70_fp16_gemv",
        "enable_qwen38_sm70_fp16_fused_hc",
    ):
        monkeypatch.setattr(mtp, name, lambda *args: None)
    root = VllmConfig(compilation_config=CompilationConfig(mode=0))
    root.model_config = NS(
        hf_text_config=NS(vocab_size=32, hidden_size=2560, tie_word_embeddings=False),
        dtype=torch.half,
    )
    root.cache_config.mamba_cache_mode = "align"
    with set_current_vllm_config(root):
        mtp.Qwen4ExpMTP(vllm_config=root)
        assert get_current_vllm_config() is root
    explicit, current = seen[0]
    assert current is explicit and current is not root
    assert (
        current.compilation_config.dynamic_shapes_config.type
        == DynamicShapesType.UNBACKED
    )
    assert (
        root.compilation_config.dynamic_shapes_config.type == DynamicShapesType.BACKED
    )
