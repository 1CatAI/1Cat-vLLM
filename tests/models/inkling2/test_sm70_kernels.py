# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage 3 Gate 1: SM70 kernel parity tests for Inkling2 on Volta V100."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.quantization.sm70_turbomind import unpack_mxfp4_weight
from vllm.models.inkling2.nvidia.ops.activation import softcapped_situ_sm70
from vllm.models.inkling2.nvidia.ops.attention import rel_decode_attention
from vllm.models.inkling2.nvidia.ops.attention_sm90 import MODE_OUT_LSE, rel_attention
from vllm.models.inkling2.nvidia.ops.attn_res import attn_res
from vllm.models.inkling2.nvidia.ops.loop import blend_rms_norm, pair_rms_norm
from vllm.models.inkling2.nvidia.ops.moe_sm70 import Inkling2Sm70Experts
from vllm.models.inkling2.nvidia.ops.norm import rms_norm
from vllm.models.inkling2.nvidia.ops.sconv import (
    attention_prologue,
    sconv_seq_metadata,
    shift_rel_proj,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_device_capability(70),
    reason="Requires Volta SM70 GPU",
)


@pytest.fixture(scope="module", autouse=True)
def _sm70_dist_env():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29591",
            local_rank=0,
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        try:
            yield
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()


def test_situ_sm70():
    torch.manual_seed(0)
    M, N = 17, 1152
    gate_up = torch.randn((M, 2 * N), device="cuda", dtype=torch.float16) * 2.5
    out = torch.empty((M, N), device="cuda", dtype=torch.float16)
    softcapped_situ_sm70(gate_up, out, 4.0, 25.0)
    g = gate_up[:, 0::2].float()
    u = gate_up[:, 1::2].float()
    ref = (4.0 * torch.tanh(g / 4.0)) * torch.sigmoid(g) * (25.0 * torch.tanh(u / 25.0))
    diff = (out.float() - ref).abs().max().item()
    print(f"[PASS] softcapped_situ_sm70 max_abs_diff={diff:.6f}")
    assert diff < 0.05, f"SiTU diff too large: {diff}"


def test_norm_attnres_loop_sconv():
    torch.manual_seed(1)
    T, D, Q = 19, 2304, 3
    eps = 1e-6

    x = torch.randn((T, D), device="cuda", dtype=torch.float32)
    w = torch.randn((D,), device="cuda", dtype=torch.float16)
    y = rms_norm(x, w, eps, torch.float16)
    assert y.dtype == torch.float16
    x_rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    ref_y = (x_rms * w.float()).to(torch.float16)
    diff = (y.float() - ref_y.float()).abs().max().item()
    print(f"[PASS] rms_norm(fp32->fp16) max_abs_diff={diff:.6f}")
    assert diff < 0.02

    # attn_res (FP16/BF16 act dtype for stack/last/addend, FP32 for queries)
    n_stack = 2
    stack = torch.randn((n_stack, T, D), device="cuda", dtype=torch.float16)
    last = torch.randn((T, D), device="cuda", dtype=torch.float16)
    addend = torch.randn((T, D), device="cuda", dtype=torch.float16)
    queries = torch.randn((Q, D), device="cuda", dtype=torch.float32)
    norm_w = torch.randn((Q, D), device="cuda", dtype=torch.float16)
    out = attn_res(
        stack,
        last,
        addend,
        queries,
        norm_w,
        eps,
        True,
        eps=eps,
    )
    assert out.dtype == torch.float16 and out.shape == (Q, T, D)

    sources = torch.cat(
        [stack.float(), (last.float() + addend.float()).unsqueeze(0)], dim=0
    )  # (S, T, D)
    s_norm = sources * torch.rsqrt(sources.pow(2).mean(-1, keepdim=True) + eps)
    scores = torch.einsum("qd,std->qst", queries, s_norm)
    probs = torch.softmax(scores, dim=1)
    mix = torch.einsum("qst,std->qtd", probs, s_norm)
    mix_norm = mix * torch.rsqrt(mix.pow(2).mean(-1, keepdim=True) + eps)
    ref_out = mix_norm * norm_w.float().unsqueeze(1)
    diff_ar = (out.float() - ref_out).abs().max().item()
    print(f"[PASS] attn_res max_abs_diff={diff_ar:.6f}")
    assert diff_ar < 0.02

    # loop ops: pair_rms_norm / blend_rms_norm expect (S, T, D)
    S = 4
    dest = torch.randn((S, T, D), device="cuda", dtype=torch.float32)
    msg = torch.randn((S, T, D), device="cuda", dtype=torch.float32)
    p = pair_rms_norm(dest, msg)
    assert p.dtype == torch.float16 and p.shape == (S, T, 2 * D)
    gf = torch.randn((S, T, 2 * D), device="cuda", dtype=torch.float32)
    w_loop = torch.randn((S, D), device="cuda", dtype=torch.float16)
    blended = blend_rms_norm(dest, msg, gf, w_loop, eps)
    assert blended.dtype == torch.float16 and blended.shape == (S, T, D)

    # sconv metadata
    qstart = torch.tensor([0, 7, 19], device="cuda", dtype=torch.int32)
    seq_idx = torch.empty((24,), device="cuda", dtype=torch.int32)
    query_start = torch.empty((24,), device="cuda", dtype=torch.int32)
    sconv_seq_metadata(qstart, 2, 19, seq_idx, query_start, 24)
    assert (
        seq_idx[6].item() == 0
        and seq_idx[7].item() == 1
        and seq_idx[18].item() == 1
        and seq_idx[19].item() == 0
        and query_start[7].item() == 7
    )
    print("[PASS] loop blend + sconv_seq_metadata")


def test_rel_attention():
    torch.manual_seed(2)
    H, H_kv, D, E = 24, 8, 128, 1024
    G = H // H_kv
    page = 16
    window = 1024
    for mode, n_tok in [("decode", 1), ("prefill", 35)]:
        num_blocks = 8
        key_cache = (
            torch.randn((num_blocks, page, H_kv, D), device="cuda", dtype=torch.float16)
            * 0.25
        )
        val_cache = (
            torch.randn((num_blocks, page, H_kv, D), device="cuda", dtype=torch.float16)
            * 0.25
        )
        kv_len = 48 if mode == "decode" else n_tok
        bt = torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32)
        seq_lens = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
        cu_q = torch.tensor([0, n_tok], device="cuda", dtype=torch.int32)
        q = torch.randn((n_tok, H, D), device="cuda", dtype=torch.float16) * 0.25
        # Unsheared ref: put zeros or structured bias
        rel = torch.zeros((n_tok + 128, H, E + 256), device="cuda", dtype=torch.float16)
        out = torch.empty((n_tok, H, D), device="cuda", dtype=torch.float32)
        lse = torch.empty((H, n_tok), device="cuda", dtype=torch.float32)
        scale = D**-0.5
        rel_attention(
            q,
            key_cache,
            val_cache,
            out,
            lse,
            rel,
            block_table=bt,
            cache_seqlens=seq_lens,
            cu_seqlens_q=cu_q,
            max_query_len=n_tok,
            softmax_scale=scale,
            window=window,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        # Dense PyTorch ref
        k_seq = torch.cat([key_cache[b] for b in bt[0]], dim=0)[:kv_len]
        v_seq = torch.cat([val_cache[b] for b in bt[0]], dim=0)[:kv_len]
        k_exp = k_seq.repeat_interleave(G, dim=1)  # (kv_len, H, D)
        v_exp = v_seq.repeat_interleave(G, dim=1)
        scores = torch.einsum("thd,shd->hts", q.float(), k_exp.float()) * scale
        past = kv_len - n_tok
        qi = (past + torch.arange(n_tok, device="cuda"))[:, None]
        ki = torch.arange(kv_len, device="cuda")[None, :]
        mask = (ki <= qi) & ((qi - ki) < window)
        scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        ref = torch.einsum("hts,shd->thd", probs, v_exp.float())
        diff = (out - ref).abs().max().item()
        print(
            f"[PASS] rel_attention({mode}, T={n_tok}, kv={kv_len}) "
            f"max_abs_diff={diff:.6f}"
        )
        assert diff < 0.02

    # Also test rel_decode_attention (gate & dec_out are (T, H * D))
    gate = torch.randn((1, H * D), device="cuda", dtype=torch.float16)
    sinks = torch.randn((1, H), device="cuda", dtype=torch.float32) * 0.5
    dec_out = torch.empty((1, H * D), device="cuda", dtype=torch.float16)
    key_cache = (
        torch.randn((8, page, H_kv, D), device="cuda", dtype=torch.float16) * 0.25
    )
    val_cache = (
        torch.randn((8, page, H_kv, D), device="cuda", dtype=torch.float16) * 0.25
    )
    rel_decode_attention(
        torch.randn((1, H, D), device="cuda", dtype=torch.float16) * 0.25,
        key_cache,
        val_cache,
        torch.zeros((1 + 128, H, E + 256), device="cuda", dtype=torch.float16),
        gate,
        sinks,
        dec_out,
        block_table=torch.tensor([[0, 1, 2, 3]], device="cuda", dtype=torch.int32),
        cache_seqlens=torch.tensor([48], device="cuda", dtype=torch.int32),
        cu_seqlens_q=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        max_query_len=1,
        num_actual_tokens=1,
        softmax_scale=D**-0.5,
        is_local=True,
        num_splits=2,
    )
    assert torch.isfinite(dec_out).all()
    print("[PASS] rel_decode_attention")


def _dequant_nvfp4_ref(w_packed, w_scale_e4m3, w_amax, w_denom):
    # w_packed: [E, N, K//2] uint8 -> unpacked [-6..6] float -> split K in 16s
    w_unp = unpack_mxfp4_weight(
        w_packed, is_nvfp4=True
    ).float()  # wait: is_nvfp4? check below
    return w_unp


def test_experts_nvfp4_and_mxfp8():
    from types import SimpleNamespace

    torch.manual_seed(3)
    num_experts, topk, K, inter = 16, 4, 256, 128
    cfg = SimpleNamespace(
        n_routed_experts=num_experts,
        num_experts_per_tok=topk,
        hidden_size=K,
        intermediate_size=inter,
        mlp_gate_softcap=4.0,
        mlp_up_softcap=25.0,
    )

    # Test MXFP8 path
    exp_mxfp8 = Inkling2Sm70Experts(cfg, "mxfp8").cuda()
    # Populate MXFP8 weights (E, 2*inter, K) F8_E4M3 + (E, 2*inter, K//32) U8 E8M0
    rng_w13 = (torch.randn((num_experts, 2 * inter, K), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    rng_s13 = torch.randint(
        123, 126, (num_experts, 2 * inter, K // 32), device="cuda", dtype=torch.uint8
    )
    rng_w2 = (torch.randn((num_experts, K, inter), device="cuda") * 0.5).to(
        torch.float8_e4m3fn
    )
    rng_s2 = torch.randint(
        123, 126, (num_experts, K, inter // 32), device="cuda", dtype=torch.uint8
    )
    exp_mxfp8.w13_weight.data.copy_(rng_w13)
    exp_mxfp8.w13_weight_scale.data.copy_(rng_s13)
    exp_mxfp8.w2_weight.data.copy_(rng_w2)
    exp_mxfp8.w2_weight_scale.data.copy_(rng_s2)
    exp_mxfp8.process_weights_after_loading()

    for M in (1, 4, 24):
        x = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.25
        topk_ids = torch.randint(
            0, num_experts, (M, topk), device="cuda", dtype=torch.int32
        )
        for r in range(M):
            topk_ids[r] = torch.randperm(num_experts, device="cuda", dtype=torch.int32)[
                :topk
            ]
        topk_w = torch.rand((M, topk), device="cuda", dtype=torch.float32) + 0.1
        out = exp_mxfp8(x, topk_ids, topk_w, reduce=False)

        # Reference FP32
        w13_fp32 = exp_mxfp8.w13_fp16.float()
        w2_fp32 = exp_mxfp8.w2_fp16.float()
        ref = torch.zeros((M, K), device="cuda", dtype=torch.float32)
        for m in range(M):
            for k_idx in range(topk):
                e = int(topk_ids[m, k_idx].item())
                h = x[m].float() @ w13_fp32[e].t()
                g, u = h[0::2], h[1::2]
                act = (
                    (4.0 * torch.tanh(g / 4.0))
                    * torch.sigmoid(g)
                    * (25.0 * torch.tanh(u / 25.0))
                )
                ref[m] += topk_w[m, k_idx] * (act @ w2_fp32[e].t())
        diff = (out.float() - ref).abs().max().item()
        rel = diff / max(ref.abs().max().item(), 1e-6)
        print(
            f"[PASS] Inkling2Sm70Experts(mxfp8, M={M}) max_abs={diff:.6f} rel={rel:.6f}"
        )
        assert rel < 0.01 and diff < 0.05

    # Test NVFP4 path
    exp_nvfp4 = Inkling2Sm70Experts(cfg, "nvfp4").cuda()
    exp_nvfp4.w13_weight.data.copy_(
        torch.randint(
            0, 256, exp_nvfp4.w13_weight.shape, device="cuda", dtype=torch.uint8
        )
    )
    exp_nvfp4.w13_weight_scale.data.copy_(
        torch.ones(
            exp_nvfp4.w13_weight_scale.shape, device="cuda", dtype=torch.float16
        ).to(torch.float8_e4m3fn)
    )
    exp_nvfp4.w13_weight_amax.data.fill_(16.0)
    exp_nvfp4.w13_weight_outer_scale_denominator.data.fill_(2688.0)
    exp_nvfp4.w2_weight.data.copy_(
        torch.randint(
            0, 256, exp_nvfp4.w2_weight.shape, device="cuda", dtype=torch.uint8
        )
    )
    exp_nvfp4.w2_weight_scale.data.copy_(
        torch.ones(
            exp_nvfp4.w2_weight_scale.shape, device="cuda", dtype=torch.float16
        ).to(torch.float8_e4m3fn)
    )
    exp_nvfp4.w2_weight_amax.data.fill_(16.0)
    exp_nvfp4.w2_weight_outer_scale_denominator.data.fill_(2688.0)

    # Save fp32 dequant ref BEFORE process_weights_after_loading frees weights
    _FP4_LUT = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device="cuda",
        dtype=torch.float32,
    )

    def _nvfp4_to_fp32(w_u8, s_e4m3, amax, denom):
        # unpack_mxfp4_weight(w_u8[e]) returns [K, N] nibbles 0..15 -> [N, K]
        outs = []
        eff_global = amax.float() / denom.float()
        for e in range(w_u8.shape[0]):
            nib = unpack_mxfp4_weight(w_u8[e]).t().long()  # [N, K]
            vals = _FP4_LUT[nib]
            scales = (
                (s_e4m3[e].float() * eff_global[e]).half().float()
            )  # match FP16 scale round
            outs.append(
                (vals.view(vals.shape[0], -1, 16) * scales.unsqueeze(-1)).view(
                    vals.shape
                )
            )
        return torch.stack(outs)

    w13_ref = _nvfp4_to_fp32(
        exp_nvfp4.w13_weight.data,
        exp_nvfp4.w13_weight_scale.data,
        exp_nvfp4.w13_weight_amax.data,
        exp_nvfp4.w13_weight_outer_scale_denominator.data,
    )
    w2_ref = _nvfp4_to_fp32(
        exp_nvfp4.w2_weight.data,
        exp_nvfp4.w2_weight_scale.data,
        exp_nvfp4.w2_weight_amax.data,
        exp_nvfp4.w2_weight_outer_scale_denominator.data,
    )
    exp_nvfp4.process_weights_after_loading()

    for M in (1, 4, 24):
        x = torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.5
        topk_ids = torch.randint(
            0, num_experts, (M, topk), device="cuda", dtype=torch.int32
        )
        for r in range(M):
            topk_ids[r] = torch.randperm(num_experts, device="cuda", dtype=torch.int32)[
                :topk
            ]
        topk_w = torch.rand((M, topk), device="cuda", dtype=torch.float32) + 0.1
        out = exp_nvfp4(x, topk_ids, topk_w, reduce=False)
        ref = torch.zeros((M, K), device="cuda", dtype=torch.float32)
        for m in range(M):
            for k_idx in range(topk):
                e = int(topk_ids[m, k_idx].item())
                h = x[m].float() @ w13_ref[e].t()
                g, u = h[0::2], h[1::2]
                act = (
                    (4.0 * torch.tanh(g / 4.0))
                    * torch.sigmoid(g)
                    * (25.0 * torch.tanh(u / 25.0))
                )
                ref[m] += topk_w[m, k_idx] * (act @ w2_ref[e].t())
        diff = (out.float() - ref).abs().max().item()
        rel = diff / max(ref.abs().max().item(), 1e-6)
        print(
            f"[PASS] Inkling2Sm70Experts(nvfp4, M={M}) max_abs={diff:.6f} rel={rel:.6f}"
        )
        assert rel < 0.01 and diff < 0.05


def test_rel_attention_fp8_kv():
    """Compare attention_prologue FP8 E4M3 writer + _load_fp8_kv reader vs FP16."""
    torch.manual_seed(42)
    H, H_kv, D, d_rel, E = 24, 8, 128, 16, 1024
    W = 4
    qk_clip, rel_clip, eps = 22.6274, 8.0, 1e-6
    # Per-element scales, shaped like the model's `q_scale` / `r_scale`.
    q_scale = torch.full((H, D), D**-0.5, device="cuda", dtype=torch.float32)
    r_scale = torch.full((H, d_rel), 0.25, device="cuda", dtype=torch.float32)
    rel_proj = shift_rel_proj(
        torch.randn((d_rel, E), device="cuda", dtype=torch.float16) * 0.1,
        is_local=False,
    )
    k_w = torch.randn((H_kv * D, W), device="cuda", dtype=torch.float16) * 0.25
    v_w = torch.randn((H_kv * D, W), device="cuda", dtype=torch.float16) * 0.25

    for mode, n_tok, kv_len in [
        ("prefill", 35, 35),
        ("decode", 1, 48),
        ("decode", 1, 4096),
        ("decode", 1, 65536),
    ]:
        torch.manual_seed(100 + kv_len)
        page_16, page_8 = 32, 64
        num_blk_16 = (kv_len + page_16 - 1) // page_16
        num_blk_8 = (kv_len + page_8 - 1) // page_8

        k_in = torch.randn((kv_len, H_kv * D), device="cuda", dtype=torch.float16) * 0.5
        v_in = torch.randn((kv_len, H_kv * D), device="cuda", dtype=torch.float16) * 0.5
        q_in = torch.randn((kv_len, H * D), device="cuda", dtype=torch.float16) * 0.3
        r_in = (
            torch.randn((kv_len, H * d_rel), device="cuda", dtype=torch.float16) * 0.3
        )
        pos_all = torch.arange(kv_len, device="cuda", dtype=torch.int32)
        slots_all = torch.arange(kv_len, device="cuda", dtype=torch.int64)
        seq_idx_all = torch.zeros((kv_len,), device="cuda", dtype=torch.int32)
        qstart_all = torch.zeros((kv_len,), device="cuda", dtype=torch.int32)
        conv_bt = torch.arange(
            (kv_len + 3) // 4, device="cuda", dtype=torch.int32
        ).unsqueeze(0)
        conv_slots = torch.arange(kv_len, device="cuda", dtype=torch.int64)
        conv_16 = torch.zeros(
            ((kv_len + 3) // 4, H_kv, 4, 2048), device="cuda", dtype=torch.float16
        )
        conv_8 = conv_16.clone()

        k_cache_16 = torch.zeros(
            (num_blk_16, page_16, H_kv, D), device="cuda", dtype=torch.float16
        )
        v_cache_16 = torch.zeros_like(k_cache_16)
        k_cache_8 = torch.zeros(
            (num_blk_8, page_8, H_kv, D), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e4m3fn)
        v_cache_8 = torch.zeros(
            (num_blk_8, page_8, H_kv, D), device="cuda", dtype=torch.uint8
        ).view(torch.float8_e4m3fn)

        cq = torch.tensor([0, kv_len], device="cuda", dtype=torch.int32)
        sl = torch.tensor([kv_len], device="cuda", dtype=torch.int32)
        cq_out_16 = torch.empty((kv_len, H, D), device="cuda", dtype=torch.float16)
        rel_out_16 = torch.zeros(
            (kv_len + 128, H, E + 256), device="cuda", dtype=torch.float16
        )
        cq_out_8 = torch.empty_like(cq_out_16)
        rel_out_8 = torch.zeros_like(rel_out_16)

        for conv_c, kc, vc, cqo, ro in (
            (conv_16, k_cache_16, v_cache_16, cq_out_16, rel_out_16),
            (conv_8, k_cache_8, v_cache_8, cq_out_8, rel_out_8),
        ):
            attention_prologue(
                q_in.view(kv_len, H, D),
                r_in.view(kv_len, H, d_rel),
                k_in,
                v_in,
                q_scale,
                r_scale,
                None,
                rel_proj,
                k_w,
                v_w,
                None,
                None,
                conv_c,
                pos_all,
                conv_bt,
                seq_idx_all,
                conv_slots,
                qstart_all,
                kc,
                vc,
                slots_all,
                cqo,
                ro,
                cq,
                sl,
                clip_threshold=qk_clip,
                rel_threshold=rel_clip,
                norm_eps=eps,
            )

        q_eval_16 = cq_out_16[-n_tok:]
        q_eval_8 = cq_out_8[-n_tok:]
        rel_eval_16 = rel_out_16[kv_len - n_tok :]
        rel_eval_8 = rel_out_8[kv_len - n_tok :]
        bt_16 = torch.arange(num_blk_16, device="cuda", dtype=torch.int32).unsqueeze(0)
        bt_8 = torch.arange(num_blk_8, device="cuda", dtype=torch.int32).unsqueeze(0)
        cu_eval = torch.tensor([0, n_tok], device="cuda", dtype=torch.int32)
        out_16 = torch.empty((n_tok, H, D), device="cuda", dtype=torch.float32)
        out_8 = torch.empty_like(out_16)
        lse_16 = torch.empty((H, n_tok), device="cuda", dtype=torch.float32)
        lse_8 = torch.empty_like(lse_16)
        rel_attention(
            q_eval_16,
            k_cache_16,
            v_cache_16,
            out_16,
            lse_16,
            rel_eval_16,
            block_table=bt_16,
            cache_seqlens=sl,
            cu_seqlens_q=cu_eval,
            max_query_len=n_tok,
            softmax_scale=D**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        rel_attention(
            q_eval_8,
            k_cache_8,
            v_cache_8,
            out_8,
            lse_8,
            rel_eval_8,
            block_table=bt_8,
            cache_seqlens=sl,
            cu_seqlens_q=cu_eval,
            max_query_len=n_tok,
            softmax_scale=D**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        out_8_deq = torch.empty_like(out_16)
        lse_deq = torch.empty_like(lse_16)
        rel_attention(
            q_eval_8,
            k_cache_8.to(torch.float16),
            v_cache_8.to(torch.float16),
            out_8_deq,
            lse_deq,
            rel_eval_8,
            block_table=bt_8,
            cache_seqlens=sl,
            cu_seqlens_q=cu_eval,
            max_query_len=n_tok,
            softmax_scale=D**-0.5,
            window=0,
            num_splits=1,
            mode=MODE_OUT_LSE,
        )
        assert torch.isfinite(out_8).all() and torch.isfinite(out_16).all()
        codec_diff = (out_8 - out_8_deq).abs().max().item()
        abs_diff = (out_8 - out_16).abs()
        max_abs = abs_diff.max().item()
        mean_abs = abs_diff.mean().item()
        rel_l2 = (out_8 - out_16).norm().item() / max(out_16.norm().item(), 1e-6)
        print(
            f"[PASS] rel_attention_fp8_kv({mode}, T={n_tok}, kv={kv_len}) "
            f"codec_diff={codec_diff:.6f} rel_l2_vs_fp16={rel_l2:.4f} "
            f"max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}"
        )
        assert codec_diff <= 1e-4, f"FP8 KV codec mismatch: {codec_diff}"
        assert rel_l2 <= 0.05, f"FP8 KV L2 quantization error too large: {rel_l2}"


def main():
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:29591",
            local_rank=0,
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
        try:
            test_situ_sm70()
            test_norm_attnres_loop_sconv()
            test_rel_attention()
            test_rel_attention_fp8_kv()
            test_experts_nvfp4_and_mxfp8()
        finally:
            destroy_model_parallel()
            destroy_distributed_environment()
    print("ALL_STAGE3_GATE1_PARITY_TESTS_PASSED")


if __name__ == "__main__":
    main()
