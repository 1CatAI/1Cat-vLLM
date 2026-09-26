import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEParallelConfig, RoutingMethodType
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.expert_map_manager import ExpertMapManager
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
import vllm_exl3.exl3 as e

torch.manual_seed(0)
MUL1 = e.MUL1_MARKER_SIGNED_INT32
vc = VllmConfig()
with set_current_vllm_config(vc):
    dev='cuda:0'
    K, hidden, inter, nexp = 4, 256, 512, 8
    pc = FusedMoEParallelConfig(tp_size=1, dp_size=1, ep_size=1, pcp_size=1, pcp_rank=0, sp_size=1,
        tp_rank=0, dp_rank=0, ep_rank=0, use_ep=False, all2all_backend='none', enable_eplb=False)
    cfg = FusedMoEConfig(num_experts=nexp, experts_per_token=2, hidden_dim=hidden,
        intermediate_size_per_partition=inter, num_local_experts=nexp, num_logical_experts=nexp,
        activation=MoEActivation.SILU, device=dev, routing_method=RoutingMethodType.Renormalize,
        moe_parallel_config=pc, in_dtype=torch.float16)
    emm = ExpertMapManager(max_num_batched_tokens=64, top_k=2, global_num_experts=nexp,
        num_redundant_experts=0, num_expert_group=None, moe_parallel_config=pc,
        placement_strategy='static', enable_eplb=False)
    qc = e.Exl3Config.from_config({'quant_method': 'exl3', 'bits': K, 'bits_list': []})
    layer = RoutedExperts(layer_name='moe', params_dtype=torch.float16, moe_config=cfg,
        quant_config=qc, expert_map_manager=emm)
    method = qc.get_quant_method(layer, 'moe')
    method.create_weights(layer, nexp, hidden, inter, torch.float16)
    in_tiles, out_tiles, kw = hidden//16, inter//16, K*16
    layer.w13_trellis.copy_(torch.randint(-32768, 32767, (nexp, 2, in_tiles, out_tiles, kw), dtype=torch.int16))
    layer.w2_trellis.copy_(torch.randint(-32768, 32767, (nexp, out_tiles, in_tiles, kw), dtype=torch.int16))
    for t in (layer.w13_suh, layer.w13_svh, layer.w2_suh, layer.w2_svh):
        t.fill_(1.0)
    # Synthetic mode: no codebook markers (real checkpoints always carry
    # mcg or mul1 markers; random trellis is only valid for the plain
    # codebook, so markers are suppressed and the ABI check patched).
    e._check_moe_codebook_markers = lambda *a, **k: None
    layer.to(dev)
    print('param device:', layer.w13_trellis.device, flush=True)
    method.process_weights_after_loading(layer)
    print('inners built:', len(layer._exl3_inners))
    x = torch.randn(4, hidden, dtype=torch.float16, device=dev)
    tw = torch.rand(4, 2, dtype=torch.float16, device=dev)
    tid = torch.tensor([[0, 3], [1, 1], [2, 5], [7, 0]], dtype=torch.long, device=dev)
    y = method.apply(layer, x, tw, tid, None, None)
    print('apply ok:', tuple(y.shape), y.dtype, 'finite:', torch.isfinite(y).all().item())
    print('last apply backend:', layer._exl3_last_apply)

    # reference: per-expert reconstruct + manual MoE
    import exllamav3_ext as ext
    import torch.nn.functional as F
    def expert_weights(ei):
        # reconstruct fills (k-rows, n-cols): w13[j] = (hidden, inter),
        # w2 = (inter, hidden) — the transposed storage orientation.
        w13 = torch.empty((hidden, 2*inter), dtype=torch.half, device=dev)
        w2 = torch.empty((inter, hidden), dtype=torch.half, device=dev)
        for j in (0, 1):
            wj = torch.empty((hidden, inter), dtype=torch.half, device=dev)
            ext.reconstruct(wj, layer.w13_trellis[ei, j], K, False, False)
            w13[:, j*inter:(j+1)*inter] = wj
        ext.reconstruct(w2, layer.w2_trellis[ei], K, False, False)
        return w13, w2
    y_ref = torch.zeros(4, hidden, dtype=torch.float32, device=dev)
    xh_t = torch.empty(1, hidden, dtype=torch.half, device=dev)
    for t in range(4):
        for kk in range(2):
            ei = int(tid[t, kk]); wgt = float(tw[t, kk])
            w13, w2 = expert_weights(ei)
            # each LinearEXL3 = had_out(gemm(had_in(x, suh)), svh)
            xh_t = torch.empty(1, hidden, dtype=torch.half, device=dev)
            ext.had_r_128(x[t:t+1], xh_t, layer.w13_suh[ei, 0], None, 1.0)
            g = (xh_t.float() @ w13[:, :inter].float()).half().contiguous()
            g_t = torch.empty_like(g)
            ext.had_r_128(g, g_t, None, layer.w13_svh[ei, 0], 1.0)
            g = g_t.float()
            ext.had_r_128(x[t:t+1], xh_t, layer.w13_suh[ei, 1], None, 1.0)
            u = (xh_t.float() @ w13[:, inter:].float()).half().contiguous()
            u_t = torch.empty_like(u)
            ext.had_r_128(u, u_t, None, layer.w13_svh[ei, 1], 1.0)
            u = u_t.float()
            act = F.silu(g) * u
            act_h = act.half().contiguous()
            ah2 = torch.empty_like(act_h)
            ext.had_r_128(act_h, ah2, layer.w2_suh[ei], None, 1.0)
            d = ah2.float() @ w2.float()
            d_h = d.half().contiguous()
            d2 = torch.empty_like(d_h)
            ext.had_r_128(d_h, d2, None, layer.w2_svh[ei], 1.0)
            y_ref[t] += wgt * d2.float()[0]
    err = (y.float() - y_ref).abs().max().item()
    scale = max(y_ref.abs().max().item(), 1e-3)
    print('rel err vs manual MoE:', err/scale)
    print('y[0,:4] ', y[0,:4].tolist())
    print('ref[0,:4]', y_ref[0,:4].tolist())
    print('y[1,:4] ', y[1,:4].tolist())
    print('ref[1,:4]', y_ref[1,:4].tolist())
    assert err/scale < 2e-2, 'MoE output mismatch'
    print('MOE E2E PASS')
