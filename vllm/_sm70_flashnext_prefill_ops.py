# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lazy loader for custom SM70 Qwen3.8-Flash-Next prefill CUDA operators."""

from __future__ import annotations

import importlib

import torch

from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling

_LOADED: bool | None = None


@torch.compiler.assume_constant_result
def has_sm70_flashnext_prefill_ops() -> bool:
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    ns = getattr(torch.ops, "vllm_sm70_fn_prefill", None)
    if ns is not None and hasattr(ns, "moe_prefill_raw"):
        _LOADED = True
        _ensure_dynamo_ops()
        return True
    try:
        importlib.import_module("vllm._sm70_flashnext_prefill_C")
    except ImportError:
        _LOADED = False
        return False
    ns = getattr(torch.ops, "vllm_sm70_fn_prefill", None)
    _LOADED = bool(ns is not None and hasattr(ns, "moe_prefill_raw"))
    if _LOADED:
        _ensure_dynamo_ops()
    return _LOADED


_DYNAMO_OPS_REGISTERED = False
_PACKED_SHARED_GATE_UP: dict[tuple[int, int, int], torch.Tensor] = {}
_PACKED_ROUTER_SHARED_840: dict[tuple[int, int], torch.Tensor] = {}
_PACKED_QSA_4224: dict[tuple[int, int, int], torch.Tensor] = {}
_PENDING_SHARED_ACT: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
_PENDING_SHARED_GATE_LOGITS: dict[tuple[int, int], torch.Tensor] = {}
_PENDING_QSA_FOLDED8: dict[int, tuple[int, torch.Tensor, torch.Tensor]] = {}


def pop_pending_qsa_folded8(
    device_idx: int, idx_ptr: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    entry = _PENDING_QSA_FOLDED8.pop(device_idx, None)
    if entry is not None and entry[0] == idx_ptr:
        return entry[1], entry[2]
    return None


def pop_pending_shared_gate_logits(
    device_idx: int, down_weight_ptr: int
) -> torch.Tensor | None:
    return _PENDING_SHARED_GATE_LOGITS.pop((device_idx, down_weight_ptr), None)


def wait_hc_ag_peers_if_needed(x: torch.Tensor | None = None) -> None:
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    if resources is None:
        return
    owner = resources.get("hc_prefill_peer")
    if (
        owner is not None
        and getattr(owner, "_ag_pull_pending", False)
        and (x is None or x.device == owner.device)
    ):
        owner.wait_ag_peers_ready()


def _linear_with_ag_overlap(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if owner is not None and getattr(owner, "_ag_pull_pending", False):
        if (
            x.ndim == 2
            and x.is_contiguous()
            and x.device == owner.device
            and int(x.data_ptr()) == getattr(owner, "_ag_pending_ptr", 0)
        ):
            m = x.shape[0]
            step = int(owner._ag_pending_local_rows)
            first = min(m, int(owner.rank) * step)
            last = min(m, first + step)
            out = torch.empty((m, weight.shape[0]), dtype=x.dtype, device=x.device)
            if last > first:
                torch.mm(x[first:last], weight.T, out=out[first:last])
            owner.wait_ag_peers_ready()
            if first > 0:
                torch.mm(x[:first], weight.T, out=out[:first])
            if last < m:
                torch.mm(x[last:], weight.T, out=out[last:])
            return out
        owner.wait_ag_peers_ready()
    return torch.nn.functional.linear(x, weight)


def _linear2_with_ag_overlap(
    x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm.runtime_resources import current_runtime_resources

    resources = current_runtime_resources()
    owner = resources.get("hc_prefill_peer") if resources is not None else None
    if owner is not None and getattr(owner, "_ag_pull_pending", False):
        if (
            x.ndim == 2
            and x.is_contiguous()
            and x.device == owner.device
            and int(x.data_ptr()) == getattr(owner, "_ag_pending_ptr", 0)
        ):
            m = x.shape[0]
            step = int(owner._ag_pending_local_rows)
            first = min(m, int(owner.rank) * step)
            last = min(m, first + step)
            out1 = torch.empty((m, w1.shape[0]), dtype=x.dtype, device=x.device)
            if last > first:
                torch.mm(x[first:last], w1.T, out=out1[first:last])
            owner.wait_ag_peers_ready()
            if first > 0:
                torch.mm(x[:first], w1.T, out=out1[:first])
            if last < m:
                torch.mm(x[last:], w1.T, out=out1[last:])
            out2 = torch.nn.functional.linear(x, w2)
            return out1, out2
        owner.wait_ag_peers_ready()
    return torch.nn.functional.linear(x, w1), torch.nn.functional.linear(x, w2)


def has_packed_router_shared(device_idx: int, router_ptr: int) -> bool:
    return (device_idx, router_ptr) in _PACKED_ROUTER_SHARED_840


def has_packed_qsa_4224(device_idx: int, qkv_ptr: int, idx_ptr: int) -> bool:
    return (device_idx, qkv_ptr, idx_ptr) in _PACKED_QSA_4224


def prepack_shared_expert_weights(model: torch.nn.Module) -> None:
    if not has_sm70_flashnext_prefill_ops():
        return
    # 1) Zero-extra-VRAM standalone [328, 2560] = [shared_gu(320); shared_eg(1); pad(7)]
    for mod in model.modules():
        if not getattr(mod, "_sm70_exact_shared_expert_gate", False):
            continue
        gu_proj = getattr(mod, "gate_up_proj", None)
        eg_proj = getattr(mod, "expert_gate", None)
        gu_w = getattr(gu_proj, "weight", None)
        gt_w = getattr(eg_proj, "weight", None)
        if not (
            isinstance(gu_w, torch.Tensor)
            and isinstance(gt_w, torch.Tensor)
            and gu_w.is_cuda
            and gt_w.is_cuda
            and gu_w.dtype == gt_w.dtype == torch.float16
            and gu_w.shape == (320, 2560)
            and gt_w.shape == (1, 2560)
        ):
            continue
        dev_idx = gu_w.device.index or 0
        if (
            dev_idx,
            int(gu_w.data_ptr()),
            int(gt_w.data_ptr()),
        ) in _PACKED_SHARED_GATE_UP:
            continue
        packed_w = torch.zeros((328, 2560), dtype=torch.float16, device=gu_w.device)
        packed_w[:320].copy_(gu_w)
        packed_w[320:321].copy_(gt_w)
        gu_w.data = packed_w[:320]
        w_key = (dev_idx, int(gu_w.data_ptr()), int(gt_w.data_ptr()))
        _PACKED_SHARED_GATE_UP[w_key] = packed_w

    # 2) Zero-extra-VRAM QSA [4224, 2560] = [qkv(3584); idx_qk(640)]
    for mod in model.modules():
        qkv = getattr(mod, "qkv_proj", None)
        indexer = getattr(mod, "indexer", None)
        idx_proj = getattr(indexer, "index_qk_proj", None)
        if qkv is None or idx_proj is None:
            continue
        qkv_w = getattr(qkv, "weight", None)
        idx_w = getattr(idx_proj, "weight", None)
        if not (
            isinstance(qkv_w, torch.Tensor)
            and isinstance(idx_w, torch.Tensor)
            and qkv_w.is_cuda
            and idx_w.is_cuda
            and qkv_w.dtype == idx_w.dtype == torch.float16
            and qkv_w.shape == (3584, 2560)
            and idx_w.shape == (640, 2560)
            and getattr(qkv, "bias", None) is None
            and getattr(idx_proj, "bias", None) is None
        ):
            continue
        dev_idx = qkv_w.device.index or 0
        if (dev_idx, int(qkv_w.data_ptr()), int(idx_w.data_ptr())) in _PACKED_QSA_4224:
            qkv._fn_packed_4224 = True  # type: ignore[attr-defined]
            continue
        packed_4224 = torch.empty(
            (4224, 2560), dtype=torch.float16, device=qkv_w.device
        )
        packed_4224[:3584].copy_(qkv_w)
        packed_4224[3584:].copy_(idx_w)
        qkv_w.data = packed_4224[:3584]
        idx_w.data = packed_4224[3584:]
        _PACKED_QSA_4224[(dev_idx, int(qkv_w.data_ptr()), int(idx_w.data_ptr()))] = (
            packed_4224
        )
        qkv._fn_packed_4224 = True  # type: ignore[attr-defined]


def _fn_router_shared_prefill(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    m = x.shape[0]
    dev_idx = x.device.index or 0
    packed_840 = _PACKED_ROUTER_SHARED_840.get((dev_idx, int(router_weight.data_ptr())))
    if (
        packed_840 is not None
        and m > 64
        and x.ndim == 2
        and x.shape[1] == 2560
        and x.dtype == torch.float16
        and not torch.cuda.is_current_stream_capturing()
    ):
        x_c = x if x.is_contiguous() else x.contiguous()
        raw = _linear_with_ag_overlap(x_c, packed_840)
        router_logits = torch.empty((m, 512), dtype=torch.float16, device=x_c.device)
        act_out = torch.empty((m, 160), dtype=torch.float16, device=x_c.device)
        gate_logits = torch.empty((m, 1), dtype=torch.float16, device=x_c.device)
        torch.ops.vllm_sm70_fn_prefill.split_router_silu_mul_gate(
            router_logits, act_out, gate_logits, raw
        )
        del raw
        _PENDING_SHARED_ACT[(dev_idx, int(gate_up_weight.data_ptr()))] = (
            act_out,
            gate_logits,
        )
        return router_logits
    wait_hc_ag_peers_if_needed(x)
    return torch.nn.functional.linear(x, router_weight)


def _fn_router_shared_prefill_fake(
    x: torch.Tensor,
    router_weight: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    del router_weight, gate_up_weight
    return x.new_empty((*x.shape[:-1], 512))


def maybe_fused_moe_router_shared_prefill(
    hidden_states: torch.Tensor,
    gate_proj: torch.nn.Module,
    shared_expert: torch.nn.Module | None,
) -> torch.Tensor | None:
    if shared_expert is None or is_sm70_decode_graph_compiling():
        return None
    router_w = getattr(gate_proj, "weight", None)
    gu_proj = getattr(shared_expert, "gate_up_proj", None)
    gu_w = getattr(gu_proj, "weight", None)
    if not (
        getattr(gate_proj, "_fn_packed_840", False)
        and isinstance(router_w, torch.Tensor)
        and isinstance(gu_w, torch.Tensor)
        and router_w.is_cuda
        and has_sm70_flashnext_prefill_ops()
    ):
        return None
    return torch.ops.vllm.qwen38_fn_router_shared_prefill(hidden_states, router_w, gu_w)


def _fn_qsa_input_prefill(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    idx_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m = x.shape[0]
    if m <= 2048:
        dev_idx = x.device.index or 0
        packed = _PACKED_QSA_4224.get(
            (dev_idx, int(qkv_weight.data_ptr()), int(idx_weight.data_ptr()))
        )
        if packed is not None:
            both = (
                _linear_with_ag_overlap(x, packed)
                if x.ndim == 2 and not torch.cuda.is_current_stream_capturing()
                else torch.nn.functional.linear(x, packed)
            )
            return both[..., :3584].contiguous(), both[..., 3584:].contiguous()
    if x.ndim == 2 and not torch.cuda.is_current_stream_capturing():
        if (m & 7) == 0 and m > 2048 and idx_weight.shape == (640, 2560):
            from vllm.config.sm70_sparse import sparse_policy
            from vllm.forward_context import get_forward_context
            from vllm.runtime_resources import current_runtime_resources

            resources = current_runtime_resources()
            owner = resources.get("hc_prefill_peer") if resources is not None else None
            meta = get_forward_context().attn_metadata
            if isinstance(meta, list):
                meta = meta[0]
            single_req_tp4 = False
            if (
                owner is not None
                and getattr(owner, "p2p_enabled", False)
                and sparse_policy().value("qsa_indexer_cublas")
                and isinstance(meta, dict)
            ):
                for v_meta in meta.values():
                    bt = getattr(v_meta, "block_table", None)
                    num_act = getattr(v_meta, "num_actual_tokens", None)
                    if bt is not None and num_act is not None:
                        single_req_tp4 = bt.shape[0] == 1 and int(num_act) == m
                        break
            if single_req_tp4 and owner is not None:
                qkv_out = _linear_with_ag_overlap(x, qkv_weight)
                raw_keys = torch.nn.functional.linear(x, idx_weight[512:])
                h_step = m >> 3
                rk = int(owner.rank)
                q_w_t = idx_weight[:512].T
                proj_q_fold = x.new_empty((2 * h_step, 512))
                torch.mm(
                    x[rk * h_step : (rk + 1) * h_step],
                    q_w_t,
                    out=proj_q_fold[:h_step],
                )
                torch.mm(
                    x[(7 - rk) * h_step : (8 - rk) * h_step],
                    q_w_t,
                    out=proj_q_fold[h_step:],
                )
                idx_stub = x.new_empty((m, 640))
                dev_idx = x.device.index or 0
                _PENDING_QSA_FOLDED8[dev_idx] = (
                    int(idx_stub.data_ptr()),
                    proj_q_fold,
                    raw_keys,
                )
                return qkv_out, idx_stub
        return _linear2_with_ag_overlap(x, qkv_weight, idx_weight)
    wait_hc_ag_peers_if_needed(x)
    return (
        torch.nn.functional.linear(x, qkv_weight),
        torch.nn.functional.linear(x, idx_weight),
    )


def _fn_qsa_input_prefill_fake(
    x: torch.Tensor,
    qkv_weight: torch.Tensor,
    idx_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    del qkv_weight, idx_weight
    return (
        x.new_empty((*x.shape[:-1], 3584)),
        x.new_empty((*x.shape[:-1], 640)),
    )


def _fn_gdn_input_prefill(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if x.ndim == 2 and not torch.cuda.is_current_stream_capturing():
        if x.is_contiguous():
            x_c = x
        else:
            wait_hc_ag_peers_if_needed(x)
            x_c = x.contiguous()
        qkvz, ba = _linear2_with_ag_overlap(x_c, qkvz_weight, ba_weight)
    else:
        wait_hc_ag_peers_if_needed(x)
        x_c = x if x.is_contiguous() else x.contiguous()
        qkvz = torch.nn.functional.linear(x_c, qkvz_weight)
        ba = torch.nn.functional.linear(x_c, ba_weight)
    m = qkvz.shape[0]
    if m >= 256 and not torch.cuda.is_current_stream_capturing():
        return qkvz[:, :2560], qkvz[:, 2560:], ba[:, :12], ba[:, 12:]
    z = qkvz.new_empty((m, 1536))
    b = qkvz.new_empty((m, 12))
    a = qkvz.new_empty((m, 12))
    if m > 0:
        torch.ops.vllm_sm70_fn_prefill.split_gdn_qkvz_ba_tails(z, b, a, qkvz, ba)
    return qkvz[:, :2560], z, b, a


def _fn_gdn_input_prefill_fake(
    x: torch.Tensor,
    qkvz_weight: torch.Tensor,
    ba_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del qkvz_weight, ba_weight
    m = x.shape[0]
    qkvz = x.new_empty((m, 4096))
    return (
        qkvz[:, :2560],
        x.new_empty((m, 1536)),
        x.new_empty((m, 12)),
        x.new_empty((m, 12)),
    )


def _fn_shared_expert_prefill(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    batch_allowed: bool = True,
    defer_gate_allowed: bool = False,
) -> torch.Tensor:
    m = x.shape[0]
    if (
        m > 64
        and x.ndim == 2
        and x.shape[1] == 2560
        and gate_up_weight.shape == (320, 2560)
        and down_weight.shape == (2560, 160)
        and gate_weight.shape == (1, 2560)
        and x.dtype
        == gate_up_weight.dtype
        == down_weight.dtype
        == gate_weight.dtype
        == torch.float16
        and not torch.cuda.is_current_stream_capturing()
    ):
        dev_idx = x.device.index or 0
        precomputed = _PENDING_SHARED_ACT.pop(
            (dev_idx, int(gate_up_weight.data_ptr())), None
        )
        if precomputed is not None and precomputed[0].shape[0] == m:
            wait_hc_ag_peers_if_needed(x)
            act_out, gate_logits = precomputed
        else:
            wait_hc_ag_peers_if_needed(x)
            x_c = x if x.is_contiguous() else x.contiguous()
            w_key = (
                dev_idx,
                int(gate_up_weight.data_ptr()),
                int(gate_weight.data_ptr()),
            )
            packed_w = _PACKED_SHARED_GATE_UP.get(w_key)
            if packed_w is None:
                packed_w = torch.zeros(
                    (328, 2560), dtype=torch.float16, device=x_c.device
                )
                packed_w[:320].copy_(gate_up_weight)
                packed_w[320:321].copy_(gate_weight)
                _PACKED_SHARED_GATE_UP[w_key] = packed_w
            raw = torch.nn.functional.linear(x_c, packed_w)
            act_out = torch.empty((m, 160), dtype=torch.float16, device=x_c.device)
            gate_logits = torch.empty((m, 1), dtype=torch.float16, device=x_c.device)
            torch.ops.vllm_sm70_fn_prefill.silu_mul_split_gate(
                act_out, gate_logits, raw
            )
            del raw
        down_out = torch.nn.functional.linear(act_out, down_weight)
        del act_out
        if defer_gate_allowed and m >= 1024:
            _PENDING_SHARED_GATE_LOGITS[(dev_idx, int(down_weight.data_ptr()))] = (
                gate_logits
            )
            return down_out
        torch.ops.vllm_sm70_fn_prefill.shared_gate_mul_inplace(down_out, gate_logits)
        return down_out

    wait_hc_ag_peers_if_needed(x)
    gate_up = torch.nn.functional.linear(x, gate_up_weight)
    act_out = torch.empty((m, gate_up.shape[1] // 2), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(act_out, gate_up)
    import vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv  # noqa: F401

    return torch.ops.vllm.qwen38_sm70_shared_down_gated(
        x, act_out, down_weight, gate_weight, batch_allowed
    )


def _fn_shared_expert_prefill_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_weight: torch.Tensor,
    batch_allowed: bool = True,
    defer_gate_allowed: bool = False,
) -> torch.Tensor:
    del gate_up_weight, down_weight, gate_weight, batch_allowed, defer_gate_allowed
    return x.new_empty((*x.shape[:-1], 2560))


def _fn_shared_gate_mul(out: torch.Tensor, gate_logits: torch.Tensor) -> torch.Tensor:
    if (
        out.ndim == 2
        and out.shape[0] > 1
        and out.shape[1] == 2560
        and out.dtype == torch.float16
        and out.is_contiguous()
        and gate_logits.shape == (out.shape[0], 1)
        and gate_logits.dtype == torch.float16
        and gate_logits.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    ):
        torch.ops.vllm_sm70_fn_prefill.shared_gate_mul_inplace(out, gate_logits)
        return out
    return torch.sigmoid(gate_logits) * out


def _fn_shared_gate_mul_fake(
    out: torch.Tensor, gate_logits: torch.Tensor
) -> torch.Tensor:
    del gate_logits
    return torch.empty_like(out)


def _fn_shared_gate_dot_mul(
    out: torch.Tensor, x: torch.Tensor, gate_weight: torch.Tensor
) -> torch.Tensor:
    if (
        out.ndim == 2
        and out.shape[0] > 1
        and out.shape[1] == 2560
        and out.dtype == torch.float16
        and out.is_contiguous()
        and x.shape == out.shape
        and x.dtype == torch.float16
        and x.is_contiguous()
        and gate_weight.shape == (1, 2560)
        and gate_weight.dtype == torch.float16
        and gate_weight.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    ):
        torch.ops.vllm_sm70_fn_prefill.shared_gate_dot_mul_inplace(out, x, gate_weight)
        return out
    gate_logits = torch.nn.functional.linear(x, gate_weight)
    return torch.sigmoid(gate_logits) * out


def _fn_shared_gate_dot_mul_fake(
    out: torch.Tensor, x: torch.Tensor, gate_weight: torch.Tensor
) -> torch.Tensor:
    del x, gate_weight
    return torch.empty_like(out)


def _ensure_dynamo_ops() -> None:
    global _DYNAMO_OPS_REGISTERED
    if _DYNAMO_OPS_REGISTERED:
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    if not hasattr(torch.ops.vllm, "qwen38_fn_shared_gate_mul"):
        direct_register_custom_op(
            "qwen38_fn_shared_gate_mul",
            _fn_shared_gate_mul,
            mutates_args=["out"],
            fake_impl=_fn_shared_gate_mul_fake,
        )
    if not hasattr(torch.ops.vllm, "qwen38_fn_shared_gate_dot_mul"):
        direct_register_custom_op(
            "qwen38_fn_shared_gate_dot_mul",
            _fn_shared_gate_dot_mul,
            mutates_args=["out"],
            fake_impl=_fn_shared_gate_dot_mul_fake,
        )
    if not hasattr(torch.ops.vllm, "qwen38_fn_shared_expert_prefill"):
        direct_register_custom_op(
            "qwen38_fn_shared_expert_prefill",
            _fn_shared_expert_prefill,
            fake_impl=_fn_shared_expert_prefill_fake,
        )
    if not hasattr(torch.ops.vllm, "qwen38_fn_router_shared_prefill"):
        direct_register_custom_op(
            "qwen38_fn_router_shared_prefill",
            _fn_router_shared_prefill,
            fake_impl=_fn_router_shared_prefill_fake,
        )
    if not hasattr(torch.ops.vllm, "qwen38_fn_qsa_input_prefill"):
        direct_register_custom_op(
            "qwen38_fn_qsa_input_prefill",
            _fn_qsa_input_prefill,
            fake_impl=_fn_qsa_input_prefill_fake,
        )
    if not hasattr(torch.ops.vllm, "qwen38_fn_gdn_input_prefill"):
        direct_register_custom_op(
            "qwen38_fn_gdn_input_prefill",
            _fn_gdn_input_prefill,
            fake_impl=_fn_gdn_input_prefill_fake,
        )
    _DYNAMO_OPS_REGISTERED = True
