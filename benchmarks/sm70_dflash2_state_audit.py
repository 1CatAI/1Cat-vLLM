# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in fixed-prefix diagnostic; its forced acceptance is never a speed result.

Set VLLM_SM70_DFLASH2_AUDIT_ROOT and use StateAuditExtension as the worker
extension. The root's active-case.json contains name, prompt_ids and token_ids
(the complete forced tape). Remove that file for ordinary requests. Each worker
records prefill and verifier inputs, state selection, outputs and native logits.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from pathlib import Path

import torch


def gather_state(pool: torch.Tensor, indices: torch.Tensor) -> dict[str, torch.Tensor]:
    """Snapshot indexed slots without disguising padding as live slot zero."""
    indices = indices.reshape(-1).to(device=pool.device, dtype=torch.int64)
    valid = (indices >= 0) & (indices < pool.shape[0])
    values = pool.index_select(0, indices.clamp(0, pool.shape[0] - 1))
    mask = valid.reshape((-1,) + (1,) * (values.ndim - 1))
    return {
        "indices": indices.clone(),
        "valid": valid,
        "values": torch.where(mask, values, 0),
    }


def selected_ssm_slots(indices: torch.Tensor, selectors: torch.Tensor) -> torch.Tensor:
    """The recurrent verifier reads the preceding accepted slot, not column 0."""
    if indices.ndim != 2 or selectors.numel() != indices.shape[0]:
        raise ValueError("Expected [requests, slots] and one selector per request")
    columns = selectors.reshape(-1, 1).to(torch.int64) - 1
    valid = (columns >= 0) & (columns < indices.shape[1])
    slots = indices.gather(1, columns.clamp(0, indices.shape[1] - 1))
    return torch.where(valid, slots, -1).reshape(-1)


def install() -> None:
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gd
    from vllm.model_executor.models import qwen3_next as qn
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    root = Path(os.environ["VLLM_SM70_DFLASH2_AUDIT_ROOT"])
    mode = os.environ.get("VLLM_SM70_DFLASH2_AUDIT_MODE", "control")
    directory = root / "captures" / mode
    directory.mkdir(parents=True, exist_ok=True)
    layers = {
        int(x)
        for x in os.environ.get("VLLM_SM70_DFLASH2_AUDIT_LAYERS", "0,1").split(",")
    }
    buffers: dict[str, torch.Tensor] = {}
    markers: dict[str, torch.Tensor] = {}
    epoch: torch.Tensor | None = None

    def caller_layer() -> int | None:
        if epoch is None:
            return None
        # Functionalization may clone cache arguments or reuse their addresses.
        # Bind observations to the actual GDN call, never a storage pointer.
        frame = sys._getframe(2)
        for _ in range(8):
            module = frame.f_locals.get("self")
            if isinstance(module, gd.QwenGatedDeltaNetAttention):
                layer = gd._sm70_gdn_layer_idx(module.prefix)
                return layer if layer in layers else None
            frame = frame.f_back
            if frame is None:
                break
        return None

    def record(key: str, tensor: torch.Tensor | None) -> None:
        if tensor is None or epoch is None:
            return
        key = f"{key}:{tuple(tensor.shape)}"
        if key not in buffers:
            buffers[key] = torch.empty_like(tensor)
            markers[key] = torch.empty_like(epoch)
        buffers[key].copy_(tensor)
        markers[key].copy_(epoch)

    def record_slots(key: str, pool: torch.Tensor, indices: torch.Tensor) -> None:
        for label, tensor in gather_state(pool, indices).items():
            record(f"{key}/{label}", tensor)

    initialize = GPUModelRunner.initialize_kv_cache

    @functools.wraps(initialize)
    def initialize_kv_cache(self, *args, **kwargs):
        nonlocal epoch
        result = initialize(self, *args, **kwargs)
        epoch = torch.full((1,), -1, device=self.device, dtype=torch.int64)
        for name, module in self.model.named_modules():
            if not isinstance(module, gd.QwenGatedDeltaNetAttention):
                continue
            layer = gd._sm70_gdn_layer_idx(name)
            if layer not in layers:
                continue
            original = module.chunk_gated_delta_rule.forward

            def chunk(*args, _original=original, _layer=layer, **kwargs):
                key = f"prefill/layer{_layer}/recurrent"
                for label in (
                    "q",
                    "k",
                    "v",
                    "g",
                    "beta",
                    "cu_seqlens",
                    "has_initial_state",
                ):
                    record(f"{key}/{label}", kwargs.get(label))
                state = kwargs.get("initial_state")
                indices = kwargs.get("state_indices")
                if state is not None:
                    if indices is None:
                        record(f"{key}/input_state", state)
                    else:
                        record_slots(f"{key}/input_state", state, indices)
                out = _original(*args, **kwargs)
                record(f"{key}/output", out[0])
                if indices is not None:
                    record_slots(f"{key}/output_state", state, indices)
                else:
                    record(f"{key}/output_state", out[1])
                return out

            module.chunk_gated_delta_rule.forward = chunk
        return result

    GPUModelRunner.initialize_kv_cache = initialize_kv_cache
    recurrent = gd.fused_recurrent_gated_delta_rule

    @functools.wraps(recurrent)
    def recurrent_wrapper(*args, **kwargs):
        state = kwargs.get("initial_state")
        indices = kwargs.get("ssm_state_indices")
        selectors = kwargs.get("num_accepted_tokens")
        layer = caller_layer()
        if layer is None or indices is None or selectors is None:
            return recurrent(*args, **kwargs)
        key = f"verify/layer{layer}/recurrent"
        record(f"route/verify/layer{layer}/split", epoch)
        for label in ("q", "k", "v", "g", "beta", "cu_seqlens"):
            record(f"{key}/{label}", kwargs.get(label))
        record(f"{key}/slot_table", indices)
        record(f"{key}/selectors", selectors)
        record_slots(
            f"{key}/input_state", state, selected_ssm_slots(indices, selectors)
        )
        out = recurrent(*args, **kwargs)
        record(f"{key}/output", out[0])
        record_slots(f"{key}/output_states", state, indices)
        return out

    gd.fused_recurrent_gated_delta_rule = recurrent_wrapper
    packed = gd.fused_sigmoid_gating_delta_rule_update_mixed_qkv_out

    @functools.wraps(packed)
    def packed_wrapper(*args, **kwargs):
        layer = caller_layer()
        if layer is None or kwargs.get("precomputed_g") is None:
            return packed(*args, **kwargs)
        # Keep the same raw metadata coverage as the split verifier. The
        # packed operator receives only the live slice of these parent args.
        frame = sys._getframe(1)
        for _ in range(8):
            if frame.f_code.co_name == "_forward_dflash2_packed_gdn_verify":
                indices = frame.f_locals["spec_state_indices_tensor"]
                selectors = frame.f_locals["spec_state_slot_selectors"]
                break
            frame = frame.f_back
            if frame is None:
                raise RuntimeError("Missing packed-verifier metadata provenance")
        else:
            raise RuntimeError("Missing packed-verifier caller")
        key = f"verify/layer{layer}/recurrent"
        record(f"route/verify/layer{layer}/packed", epoch)
        mixed = kwargs["mixed_qkv"]
        tokens = mixed.shape[0]
        q_heads, v_heads = kwargs["num_q_heads"], kwargs["num_v_heads"]
        dk, dv = kwargs["head_k_dim"], kwargs["head_v_dim"]
        q, k, v = mixed.split([q_heads * dk, q_heads * dk, v_heads * dv], dim=1)
        for label, tensor in (
            ("q", q.reshape(1, tokens, q_heads, dk)),
            ("k", k.reshape(1, tokens, q_heads, dk)),
            ("v", v.reshape(1, tokens, v_heads, dv)),
            ("g", kwargs["precomputed_g"].reshape(1, tokens, v_heads)),
            ("beta", kwargs["precomputed_beta"].reshape(1, tokens, v_heads)),
            ("cu_seqlens", kwargs["cu_seqlens"]),
            ("slot_table", indices),
            ("selectors", selectors),
        ):
            record(f"{key}/{label}", tensor)
        state = kwargs["initial_state"]
        record_slots(
            f"{key}/input_state", state, selected_ssm_slots(indices, selectors)
        )
        out = packed(*args, **kwargs)
        record(f"{key}/output", out[0].transpose(0, 1))
        record_slots(f"{key}/output_states", state, indices)
        return out

    gd.fused_sigmoid_gating_delta_rule_update_mixed_qkv_out = packed_wrapper
    for function_name, phase in (
        ("causal_conv1d_fn", "prefill"),
        ("causal_conv1d_update", "verify"),
    ):
        original = getattr(gd, function_name)

        @functools.wraps(original)
        def convolution(*args, _original=original, _phase=phase, **kwargs):
            state = kwargs.get("conv_states") if _phase == "prefill" else args[1]
            indices = kwargs.get(
                "cache_indices" if _phase == "prefill" else "conv_state_indices"
            )
            layer = caller_layer()
            if layer is None or indices is None:
                return _original(*args, **kwargs)
            key = f"{_phase}/layer{layer}/conv"
            record(f"{key}/input", args[0])
            for label in (
                "has_initial_state",
                "num_accepted_tokens",
                "query_start_loc",
            ):
                record(f"{key}/{label}", kwargs.get(label))
            record_slots(f"{key}/input_state", state, indices)
            out = _original(*args, **kwargs)
            record(f"{key}/output", out)
            record_slots(f"{key}/output_state", state, indices)
            return out

        setattr(gd, function_name, convolution)

    def active(self, batch):
        if not hasattr(self, "_state_audit_requests"):
            self._state_audit_requests = {}
        request_id = batch.req_ids[0]
        if request_id not in self._state_audit_requests:
            path = root / "active-case.json"
            entry = None
            if path.exists():
                case = json.loads(path.read_text())
                entry = {
                    "case": case,
                    "tape": torch.tensor(
                        case["token_ids"], device=self.device, dtype=torch.int64
                    ),
                    "step": 0,
                }
            self._state_audit_requests[request_id] = entry
        return self._state_audit_requests[request_id]

    prepare = GPUModelRunner.prepare_inputs

    @functools.wraps(prepare)
    def prepare_inputs(self, *args, **kwargs):
        batch = prepare(self, *args, **kwargs)
        if epoch is not None:
            epoch.add_(1)
        entry = active(self, batch)
        if entry is not None:
            if batch.num_reqs != 1:
                raise ValueError(
                    "Fixed-prefix state audit requires exactly one request"
                )
            batch.input_ids[: batch.num_tokens].copy_(
                entry["tape"][batch.positions[: batch.num_tokens]]
            )
        return batch

    GPUModelRunner.prepare_inputs = prepare_inputs
    sample = GPUModelRunner.sample

    @functools.wraps(sample)
    def sample_fixed_prefix(self, hidden, batch, grammar):
        entry = active(self, batch)
        if entry is None:
            return sample(self, hidden, batch, grammar)
        rank = torch.distributed.get_rank()
        step = entry["step"]
        phase = "prefill" if step == 0 else "verify"
        hs = hidden[batch.logits_indices]
        positions = batch.positions[batch.logits_indices]
        native = self.model.compute_logits(hs)

        def cpu(tensor):
            return tensor.detach().cpu().clone()

        assert epoch is not None
        current_epoch = int(epoch.item())
        keys = list(markers)
        epochs = torch.cat([markers[key] for key in keys]).cpu().tolist()
        fresh = {
            key for key, observed in zip(keys, epochs) if observed == current_epoch
        }
        tensors = {
            key: cpu(value)
            for key, value in buffers.items()
            if key.startswith(phase + "/") and key in fresh
        }
        layer_tensors = {
            key: {**qn._SM70_QWEN_LAYER_GRAPH_META[key], "tensor": cpu(value)}
            for key, value in qn._SM70_QWEN_LAYER_GRAPH_BUFFERS.items()
            if value.ndim and value.shape[0] in (batch.num_tokens, hidden.shape[0])
        }
        result = {
            "rank": rank,
            "step": step,
            "case": entry["case"]["name"],
            "phase": phase,
            "positions": cpu(positions),
            "input_ids": cpu(batch.input_ids[: batch.num_tokens]),
            "hidden": cpu(hs),
            "native_logits": cpu(native) if rank == 0 else None,
            "states": tensors,
            "verifier_routes": sorted(
                key.split(":")[0] for key in fresh if key.startswith("route/")
            ),
            "tensors": layer_tensors,
            "cuda_rng": torch.cuda.get_rng_state(self.device),
            "cpu_rng": torch.get_rng_state(),
            "num_draft_tokens": batch.num_draft_tokens,
            "capture_epoch": current_epoch,
            "expected_layers": sorted(layers),
            "sampling": {
                name: cpu(
                    getattr(self.sampler.sampling_states, name).gpu.index_select(
                        0, batch.idx_mapping.to(torch.int64)
                    )
                )
                for name in ("seeds", "temperature", "top_k", "top_p", "min_p")
            },
        }
        torch.save(
            result, directory / f"{entry['case']['name']}-rank{rank}-step{step}.pt"
        )
        entry["step"] += 1
        next_ids = entry["tape"][positions + 1].view(1, -1).to(torch.int32)
        count = torch.full(
            (1,), next_ids.shape[1], device=self.device, dtype=torch.int32
        )
        # Teacher forcing is a diagnostic control, never an acceptance metric.
        return (
            SamplerOutput(next_ids, None, None, count),
            count,
            torch.zeros_like(count),
        )

    GPUModelRunner.sample = sample_fixed_prefix


class StateAuditExtension:
    pass


if os.getenv("VLLM_SM70_DFLASH2_AUDIT_ROOT"):
    install()
