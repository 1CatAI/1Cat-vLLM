# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in full-vocabulary teacher logits from the actual MTP target graph."""

from pathlib import Path
from typing import Any

import torch

from vllm.sm70_graph_observer import GraphParityWorkerExtension


class GGUFTeacherWorkerExtension(GraphParityWorkerExtension):
    def read_host_kv_memory(self):
        """Read storage accounting and cache counters outside timed replay."""
        runner = self.model_runner
        config = runner.kv_cache_config
        owners = []
        staging = {}
        resolution = {}
        for name, module in runner.compilation_config.static_forward_context.items():
            state = getattr(module, "host_kv", None)
            if state is None:
                continue
            tensors = [
                state.hot_values,
                state.tags,
                state.stamps,
                state.hands,
                state.page_slots,
                state.epoch,
                state._stats,
            ]
            staging[state.staging.untyped_storage().data_ptr()] = (
                state.staging.untyped_storage().nbytes()
            )
            for tensor in (state.initial, state.resolved, state.remapped):
                storage = tensor.untyped_storage()
                resolution[storage.data_ptr()] = storage.nbytes()
            owners.append(
                {
                    "layer": name,
                    "host_dtype": "fp8_e4m3" if state.fp8 else "float16",
                    "attention_reader": "protected_hot_and_staged_misses",
                    "history_storage": (
                        "device_reference" if state.device_reference else "host"
                    ),
                    "host_bytes": (
                        0
                        if state.device_reference
                        else state.history.nbytes + state.host_scales.nbytes
                    ),
                    "device_history_bytes": (
                        state.history.nbytes + state.scales.nbytes
                        if state.device_reference
                        else 0
                    ),
                    "device_hot_bytes": sum(t.nbytes for t in tensors),
                    "stats": state.stats.cpu().tolist(),
                }
            )
        return {
            "rank": self.rank,
            "blocks": config.num_blocks,
            "host_pool_bytes": sum(
                t.size for t in config.kv_cache_tensors if t.host_backed
            ),
            "device_pool_bytes": sum(
                t.size for t in config.kv_cache_tensors if not t.host_backed
            ),
            "shared_staging_bytes": sum(staging.values()),
            "shared_resolution_bytes": sum(resolution.values()),
            "owners": owners,
            "torch_allocated_bytes": torch.accelerator.memory_allocated(),
            "torch_reserved_bytes": torch.accelerator.memory_reserved(),
        }

    @torch.inference_mode()
    def inspect_ple_snapshots(self, directory: str):
        """Locate the first TP divergence at the PLE residual boundary."""
        import hashlib

        import torch.distributed as dist

        from vllm.distributed import get_tp_group
        from vllm.models.qwen4_exp.nvidia.ple_layer import _PLE_DIAGNOSTIC_BUFFERS

        torch.accelerator.synchronize()
        stages = {key: value.cpu() for key, value in _PLE_DIAGNOSTIC_BUFFERS.items()}
        if not stages:
            raise RuntimeError("PLE M5 snapshots were not recorded")
        weights = {
            name: value.detach().cpu()
            for name, value in self.model_runner.model.named_parameters()
            if ".ple." in name and ".ple_embedding." not in name
        }
        digests = {
            key: hashlib.sha256(
                value.contiguous().reshape(-1).view(torch.uint8).numpy()
            ).hexdigest()
            for key, value in weights.items()
        }
        gathered: list[Any] = [None] * 4
        dist.all_gather_object(
            gathered,
            {"stages": stages, "weight_hashes": digests},
            group=get_tp_group().cpu_group,
        )
        reference = gathered[0]["stages"]
        rows = []
        for key in sorted(reference):
            value, expected = stages[key].double(), reference[key].double()
            difference = value - expected
            rows.append(
                {
                    "stage": key,
                    "max_abs_vs_rank0": float(difference.abs().max()),
                    "relative_l2_vs_rank0": float(
                        difference.norm() / (expected.norm() + 1e-9)
                    ),
                    "finite": bool(torch.isfinite(value).all()),
                    "exactly_equal_vs_rank0": torch.equal(stages[key], reference[key]),
                }
            )
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"stages": stages, "weights": weights}, root / f"ple-rank{self.rank}.pt"
        )
        return {
            "rank": self.rank,
            "scope": "diagnostic-only actual M5 PLE stage equality across TP",
            "rows": rows,
            "weight_hashes": digests,
            "weight_hashes_equal_to_rank0": digests == gathered[0]["weight_hashes"],
        }

    @torch.inference_mode()
    def inspect_hcx_snapshots(self, directory: str):
        """Compare graph-recorded real HC inputs with isolated dense arithmetic.

        This RPC runs between completed requests. Diagnostic copies are in the
        captured target graph, so these runs are not latency measurements.
        """
        import torch.distributed as dist
        import torch.nn.functional as F

        from vllm.distributed import (
            get_tp_group,
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_all_reduce_sum2,
        )
        from vllm.models.qwen4_exp.nvidia.hyperconnection import _PARTIAL_MODULES
        from vllm.models.qwen4_exp.nvidia.ops.hc import (
            hc_combine_norm,
            hc_gate_mix,
            hc_silu,
        )
        from vllm.models.qwen4_exp.nvidia.sm70_hcx import current_hcx_runtime

        runtime = current_hcx_runtime()
        if runtime is None or not runtime.diagnostic or not runtime.snapshots:
            raise RuntimeError("HCX M5 snapshots were not recorded")
        torch.accelerator.synchronize()
        keys = sorted(runtime.snapshots)
        epochs = {key: int(runtime.snapshots[key]["epoch"].item()) for key in keys}
        keys.sort(key=epochs.__getitem__)
        frames = [None] * 4
        dist.all_gather_object(frames, epochs, group=get_tp_group().cpu_group)
        if any(frame != epochs for frame in frames):
            raise RuntimeError(f"HCX snapshot frames differ across TP ranks: {frames}")
        report = []
        failed = []
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        for key in keys:
            snapshot = runtime.snapshots[key]
            module = _PARTIAL_MODULES[key]
            first, second = snapshot["partial"], snapshot.get("secondary")
            reduced = (
                tensor_model_parallel_all_reduce(first)
                if second is None
                else tensor_model_parallel_all_reduce_sum2(first, second)
            )
            hidden, xn = hc_combine_norm(
                snapshot["hidden"],
                reduced,
                snapshot["injection"],
                module.hc_norm.weight,
                module.config.rms_norm_eps,
                module.hc_count,
            )
            down_weight = module.input_mix_weight_down_block_inject.weight
            up_weight = module.input_mix_weight_up.weight
            down = F.linear(xn.float(), down_weight.float()).half()
            gate = F.linear(hc_silu(down[:, :320], 4).float(), up_weight.float()).half()
            expected = {
                "hidden_out": hidden,
                "block_out": hc_gate_mix(xn, gate, 4),
                "injection_out": down[:, 320:324],
            }
            errors = {}
            for name, reference in expected.items():
                actual = snapshot[name].float()
                difference = actual - reference.float()
                errors[name] = {
                    "max_abs": float(difference.abs().max().item()),
                    "relative_l2": float(
                        (difference.norm() / (reference.float().norm() + 1e-9)).item()
                    ),
                    "finite": bool(torch.isfinite(actual).all().item()),
                }
            row = dict(name=key, epoch=epochs[key], errors=errors)
            report.append(row)
            failed.append((key, expected, errors))
        # Write after all coupled collectives; save first and worst boundaries.
        worst = sorted(
            failed,
            key=lambda item: max(error["relative_l2"] for error in item[2].values()),
            reverse=True,
        )[:3]
        selected = {
            key: (expected, errors) for key, expected, errors in failed[:3] + worst
        }
        for key, (expected, errors) in selected.items():
            module = _PARTIAL_MODULES[key]
            torch.save(
                dict(
                    inputs={
                        name: value.cpu()
                        for name, value in runtime.snapshots[key].items()
                    },
                    norm=module.hc_norm.weight.cpu(),
                    down=module.input_mix_weight_down_block_inject.weight.cpu(),
                    up=module.input_mix_weight_up.weight.cpu(),
                    reference={name: value.cpu() for name, value in expected.items()},
                    errors=errors,
                ),
                root / f"rank{self.rank}-{key}.pt",
            )
        return {
            "rank": self.rank,
            "rows": report,
            "scope": "HCX actual M5 graph inputs",
        }

    def start_teacher_capture(self, directory: str, key: str):
        if hasattr(self, "_teacher_original_execute"):
            raise RuntimeError("Teacher capture is already active")
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self._teacher_count = 0
        runner = self.model_runner
        original = runner.execute_model
        self._teacher_original_execute = original

        def execute(*args, **kwargs):
            result = original(*args, **kwargs)
            state = runner.execute_model_state
            if state is None or self._teacher_count:
                return result
            batch = state.input_batch
            if batch.num_reqs != 1 or batch.num_draft_tokens != 4:
                return result
            if batch.num_tokens_after_padding != 5 or state.hidden_states is None:
                raise RuntimeError("Teacher capture requires the actual M5 target")
            with torch.inference_mode():
                index = batch.logits_indices[:1]
                hidden = state.hidden_states[index]
                # All TP workers participate in the ordinary full LM head.
                # This runs after replay and before sampling constraints, and
                # cannot affect the target graph's hidden states or logits.
                logits = runner.model.compute_logits(hidden)
                if logits is None or logits.shape != (1, runner.vocab_size):
                    raise RuntimeError("Teacher logits must cover the full vocabulary")
                self._teacher_count += 1
                if self.rank == 0:
                    ple_samples = {}
                    for name, module in runner.model.named_modules():
                        packet = getattr(module, "_gpu_output_buffer", None)
                        if packet is None:
                            continue
                        decoded = getattr(module, "_packed_result_output", packet)
                        sample = {
                            "packet": packet[:5].detach().cpu().clone(),
                            "decoded": decoded[:5].detach().cpu().clone(),
                            "layout": getattr(module, "_packed_result_layout", None),
                        }
                        book = getattr(module, "_packed_result_codebook", None)
                        if book is not None:
                            sample["codebook"] = book.detach().cpu().clone()
                        ple_samples[name] = sample
                    if ple_samples:
                        torch.save(
                            {
                                "layers": ple_samples,
                                "input_ids": batch.input_ids[:5].detach().cpu(),
                                "positions": batch.positions[:5].detach().cpu(),
                                "ngram_context": runner.model_state.ngram_context[:1]
                                .detach()
                                .cpu(),
                            },
                            root / f"{key}-ple.pt",
                        )
                    kv_samples = {}
                    mappings = state.slot_mappings_by_layer or {}
                    for (
                        name,
                        module,
                    ) in runner.compilation_config.static_forward_context.items():
                        cache = getattr(module, "host_kv", None)
                        slots = mappings.get(name)
                        if cache is None or slots is None:
                            continue
                        slots = slots[:5]
                        slots = slots[slots >= 0]
                        blocks, offsets = (
                            slots // cache.page_size,
                            slots % cache.page_size,
                        )
                        kinds = torch.arange(2, device=slots.device)
                        values = cache.history[
                            blocks[:, None], kinds[None, :], offsets[:, None], 0
                        ].cpu()
                        if cache.fp8:
                            scales = cache.scales[slots].cpu()
                            values = (
                                values.view(torch.float8_e4m3fn).float()
                                * scales[:, :, None]
                            ).half()
                        kv_samples[name] = values
                    if kv_samples:
                        torch.save(kv_samples, root / f"{key}-kv.pt")
                    torch.save(
                        dict(
                            logits=logits.detach().float().cpu(),
                            position=batch.positions[index].cpu(),
                            input_ids=batch.input_ids[index].cpu(),
                            request_ids=list(batch.req_ids),
                        ),
                        root / f"{key}.pt",
                    )
            return result

        runner.execute_model = execute
        return {"rank": self.rank, "active": True}

    def stop_teacher_capture(self):
        original = getattr(self, "_teacher_original_execute", None)
        if original is None:
            raise RuntimeError("Teacher capture is not active")
        self.model_runner.execute_model = original
        del self._teacher_original_execute
        return {"rank": self.rank, "captured": self._teacher_count}
