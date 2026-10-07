# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit benchmark-only CPU stage observer for TP graph launch diagnosis."""

import os
import threading
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any


class CPUStageRecorder:
    def __init__(self, rank, limit=131072):
        self.rank = rank
        self.limit = limit
        self.enabled = False
        self.nvtx = False
        self.events = []
        self.step = 0
        self.dropped = 0
        self._target = threading.local()
        self.gpu_timing = False
        self.gpu_anchor = None
        self.gpu_events = []

    @contextmanager
    def stage(self, label, metadata=None):
        if not self.enabled:
            yield
            return
        event = {
            "label": label,
            "step": self.step,
            "thread": threading.get_native_id(),
            "start_ns": time.perf_counter_ns(),
            **(metadata or {}),
        }
        gpu_pair = None
        if self.gpu_timing and label in (
            "worker.execute",
            "worker.sample",
            "target.replay",
            "draft.propose",
            "runner.sample",
        ):
            import torch

            gpu_pair = (
                torch.Event(device="cuda", enable_timing=True),
                torch.Event(device="cuda", enable_timing=True),
            )
            gpu_pair[0].record()
        if self.nvtx:
            import torch

            torch.cuda.nvtx.range_push("graph_parity." + label)
        try:
            yield
        finally:
            if gpu_pair is not None:
                gpu_pair[1].record()
                if len(self.gpu_events) < self.limit:
                    self.gpu_events.append((event, gpu_pair))
            if self.nvtx:
                torch.cuda.nvtx.range_pop()
            event["end_ns"] = time.perf_counter_ns()
            if len(self.events) < self.limit:
                self.events.append(event)
            else:
                self.dropped += 1

    def wrap(self, owner, name, label, metadata=None, advances_step=False):
        if owner is None or not hasattr(owner, name):
            return
        original = getattr(owner, name)

        @wraps(original)
        def observed(*args, **kwargs):
            if not self.enabled:
                return original(*args, **kwargs)
            if advances_step:
                self.step += 1
            fields = metadata(*args, **kwargs) if metadata else None
            with self.stage(label, fields):
                return original(*args, **kwargs)

        setattr(owner, name, observed)

    def wrap_target_replay(self, manager, graph_class):
        original_manager = manager.run_fullgraph
        original_replay = graph_class.replay

        @wraps(original_manager)
        def managed(desc):
            if not self.enabled:
                return original_manager(desc)
            fields = {
                "tokens": desc.num_tokens,
                "requests": desc.num_reqs,
                "uniform": desc.uniform_token_count,
            }
            self._target.fields = fields
            try:
                with self.stage("target.manager", fields):
                    return original_manager(desc)
            finally:
                self._target.fields = None

        @wraps(original_replay)
        def replay(graph):
            fields = getattr(self._target, "fields", None)
            if not self.enabled or fields is None:
                return original_replay(graph)
            # Timestamp immediately before the actual replay API, after any
            # manager-side offloader wait. Drafter graphs remain excluded.
            with self.stage("target.replay", fields):
                return original_replay(graph)

        manager.run_fullgraph = managed
        graph_class.replay = replay

    def read(self, stop=True):
        if stop:
            self.enabled = False
        gpu_records = []
        if self.gpu_events:
            import torch

            torch.accelerator.synchronize()
            assert self.gpu_anchor is not None
            for event, (begin, end) in self.gpu_events:
                gpu_records.append(
                    {
                        "label": event["label"],
                        "step": event["step"],
                        "start_ms": self.gpu_anchor.elapsed_time(begin),
                        "end_ms": self.gpu_anchor.elapsed_time(end),
                        "elapsed_ms": begin.elapsed_time(end),
                    }
                )
        return {
            "rank": self.rank,
            "pid": os.getpid(),
            "clock": "perf_counter_ns",
            "events": list(self.events),
            "dropped": self.dropped,
            "gpu_events": gpu_records,
            "gpu_event_scope": "current-stream envelopes; nested spans overlap",
        }


class GraphParityWorkerExtension:
    """Attach through worker_extension_cls; ordinary workers remain unchanged."""

    rank: int
    model_runner: Any

    def read_prefill_memory(self, reset_peak: bool = False):
        """Capture worker allocation admission between benchmark requests."""
        import torch

        torch.accelerator.synchronize()
        free, total = torch.accelerator.get_memory_info()
        result = {
            "rank": self.rank,
            "allocated_bytes": torch.accelerator.memory_allocated(),
            "reserved_bytes": torch.accelerator.memory_reserved(),
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "device_free_bytes": free,
            "device_total_bytes": total,
        }
        if reset_peak:
            torch.accelerator.reset_peak_memory_stats()
        return result

    def read_prefill_storages(self):
        """Inventory live device storage, separating mapped host aliases."""
        import torch
        from cuda.bindings import runtime

        from vllm.model_executor.layers.quantization import gguf_dense_hmma

        seen = set()
        storages = {}
        errors = []

        def walk(value, owner, depth=0):
            if isinstance(value, torch.nn.parameter.UninitializedParameter):
                return
            if isinstance(value, torch.Tensor):
                if value.is_cuda and value.numel():
                    storage = value.untyped_storage()
                    key = (value.device.index, storage.data_ptr())
                    if key not in storages:
                        error, attributes = runtime.cudaPointerGetAttributes(key[1])
                        if error != runtime.cudaError_t.cudaSuccess:
                            errors.append({"owner": owner, "error": int(error)})
                        else:
                            storages[key] = {
                                "bytes": storage.nbytes(),
                                "memory_type": int(attributes.type),
                                "owners": [],
                                "shape": list(value.shape),
                                "dtype": str(value.dtype),
                            }
                    if key in storages and owner not in storages[key]["owners"]:
                        storages[key]["owners"].append(owner)
                return
            if id(value) in seen or depth > 24:
                return
            seen.add(id(value))
            items: Any
            if isinstance(value, dict):
                items = value.items()
            elif isinstance(value, (list, tuple)):
                items = enumerate(value)
            elif isinstance(value, torch.nn.Module) or type(
                value
            ).__module__.startswith("vllm."):
                items = vars(value).items() if hasattr(value, "__dict__") else ()
            else:
                return
            for name, child in items:
                walk(child, owner + "." + str(name), depth + 1)

        walk(self.model_runner, "runner")
        walk(gguf_dense_hmma._workspaces, "gguf_dense_workspaces")
        records = sorted(storages.values(), key=lambda value: -value["bytes"])
        return {
            "rank": self.rank,
            "scope": "reachable Python tensors; excludes native-only allocations",
            "device_storage_bytes": sum(
                value["bytes"]
                for value in records
                if value["memory_type"]
                == int(runtime.cudaMemoryType.cudaMemoryTypeDevice)
            ),
            "mapped_host_storage_bytes": sum(
                value["bytes"]
                for value in records
                if value["memory_type"]
                == int(runtime.cudaMemoryType.cudaMemoryTypeHost)
            ),
            "storages": records,
            "pointer_errors": errors,
        }

    def set_gguf_prefill_routing_policy(self, enabled: bool):
        """Compare routing between completed requests without reloading weights."""
        if type(enabled) is not bool:
            raise TypeError("GGUF prefill routing requires a boolean policy")
        runner = self.model_runner
        configs = [runner.vllm_config]
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            configs.append(speculator.vllm_config)
        for config in configs:
            config.kernel_config.sm70_gguf.prefill_routing = enabled
        return {"rank": self.rank, "prefill_routing": enabled}

    def set_qsa_host_prefill_policy(self, enabled):
        """Change only prefill dispatch between completed benchmark cohorts."""
        if not isinstance(enabled, bool):
            raise TypeError("Host prefill policy requires a boolean")
        runner = self.model_runner
        configs = [runner.vllm_config]
        speculator = getattr(runner, "speculator", None)
        if speculator is not None:
            configs.append(speculator.vllm_config)
        owners = set()
        for config in configs:
            for module in config.compilation_config.static_forward_context.values():
                if getattr(module, "host_kv_enabled", False) and hasattr(
                    module, "host_kv_prefill_enabled"
                ):
                    module.host_kv_prefill_enabled = enabled
                    owners.add(id(module))
            config.kernel_config.qsa_host_kv_prefill = enabled
        if not owners:
            raise RuntimeError("No admitted host QSA owners")
        return {"rank": self.rank, "enabled": enabled, "owners": len(owners)}

    def set_mtp_execution_policy(self, draft_single_graph, greedy_verify):
        """Benchmark RPC: change host dispatch between completed cohorts.

        Draft graphs must already have been captured during engine startup.
        Changing this policy neither recaptures graphs nor modifies weights.
        """
        if not isinstance(draft_single_graph, bool) or not isinstance(
            greedy_verify, bool
        ):
            raise TypeError("MTP execution policy requires boolean values")
        runner = self.model_runner
        speculator = runner.speculator
        if speculator is None or getattr(speculator, "method", None) != "mtp":
            raise RuntimeError("Execution policy requires an MTP speculator")
        manager = getattr(speculator, "multistep_cudagraph_manager", None)
        if draft_single_graph and (manager is None or not manager.graphs):
            raise RuntimeError("Single-graph draft shapes were not captured")
        if greedy_verify and not hasattr(runner.model, "get_top_tokens"):
            raise RuntimeError("Target model does not expose local argmax")
        for config in (runner.vllm_config, speculator.vllm_config):
            config.kernel_config.sm70_draft_single_graph = draft_single_graph
            config.kernel_config.sm70_greedy_verify = greedy_verify
        return {
            "rank": self.rank,
            "draft_single_graph": draft_single_graph,
            "greedy_verify": greedy_verify,
        }

    def set_graph_input_preparation(self, early):
        state = self.model_runner.model_state
        declared = type(state).supports_early_input_preparation
        if early and not declared:
            raise RuntimeError("Model state has not declared independent inputs")
        state.supports_early_input_preparation = early
        return {"rank": self.rank, "early": early, "declared": declared}

    def set_ple_input_preparation(self, fused):
        state = self.model_runner.model_state
        if not hasattr(state, "_ple_kernel_config"):
            raise RuntimeError("Model state does not expose PLE input preparation")
        state._ple_kernel_config.ple_input_prepare = bool(fused)
        return {"rank": self.rank, "ple_input_prepare": bool(fused)}

    def start_graph_parity_observer(self, nvtx=False, gpu_timing=False):
        if not hasattr(self, "_graph_parity_recorder"):
            from vllm.v1.executor.multiproc_executor import WorkerProc
            from vllm.v1.worker.gpu.async_utils import AsyncOutput

            rec = CPUStageRecorder(self.rank)
            self._graph_parity_recorder = rec
            runner = self.model_runner
            rec.wrap(self, "execute_model", "worker.execute", advances_step=True)
            rec.wrap(self, "sample_tokens", "worker.sample")
            for name in (
                "prepare_inputs",
                "prepare_attn",
                "finish_requests",
                "free_states",
                "add_requests",
                "update_requests",
                "sample",
                "postprocess_sampled",
            ):
                rec.wrap(runner, name, "runner." + name)
            for name in ("prepare_inputs", "prepare_attn", "preprocess_state"):
                rec.wrap(runner.model_state, name, "model_state." + name)
            seen_builders = set()
            for group_id, groups in enumerate(runner.attn_groups):
                for group in groups:
                    builder = group.get_metadata_builder(0)
                    if id(builder) not in seen_builders:
                        seen_builders.add(id(builder))
                        rec.wrap(
                            builder,
                            "build",
                            f"metadata.{type(builder).__name__}.{group_id}",
                        )
            from vllm.v1.worker.gpu.model_states import mamba_hybrid

            rec.wrap(
                mamba_hybrid,
                "prepare_dflash2_gdn_group_metadata",
                "metadata.gdn_shared",
            )
            rec.wrap(runner.block_tables, "apply_staged_writes", "block_tables.apply")
            rec.wrap(runner.speculator, "propose", "draft.propose")
            rec.wrap(runner._ple_offload_connector, "prepare_forward", "ple.prepare")
            import torch

            rec.wrap_target_replay(runner.cudagraph_manager, torch.cuda.CUDAGraph)
            rec.wrap(AsyncOutput, "get_output", "output.materialize")
            rec.wrap(WorkerProc, "enqueue_output", "output.serialize")
        rec = self._graph_parity_recorder
        rec.events.clear()
        rec.step = rec.dropped = 0
        rec.gpu_events.clear()
        rec.gpu_timing = gpu_timing
        rec.gpu_anchor = None
        if gpu_timing:
            import torch

            rec.gpu_anchor = torch.Event(device="cuda", enable_timing=True)
            rec.gpu_anchor.record()
        rec.nvtx = nvtx
        rec.enabled = True
        return {
            "rank": self.rank,
            "clock": "perf_counter_ns",
            "nvtx": nvtx,
            "gpu_timing": gpu_timing,
        }

    def read_graph_parity_observer(self, stop=True):
        return self._graph_parity_recorder.read(stop)

    def start_attention_transfer_diagnosis(self):
        """Record one attention preparation's implicit host transfers."""
        import traceback

        import torch
        from torch.utils._python_dispatch import TorchDispatchMode

        records = []

        class TransferDiagnosis(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                suspicious = False
                if func is torch.ops.aten.index.Tensor and args[0].is_cuda:
                    suspicious = any(
                        index is not None and index.device.type == "cpu"
                        for index in args[1]
                    )
                elif func is torch.ops.aten._to_copy.default:
                    destination = kwargs.get("device")
                    suspicious = (
                        args[0].device.type == "cpu"
                        and destination is not None
                        and destination.type == "cuda"
                        and not kwargs.get("non_blocking", False)
                    )
                if suspicious:
                    records.append(
                        {
                            "operator": str(func),
                            "shape": list(args[0].shape),
                            "stack": traceback.format_stack(limit=16),
                        }
                    )
                return func(*args, **kwargs)

        state = self.model_runner.model_state
        original = state.prepare_attn

        @wraps(original)
        def diagnosed(*args, **kwargs):
            batch = args[0] if args else kwargs["input_batch"]
            if batch.num_tokens != 5 or batch.num_reqs != 1:
                return original(*args, **kwargs)
            state.prepare_attn = original
            with TransferDiagnosis():
                return original(*args, **kwargs)

        self._attention_transfer_diagnosis = records
        state.prepare_attn = diagnosed
        return {"rank": self.rank}

    def read_attention_transfer_diagnosis(self):
        return {"rank": self.rank, "records": self._attention_transfer_diagnosis}

    def start_graph_parity_capture(self):
        import torch

        torch.cuda.cudart().cudaProfilerStart()

    def stop_graph_parity_capture(self):
        import torch

        torch.accelerator.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
