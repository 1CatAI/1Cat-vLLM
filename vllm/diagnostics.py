# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Engine-owned diagnostic counters, capture buffers and output management."""

import json
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from vllm.config.diagnostic_dump import (
    TensorDumpConfig,
)
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.runtime_resources import current_runtime_resources, runtime_resources_for


@dataclass
class DiagnosticChannel:
    policy: TensorDumpConfig
    counts: dict[str, int] = field(default_factory=dict)
    saves: dict[str, int] = field(default_factory=dict)
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    retired_buffers: list[torch.Tensor] = field(default_factory=list)
    reports: int = 0
    engine_tag: str = ""

    def advance(self, key: str, *, start: int = 0) -> int:
        count = self.counts.get(key, start)
        self.counts[key] = count + 1
        return count

    def take(self, key: str, limit: int, *, unlimited: bool = False) -> bool:
        used = self.saves.get(key, 0)
        if limit <= 0 and not unlimited or limit > 0 and used >= limit:
            return False
        self.saves[key] = used + 1
        return True

    def capture(self, key: str, tensor: torch.Tensor, metadata: dict, *, refresh=False):
        buffer = self.buffers.get(key)
        if (
            buffer is None
            or buffer.shape != tensor.shape
            or buffer.dtype != tensor.dtype
            or buffer.device != tensor.device
        ):
            if buffer is not None:
                # Graphs captured earlier still own the previous address.
                self.retired_buffers.append(buffer)
            buffer = torch.empty_like(tensor)
            self.buffers[key] = buffer
            self.metadata[key] = metadata
        elif refresh:
            self.metadata[key].update(metadata)
        buffer.copy_(tensor)

    def write(self, filename: str, payload: dict) -> str:
        assert self.policy.directory
        return write_payload(self.policy.directory, filename, payload, self.engine_tag)

    def output_path(self, filename: str) -> str:
        assert self.policy.directory
        os.makedirs(self.policy.directory, exist_ok=True)
        return output_path(self.policy.directory, filename, self.engine_tag)

    def append_json(self, filename: str, payload: dict) -> None:
        with open(self.output_path(filename), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True) + "\n")

    def flush_graph(self, step: int, stage: str, *, gdn=False, trigger=True) -> None:
        policy = self.policy
        if not policy.directory or trigger and not policy.can_save():
            return
        if not policy.allows("steps", step):
            return
        for key, buffer in self.buffers.items():
            meta = self.metadata.get(key, {})
            label = safe_name(meta.get("label", "unknown"))
            source = safe_name(meta.get("source" if gdn else "layer_type", "unknown"))
            layer = meta.get("layer_idx", -1)
            layer_text = (
                f"{layer:02d}" if isinstance(layer, int) else "none" if gdn else "-1"
            )
            shape = "x".join(str(dim) for dim in buffer.shape)
            self.write(
                f"pid{os.getpid()}_step{step:04d}_layer{layer_text}_{source}_{label}_shape{shape}.pt",
                {
                    **meta,
                    "step": step,
                    "stage": stage,
                    "graph_buffer_key": key,
                    "tensor": buffer.detach().cpu(),
                },
            )


def safe_name(value) -> str:
    return str(value).replace("/", "_").replace(".", "_")


class EngineDiagnostics:
    def __init__(self, trace):
        self.engine_tag = f"engine{uuid.uuid4().hex}"
        self.trace = trace
        config = trace.dumps
        self.sampling = trace.sampling
        self.runner_steps = config.runner_steps
        self.counters: dict[str, int] = {}
        self.timing_sums: dict[str, float] = {}
        self.timing_calls = 0
        self.histories: dict[str, dict] = {}
        self.channels = {
            name: DiagnosticChannel(policy, engine_tag=self.engine_tag)
            for name, policy in trace.dump_channels().items()
        }
        # MoE and model layers share selection, not counters or capture buffers.
        self.channels["moe_runner"] = DiagnosticChannel(
            config.qwen_layer, engine_tag=self.engine_tag
        )

    def close(self):
        for channel in self.channels.values():
            channel.counts.clear()
            channel.saves.clear()
            channel.buffers.clear()
            channel.metadata.clear()
            channel.retired_buffers.clear()
        self.counters.clear()
        self.timing_sums.clear()
        self.histories.clear()


# Historical imports can access these independent, unconfigured helper owners.
# Engine dispatch never uses them and never reads legacy inputs through them.
_legacy_channels: dict[str, DiagnosticChannel] = {}


def legacy_channel(name: str) -> DiagnosticChannel:
    if name not in _legacy_channels:
        _legacy_channels[name] = DiagnosticChannel(TensorDumpConfig())
    return _legacy_channels[name]


def diagnostics_for(config=None) -> EngineDiagnostics | None:
    resources = (
        runtime_resources_for(config)
        if config is not None
        else current_runtime_resources()
    )
    if not resources:
        # Synthetic/legacy forward contexts may have no engine configuration.
        return None
    owner = resources.get("diagnostics")
    if owner is None:
        trace = resources.get("runtime_trace")
        if trace is None:
            raise RuntimeError("Engine diagnostics require an initialized trace policy")
        owner = EngineDiagnostics(trace)
        resources["diagnostics"] = owner
    return owner


def diagnostic_channel(name: str, *, owner=None) -> DiagnosticChannel:
    if owner is None and is_forward_context_available():
        owner = get_forward_context().runtime_resources.get("diagnostics")
    owner = diagnostics_for() if owner is None else owner
    if owner is not None:
        return owner.channels[name]
    channel = legacy_channel(name)
    # Old no-config calls deliberately remain a separate compatibility adapter.
    from vllm.config.sm70_runtime import RuntimeTraceConfig

    channel.policy = RuntimeTraceConfig.legacy_dump_channel(
        "qwen_layer" if name == "moe_runner" else name
    )
    return channel


def layer_dump_requested(
    layer_idx: int, label: str | None = None, *, moe=False
) -> bool:
    policy = diagnostic_channel("moe_runner" if moe else "qwen_layer").policy
    return bool(
        policy.directory
        and (not moe or layer_idx >= 0)
        and policy.allows("layers", layer_idx)
        and (label is None or policy.allows("labels", label))
    )


def record_layer_tensor(
    tensor: torch.Tensor, label: str, layer_idx: int, layer_type: str, *, moe=False
) -> torch.Tensor:
    channel = diagnostic_channel("moe_runner" if moe else "qwen_layer")
    policy = channel.policy
    # Preserve the original non-aliasing model op and aliasing MoE op schemas.
    if not moe:
        tensor = tensor.clone()
    if not policy.directory or not policy.allows_tokens(tensor):
        return tensor
    if not moe and not policy.allows("labels", label):
        return tensor
    if policy.capture and tensor.is_cuda:
        shape = tuple(tensor.shape)
        key = f"{os.getpid()}:{layer_idx}:{label}:{shape}"
        if moe:
            key += f":{tensor.dtype}"
        channel.capture(
            key,
            tensor,
            {
                "label": label,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "shape": shape,
                "dtype": str(tensor.dtype),
                "pid": os.getpid(),
            },
        )
        if not moe and torch.cuda.is_current_stream_capturing():
            return tensor
    if moe or not policy.direct_save or not policy.can_save():
        return tensor
    if torch.cuda.is_current_stream_capturing():
        return tensor
    key = f"{os.getpid()}:{layer_idx}:{label}"
    count = channel.advance(key)
    if not policy.allows("counts", count):
        return tensor
    assert policy.max_dumps is not None
    if channel.take(key, policy.max_dumps, unlimited=True):
        channel.write(
            f"pid{os.getpid()}_layer{layer_idx:02d}_{safe_name(layer_type)}_{safe_name(label)}_{count:03d}.pt",
            {
                "label": label,
                "layer_idx": layer_idx,
                "layer_type": layer_type,
                "count": count,
                "pid": os.getpid(),
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "tensor": tensor.detach().cpu(),
            },
        )
    return tensor


def flush_layer_buffers(step: int, stage: str, *, moe=False, owner=None) -> None:
    channel = diagnostic_channel("moe_runner" if moe else "qwen_layer", owner=owner)
    policy = channel.policy
    if not policy.directory or not channel.buffers:
        return
    if not moe and not policy.can_save():
        return
    if not policy.allows("steps", step):
        return
    channel.flush_graph(step, stage, trigger=not moe)
    if not moe:
        # Retain the old model-flush checkpoint's paired MoE flush.
        flush_layer_buffers(step, stage, moe=True, owner=owner)


def bind_diagnostics(config=None) -> EngineDiagnostics:
    owner = diagnostics_for(config)
    if owner is not None:
        return owner
    from vllm.config.sm70_runtime import RuntimeTraceConfig

    return EngineDiagnostics(RuntimeTraceConfig())


def flush_runner_graphs(stage: str, *, owner=None) -> None:
    owner = bind_diagnostics() if owner is None else owner
    qwen, gdn = owner.channels["qwen_layer"], owner.channels["gdn_graph"]
    if not qwen.policy.capture and not gdn.policy.capture:
        return
    step = owner.counters.get("runner_graph_steps", 0) + 1
    owner.counters["runner_graph_steps"] = step
    if owner.runner_steps is not None and step not in owner.runner_steps:
        return
    if qwen.policy.capture:
        flush_layer_buffers(step, stage, owner=owner)
    if gdn.policy.capture:
        gdn.flush_graph(step, stage, gdn=True)


_legacy_histories: dict[str, dict] = {}


def legacy_history(name: str) -> dict:
    return _legacy_histories.setdefault(name, {})


def diagnostic_history(name: str, *, owner=None) -> dict:
    owner = diagnostics_for() if owner is None else owner
    if owner is None:
        return legacy_history(name)
    return owner.histories.setdefault(name, {})


def output_path(directory: str, filename: str, engine_tag: str = "") -> str:
    if engine_tag:
        stem, extension = os.path.splitext(filename)
        filename = f"{stem}_{engine_tag}{extension}"
    return os.path.join(directory, filename)


def write_payload(
    directory: str, filename: str, payload: dict, engine_tag: str = ""
) -> str:
    os.makedirs(directory, exist_ok=True)
    path = output_path(directory, filename, engine_tag)
    torch.save(payload, path)
    return path


def write_json_payload(
    directory: str, filename: str, payload: dict, engine_tag: str = ""
) -> str:
    os.makedirs(directory, exist_ok=True)
    path = output_path(directory, filename, engine_tag)
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path


def diagnostic_engine_tag() -> str:
    owner = diagnostics_for()
    return "" if owner is None else owner.engine_tag
