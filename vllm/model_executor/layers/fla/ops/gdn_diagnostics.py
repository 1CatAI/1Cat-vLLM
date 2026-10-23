# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GDN observation payloads using the shared per-engine diagnostic owner."""

import os

import torch

from vllm.config.sm70_runtime import capture_runtime_trace
from vllm.diagnostics import diagnostic_channel, diagnostic_history, safe_name
from vllm.logger import init_logger
from vllm.model_executor.layers.fla.ops.gdn_stages import mixed_qkv_decode_layout
from vllm.utils.torch_utils import LayerNameType

logger = init_logger("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")


def layer_index(layer_name: LayerNameType) -> int | None:
    parts = str(layer_name).split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers":
            try:
                return int(parts[idx + 1])
            except ValueError:
                return None
    return None


def tensor_metadata(tensor):
    return {
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "data_ptr": int(tensor.data_ptr()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "dtype": str(tensor.dtype),
    }


def capture_tensor(label, layer_name, tensor, source):
    if torch.compiler.is_compiling():
        return
    channel = diagnostic_channel("gdn_graph")
    policy = channel.policy
    if not policy.capture or not policy.directory or not tensor.is_cuda:
        return
    shape = tuple(tensor.shape)
    idx = layer_index(layer_name)
    if not policy.allows("labels", label) or not policy.allows("layers", idx):
        return
    if not policy.allows("shapes", "x".join(str(dim) for dim in shape)):
        return
    key = f"{os.getpid()}:{source}:{layer_name}:{label}:{shape}"
    meta = tensor_metadata(tensor)
    channel.capture(
        key,
        tensor,
        {
            "label": label,
            "layer_name": str(layer_name),
            "layer_idx": idx,
            "source": source,
            "shape": shape,
            "dtype": str(tensor.dtype),
            "pid": os.getpid(),
            **{
                f"input_{name}": value
                for name, value in meta.items()
                if name != "dtype"
            },
        },
        refresh=True,
    )


def dump_core(label, layer_name, tensor, source="core"):
    channel = diagnostic_channel("gdn_core")
    graph = diagnostic_channel("gdn_graph").policy
    policy = channel.policy
    graph_dump = graph.capture and graph.directory
    if not policy.directory and not graph_dump:
        return
    if graph_dump:
        capture_tensor(label, layer_name, tensor, source)
    if not policy.directory or not tensor.is_cuda:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    if not policy.allows("layers", layer_index(layer_name)) or not policy.can_save():
        return
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return
    assert policy.max_dumps is not None
    key = f"{os.getpid()}:{label}"
    count = channel.counts.get(key, 0)
    if policy.max_dumps <= 0 or count >= policy.max_dumps:
        return
    channel.advance(key)
    channel.write(
        f"pid{os.getpid()}_{label}_{count:03d}_{safe_name(layer_name)}.pt",
        {
            "label": label,
            "layer_name": str(layer_name),
            **tensor_metadata(tensor),
            "source": source,
            "tensor": tensor.detach().cpu(),
        },
    )


def projection_requested(layer_name) -> bool:
    projection = diagnostic_channel("gdn_projection").policy
    graph = diagnostic_channel("gdn_graph").policy
    if not projection.directory and (not graph.capture or not graph.directory):
        return False
    idx = layer_index(layer_name)
    if idx is None:
        return False
    if graph.layers and graph.capture:
        return graph.allows("layers", idx)
    return projection.allows("layers", idx)


def dump_projection(
    tensor: torch.Tensor, label: str, layer_name: LayerNameType
) -> torch.Tensor:
    capture_tensor(label, layer_name, tensor, "proj")
    if torch.cuda.is_current_stream_capturing():
        return tensor.clone()
    channel = diagnostic_channel("gdn_projection")
    policy = channel.policy
    if policy.can_save() and not torch.cuda.is_current_stream_capturing():
        assert policy.max_dumps is not None
        key = f"{os.getpid()}:{layer_name}:{label}"
        count = channel.counts.get(key, 0)
        if policy.max_dumps > 0 and count < policy.max_dumps:
            channel.advance(key)
            channel.write(
                f"pid{os.getpid()}_{safe_name(label)}_{count:03d}_{safe_name(layer_name)}.pt",
                {
                    "label": label,
                    "layer_name": str(layer_name),
                    **tensor_metadata(tensor),
                    "tensor": tensor.detach().cpu(),
                },
            )
    return tensor.clone()


def compare_request(layer_name):
    channel = diagnostic_channel("gdn_compare")
    policy = channel.policy
    if not policy.can_save():
        return None
    if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
        return None
    idx = layer_index(layer_name)
    if idx is None or not policy.allows("layers", idx):
        return None
    if "max_dumps" in policy.filter_errors:
        raise ValueError(policy.filter_errors["max_dumps"])
    assert policy.max_dumps is not None
    if policy.max_dumps <= 0 or policy.max_dumps <= channel.reports:
        return None
    step = channel.advance(f"{os.getpid()}:{layer_name}", start=1)
    if not policy.allows("steps", step):
        return None
    channel.reports += 1
    assert policy.directory
    return channel.output_path(
        f"pid{os.getpid()}_step{step:04d}_layer{safe_name(layer_name)}.pt",
    ), step


def capture_state_slice(
    label: str,
    layer_name: LayerNameType,
    state: torch.Tensor,
    state_indices: torch.Tensor | None,
    num_tokens: int,
) -> None:
    if state_indices is None or num_tokens <= 0:
        return
    if not diagnostic_channel("gdn_graph").policy.capture:
        return
    indices = state_indices[:num_tokens].to(device=state.device, dtype=torch.long)
    indices = indices.clamp(0, state.shape[0] - 1)
    capture_tensor(
        label,
        layer_name,
        state.index_select(0, indices),
        "state",
    )
    if diagnostic_channel("gdn_graph").policy.state_indices:
        capture_tensor(
            f"{label}_indices",
            layer_name,
            indices.to(dtype=torch.int32),
            "state",
        )


def log_decode_route(
    *,
    layer_name: LayerNameType,
    stage: str,
    decision: str,
    reason: str,
    mixed_qkv: torch.Tensor,
    state_indices: torch.Tensor | None,
    num_decode_tokens: int,
) -> None:
    if not capture_runtime_trace().value("gdn_route_debug"):
        return
    if torch.compiler.is_compiling():
        return
    mixed_layout = mixed_qkv_decode_layout(mixed_qkv)
    key = f"{os.getpid()}:{stage}:{decision}:{reason}:{mixed_layout}:{layer_name}"
    counts = diagnostic_history("gdn_route")
    count = counts.get(key, 0)
    if count >= 1:
        return
    if len(counts) >= 96:
        return
    counts[key] = count + 1
    state_desc = "None"
    if state_indices is not None:
        state_desc = (
            f"shape={tuple(state_indices.shape)} "
            f"dtype={state_indices.dtype} "
            f"stride={tuple(state_indices.stride())} "
            f"contiguous={state_indices.is_contiguous()}"
        )
    logger.info(
        "SM70 FlashQLA GDN decode route debug: layer=%s stage=%s "
        "decision=%s reason=%s capture=%s tokens=%s "
        "mixed_shape=%s mixed_dtype=%s mixed_stride=%s "
        "mixed_contiguous=%s mixed_layout=%s logical_width=%s "
        "row_stride=%s state_indices=%s",
        layer_name,
        stage,
        decision,
        reason,
        torch.cuda.is_current_stream_capturing(),
        num_decode_tokens,
        tuple(mixed_qkv.shape),
        mixed_qkv.dtype,
        tuple(mixed_qkv.stride()),
        mixed_qkv.is_contiguous(),
        mixed_layout,
        mixed_qkv.shape[1] if mixed_qkv.dim() == 2 else None,
        mixed_qkv.stride(0) if mixed_qkv.dim() == 2 else None,
        state_desc,
    )
