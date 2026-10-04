# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-capture MTP annotations; never installed in speed admission runs."""

import json
from pathlib import Path

import torch


def install(worker, folder):
    runner = worker.model_runner
    draft = runner.speculator
    if hasattr(worker, "_mtp_node_trace_restore"):
        raise RuntimeError("MTP node annotations already installed")
    if draft is None or draft.num_speculative_steps != 4:
        raise ValueError("The fixed control requires four draft steps")
    restore = []
    ordinal = {"step": 0}

    def wrap(obj, name, label):
        original = getattr(obj, name)

        def observed(*args, **kwargs):
            annotation = label(*args, **kwargs) if callable(label) else label
            with torch.cuda.nvtx.range(annotation):
                return original(*args, **kwargs)

        restore.append((obj, name, original))
        setattr(obj, name, observed)

    def proposal_label(*args, **kwargs):
        ordinal["step"] = 0
        return "mtp15.draft_all"

    def decode_label(desc):
        ordinal["step"] += 1
        if ordinal["step"] > 3:
            raise RuntimeError("More than three draft decode graph replays")
        return f"mtp15.draft_step/{ordinal['step']}/M{desc.num_tokens}"

    inventory = []
    for role, model in (("target", runner.model), ("draft", draft.model)):
        for kind, tensors in (
            ("parameter", model.named_parameters(remove_duplicate=False)),
            ("buffer", model.named_buffers(remove_duplicate=False)),
        ):
            for name, tensor in tensors:
                inventory.append(
                    {
                        "role": role,
                        "kind": kind,
                        "name": name,
                        "shape": list(tensor.shape),
                        "stride": list(tensor.stride()),
                        "dtype": str(tensor.dtype),
                        "logical_bytes": tensor.numel() * tensor.element_size(),
                        "data_ptr": tensor.data_ptr(),
                        "storage_ptr": tensor.untyped_storage().data_ptr(),
                        "storage_bytes": tensor.untyped_storage().nbytes(),
                    }
                )
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"rank{worker.rank}_weights.json").write_text(
        json.dumps(
            {
                "rank": worker.rank,
                "kind": "resident_tensor_inventory_not_measured_dram_traffic",
                "tensors": inventory,
            },
            indent=2,
        )
        + "\n"
    )
    wrap(runner, "execute_model", "mtp15.target_execute")
    wrap(runner, "prepare_inputs", "mtp15.prepare")
    wrap(runner, "sample_tokens", "mtp15.sample_handoff_draft")
    wrap(runner, "sample", "mtp15.target_head_sample")
    wrap(
        runner.cudagraph_manager,
        "run_fullgraph",
        lambda desc: f"mtp15.target_forward/M{desc.num_tokens}",
    )
    wrap(draft, "propose", proposal_label)
    wrap(
        draft.prefill_cudagraph_manager,
        "run_fullgraph",
        lambda desc: f"mtp15.draft_step/0/M{desc.num_tokens}",
    )
    wrap(draft.decode_cudagraph_manager, "run_fullgraph", decode_label)
    worker._mtp_node_trace_restore = restore
    return {"rank": worker.rank, "inventory_tensors": len(inventory)}


def uninstall(worker):
    for obj, name, original in reversed(worker._mtp_node_trace_restore):
        setattr(obj, name, original)
    del worker._mtp_node_trace_restore
