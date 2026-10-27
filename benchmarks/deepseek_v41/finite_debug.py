# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Locate the first nonfinite module boundary in a real FP16 target forward."""

import json

import torch
import torch.distributed as dist


def install(worker):
    from vllm.distributed import get_tp_group

    model = worker.model_runner.model
    group = get_tp_group().device_group
    context = {}

    def inputs(module, args, kwargs):
        del module, args
        ids, positions = kwargs["input_ids"], kwargs["positions"]
        context.update(
            rows=ids.numel(),
            start=int(positions[0]),
            end=int(positions[-1]),
            tokens=ids.tolist() if ids.numel() <= 6 else ids[-8:].tolist(),
        )

    model.register_forward_pre_hook(inputs, with_kwargs=True)

    def check(name):
        def hook(module, args, output):
            del module, args
            tensors = output if isinstance(output, tuple) else (output,)
            for tensor in tensors:
                if (
                    not isinstance(tensor, torch.Tensor)
                    or not tensor.is_floating_point()
                ):
                    continue
                finite = torch.isfinite(tensor).all().to(torch.int32)
                # Stop every rank at the same boundary, before a peer can
                # enter the next collective or shared-memory wait.
                dist.all_reduce(finite, op=dist.ReduceOp.MIN, group=group)
                if not finite.item():
                    detail = dict(
                        module=name,
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        context=context,
                        nan=int(torch.isnan(tensor).sum()),
                        inf=int(torch.isinf(tensor).sum()),
                    )
                    print("V41_FIRST_NONFINITE " + json.dumps(detail), flush=True)
                    raise FloatingPointError(json.dumps(detail))

        return hook

    for name, module in model.named_modules():
        if name:
            module.register_forward_hook(check(name))
