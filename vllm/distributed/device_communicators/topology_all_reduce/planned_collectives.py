# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Explicit setup-time compiled DAG channels, including owned external relays."""

import hashlib
import os
from pathlib import Path
from typing import Any

from .collective_planner import validate
from .planned_codegen import generate, layout


def load_native(plans, topology, capacity):
    import torch
    from torch.utils.cpp_extension import get_default_build_root, load

    source = generate(plans, topology, capacity)
    cpp = Path(__file__).with_name("planned_reduce.cpp")
    arch = os.environ.get("TORCH_CUDA_ARCH_LIST") or ";".join(
        sorted(
            {
                f"{a}.{b}"
                for a, b in (
                    torch.cuda.get_device_capability(i)
                    for i in range(torch.accelerator.device_count())
                )
            }
        )
    )
    digest = hashlib.sha256(
        source.encode() + cpp.read_bytes() + arch.encode()
    ).hexdigest()
    folder = Path(os.environ.get("TORCH_EXTENSIONS_DIR", get_default_build_root())) / (
        "planned-" + digest[:20]
    )
    folder.mkdir(parents=True, exist_ok=True)
    cu = folder / "planned.cu"
    if not cu.exists() or cu.read_text() != source:
        cu.write_text(source)
    return load(
        name="vllm_topology_planned_" + digest[:20],
        sources=[str(cpp), str(cu)],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        build_directory=str(folder),
        verbose=False,
    )


class PlannedChannel:
    """One exact input width and immutable configuration per graph-safe channel.

    Build/qualify collectively before graph capture. CPU metadata uses the
    supplied Gloo group; GPU payload uses only verified direct peer mappings.
    Fixed launch geometry keeps two-epoch mailbox ownership consistent.
    """

    def __init__(
        self,
        group,
        device,
        plan,
        topology,
        *,
        threads=64,
        blocks=5,
        visible=None,
        native=None,
        candidate=0,
    ):
        import torch
        import torch.distributed as dist

        validate(plan, topology)
        if threads not in (64, 128, 256) or not 1 <= blocks <= 32:
            raise ValueError("Invalid launch geometry")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Plan setup must precede capture")
        self.group, self.device, self.plan, self.topology = (
            group,
            device,
            plan,
            topology,
        )
        self.threads, self.blocks = threads, blocks
        self.capacity = plan["payload_bytes"]
        self.allocated = {}
        self.imported = []
        self.engine: Any = None
        if visible is None:
            from .planner_probe import visible_devices

            visible = visible_devices()
        visible = list(visible)
        members = [None] * dist.get_world_size(group)
        dist.all_gather_object(members, visible[device.index], group=group)
        if members != plan["members"]:
            raise ValueError("Plan member order differs from communicator")
        self.rank = dist.get_rank(group)
        nodes = plan["devices"]
        devices = [visible.index(d) for d in nodes]
        if not set(nodes) <= set(visible):
            raise ValueError("Relay is not visible")
        if len(members) == 1 and len(nodes) == 1:
            self.native: Any = None
            return
        # Native runtime limit is an explicit setup bound, not a topology rule.
        if len(nodes) > 64:
            raise ValueError("CUDA backend supports at most 64 nodes per compiled plan")
        if native is None:
            if self.rank == 0:
                load_native([plan], topology, self.capacity)
            dist.barrier(group=group)
            native = load_native([plan], topology, self.capacity)
        self.native = native
        owners: list[Any] = [
            members.index(node) if node in members else None for node in nodes
        ]
        loads = [0] * len(members)
        for r, owner in enumerate(owners):
            if owner is None:
                owner = min(
                    range(len(members)),
                    key=lambda m: (
                        (members[m], nodes[r]) not in topology.labels,
                        loads[m],
                        m,
                    ),
                )
                owners[r] = owner
                loads[owner] += 1
        owned = [r for r, owner in enumerate(owners) if owner == self.rank]
        adjacent = [{r} for r in range(len(nodes))]
        for message in plan["messages"]:
            a, b = message["source"], message["dest"]
            adjacent[a].add(b)
            adjacent[b].add(a)
        for node in owned:
            for peer in adjacent[node] - {node}:
                if (
                    nodes[node],
                    nodes[peer],
                ) not in topology.labels or not torch.cuda.can_device_access_peer(
                    devices[node], devices[peer]
                ):
                    raise RuntimeError("Planned peer mapping unavailable")
        _, sizes = layout(plan, self.capacity)
        handles = {}
        flags = []
        for node in owned:
            pointer, handle = native.allocate(devices[node], sizes[node])
            self.allocated[node] = pointer
            handles[node] = handle
            flags.append(
                torch.zeros(32, dtype=torch.int32, device=f"cuda:{devices[node]}")
            )
        exchanged: list[Any] = [None] * len(members)
        dist.all_gather_object(exchanged, handles, group=group)
        rows = []
        for node in owned:
            row = [0] * len(nodes)
            for peer in sorted(adjacent[node]):
                if owners[peer] == self.rank:
                    row[peer] = self.allocated[peer]
                    if peer != node:
                        native.enable_peer(devices[node], devices[peer])
                else:
                    pointer = native.open_handle(
                        devices[node], exchanged[owners[peer]][peer]
                    )
                    self.imported.append((devices[node], pointer))
                    row[peer] = pointer
            rows.append(row)
        self.flags = flags
        self.nodes_owned = owned
        self.sizes = sizes
        self.devices = devices
        self.engine = native.Engine(
            device.index,
            owned,
            devices,
            rows,
            flags,
            candidate,
            threads,
            blocks,
            self.capacity,
        )
        for node in owned:
            torch.accelerator.synchronize(devices[node])
        dist.barrier(group=group)

    def accepts(self, x):
        import torch

        if self.native is None:
            return x.device == self.device
        return (
            x.device == self.device
            and x.dtype == torch.float16
            and x.is_contiguous()
            and x.nbytes == self.capacity
            and x.data_ptr() % 16 == 0
        )

    def all_reduce(self, x):
        import torch

        if not self.accepts(x):
            raise ValueError("Unqualified payload shape/dtype/layout")
        if self.native is None:
            return x
        out = torch.empty_like(x)
        self.engine.run(x, out)
        return out

    def qualify(self, repeats=3):
        """Changed dense inputs, all-rank bitwise AST, poisoned chained graphs."""
        import torch
        import torch.distributed as dist

        if self.native is None:
            return dict(
                status="passed", checks=0, precision=self.plan["precision"], local=True
            )
        checks = 0
        examples = []

        def expected(values):
            cache = []
            by_node = dict(zip(self.plan["members"], values))
            for node in self.plan["devices"]:
                cache.append(by_node.get(node, torch.zeros_like(values[0])))
            for value in self.plan["values"][len(cache) :]:
                a, b = value["children"]
                v = cache[a].float() + cache[b].float()
                cache.append(v if value["dtype"] == "fp32" else v.half())
            return cache[self.plan["outputs"][0]].half()

        for trial in range(repeats):
            generator = torch.Generator().manual_seed(723001 + trial * 31 + self.rank)
            original = (
                torch.randn(self.capacity // 2, generator=generator) * 0.125
            ).half()
            values = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(values, original, group=self.group)
            ref = expected(values)
            x = original.to(self.device)
            torch.accelerator.synchronize(self.device)
            dist.barrier(group=self.group)
            out = self.all_reduce(x)
            torch.accelerator.synchronize(self.device)
            passed = torch.equal(
                out.cpu().view(torch.int16), ref.view(torch.int16)
            ) and torch.equal(x.cpu(), original)
            statuses = [None] * len(values)
            dist.all_gather_object(statuses, passed, group=self.group)
            if not all(statuses):
                raise RuntimeError("Planned DAG dense AST qualification failed")
            checks += 2
            examples.append(
                hashlib.sha256(
                    out.cpu().view(torch.uint8).numpy().tobytes()
                ).hexdigest()
            )
        # Replay changes the input after capture. Each intermediate dependent
        # output is checked, with exact fractions chosen to avoid overflow.
        source = torch.empty(
            self.capacity // 2, dtype=torch.float16, device=self.device
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = source
            outputs = []
            for _ in range(4):
                output = self.all_reduce(output)
                outputs.append(output)
        for trial in range(repeats):
            original = torch.full(
                (self.capacity // 2,),
                (self.rank + 1 + trial) / 4096,
                dtype=torch.float16,
            )
            values = [None] * dist.get_world_size(self.group)
            dist.all_gather_object(values, original, group=self.group)
            source.copy_(original)
            for output in outputs:
                output.fill_(float("nan"))
            torch.accelerator.synchronize(self.device)
            dist.barrier(group=self.group)
            graph.replay()
            torch.accelerator.synchronize(self.device)
            passed = True
            for output in outputs:
                ref = expected(values)
                passed &= torch.equal(
                    output.cpu().view(torch.int16), ref.view(torch.int16)
                )
                values = [ref] * len(values)
                checks += 1
            statuses = [None] * len(values)
            dist.all_gather_object(statuses, passed, group=self.group)
            if not all(statuses):
                raise RuntimeError(
                    "Planned DAG changed/poisoned dependent graph failed"
                )
        return dict(
            status="passed",
            checks=checks,
            examples=examples,
            precision=self.plan["precision"],
            fingerprint=self.plan["fingerprint"],
        )

    def close(self):
        if not self.allocated:
            return
        import torch
        import torch.distributed as dist

        for node in self.nodes_owned:
            torch.accelerator.synchronize(self.devices[node])
        dist.barrier(group=self.group)
        assert self.native is not None
        self.engine.close()
        for device, pointer in self.imported:
            self.native.release(device, pointer, True)
        self.imported.clear()
        dist.barrier(group=self.group)
        for node, pointer in self.allocated.items():
            self.native.release(self.devices[node], pointer, False)
        self.allocated.clear()
