# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pair dual-store q8 norms with packed QPN2 consumers after graph capture."""

import torch

from benchmarks.kernels.sm70_qpn2_graph_nodes import (
    clone_params,
    nodes,
    only_kernel,
    set_params,
)
from vllm.model_executor.layers import layernorm as norm

OriginalGraph = torch.cuda.CUDAGraph


def capture_one(fn, parameter_count):
    fn()
    torch.accelerator.synchronize()
    graph = OriginalGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        fn()
    graph.instantiate()
    return graph, only_kernel(graph, parameter_count)


class LayoutTemplates:
    def __init__(self, projection_items, dual_module):
        projection_items = list(projection_items)
        configs = {(4128, False): 16, (3584, False): 16, (8704, True): 8}
        for item in projection_items:
            assert item["k"] == 5120
            assert item["nacc"] == 2
            assert item["split_k"] == configs[(item["n"], item["gated"])]
        dual = dual_module
        self.keep = []
        self.norms = {}
        self.projections = {}
        x = torch.empty((8, 5120), device="cuda", dtype=torch.float16)
        weight = torch.empty((5120,), device="cuda", dtype=torch.float16)
        out = torch.empty_like(x)
        packed = torch.empty_like(x)
        # Finite warmups avoid incidental NaNs; values are not used by the gate.
        x.zero_()
        weight.zero_()
        for kind in ("none", "half", "float"):
            residual = (
                None
                if kind == "none"
                else torch.zeros_like(
                    x, dtype=torch.float16 if kind == "half" else torch.float32
                )
            )
            residual_out = (
                None if residual is None else torch.empty_like(x, dtype=torch.float32)
            )
            kwargs = dict(num_stages=1)
            if kind == "float":
                old = norm._sm70_dflash2_gemma_fused_add_rms_kernel
                new = dual._sm70_dflash2_gemma_fused_add_rms_kernel_dual_store
                kwargs.update(
                    hidden_size=5120, BLOCK_SIZE=8192, epsilon=1e-6, num_warps=8
                )
            else:
                old = norm._sm70_dflash2_fixed_gemma_rms_kernel
                new = dual._sm70_dflash2_fixed_gemma_rms_kernel_dual_store
                kwargs.update(
                    HAS_RESIDUAL=residual is not None,
                    epsilon=1e-6,
                    num_warps=16,
                    enable_fp_fusion=True,
                )
            operands = (x, residual, weight, out, residual_out)
            old_graph, old_node = capture_one(
                lambda old=old, operands=operands, kwargs=kwargs: old[(8,)](
                    *operands, **kwargs
                ),
                dict(none=5, half=7, float=8)[kind],
            )
            new_graph, new_node = capture_one(
                lambda new=new, operands=operands, kwargs=kwargs: new[(8,)](
                    *operands, packed, **kwargs
                ),
                dict(none=6, half=8, float=9)[kind],
            )
            self.keep.extend([old_graph, new_graph, operands, packed])
            mapping = []
            for value in new_node.args:
                if (
                    len(value) == 8
                    and int.from_bytes(value, "little") == packed.data_ptr()
                ):
                    mapping.append(("packed", None))
                elif value in old_node.args:
                    indices = [i for i, arg in enumerate(old_node.args) if arg == value]
                    # Only the trailing zero scratch pointers may be duplicated.
                    assert len(indices) == 1 or value == bytes(len(value)), (
                        kind,
                        indices,
                    )
                    mapping.append(("old", indices[0]))
                else:
                    raise AssertionError(("Unmapped parameter", kind, value.hex()))
            assert sum(kind == "packed" for kind, _ in mapping) == 1
            out_index = old_node.args.index(out.data_ptr().to_bytes(8, "little"))
            key = (old_node.name, tuple(map(len, old_node.args)))
            assert key not in self.norms
            self.norms[key] = (old_node, new_node, mapping, out_index)
        for item in projection_items:
            if item["k"] != 5120:
                continue
            gated = item["gated"]
            width = item["n"] // (2 if gated else 1)
            result = torch.empty((8, width), device="cuda", dtype=torch.float16)
            op = (
                torch.ops._qpn2_packed_input.gated
                if gated
                else torch.ops._qpn2_packed_input.gemm
            )
            args = (
                result,
                packed,
                item["codes"],
                item["scales"],
                item["global_scale"],
                item["split_k"],
                item["nacc"],
            )
            graph, node = capture_one(lambda op=op, args=args: op(*args), 8)
            key = (gated, width, item["split_k"], item["nacc"])
            self.projections[key] = node
            self.keep.extend([graph, result, args])

    def describe(self):
        return dict(
            norms=[
                dict(
                    name=k[0],
                    sizes=k[1],
                    new_name=v[1].name,
                    new_sizes=list(map(len, v[1].args)),
                    mapping=v[2],
                    out_index=v[3],
                )
                for k, v in self.norms.items()
            ],
            projections=[
                dict(key=k, name=v.name, sizes=list(map(len, v.args)))
                for k, v in self.projections.items()
            ],
        )

    def patch(self, graph):
        kernel_nodes, parents = nodes(graph)
        edits, norms_seen, storage = [], {}, []
        for handle, projection in kernel_nodes.items():
            if (
                "qpn2" not in projection.name
                or "sm70_kernel" not in projection.name
                or len(projection.args) != 8
                or projection.integer(5) != 5120
                or projection.integer(6) != 8
            ):
                continue
            gated = "gated" in projection.name
            split = projection.params.blockDim.x // (64 if gated else 32)
            key = (gated, projection.integer(4), split, 2)
            assert key in self.projections, (projection.name, key)
            template = self.projections[key]
            input_pointer = projection.args[2]
            frontier, visited, found = list(parents[handle]), set(), None
            while frontier and found is None:
                matches, next_frontier = [], []
                for parent in frontier:
                    if parent in visited:
                        continue
                    visited.add(parent)
                    candidate = kernel_nodes.get(parent)
                    if candidate is not None and candidate.grid == (8, 1, 1):
                        norm_key = (candidate.name, tuple(map(len, candidate.args)))
                        if norm_key in self.norms:
                            out_index = self.norms[norm_key][3]
                            if candidate.args[out_index] == input_pointer:
                                matches.append((candidate, self.norms[norm_key]))
                    next_frontier.extend(parents[parent])
                assert len(matches) <= 1, "Ambiguous norm producer"
                found = matches[0] if matches else None
                frontier = next_frontier
            assert found is not None, ("Missing q8 norm producer", projection.name)
            producer, (_, new_norm, mapping, _) = found
            if producer.handle not in norms_seen:
                # Preserve the original normalized output for every other reader.
                canary = torch.full(
                    (8 * 5120 + 16,), -37, device="cuda", dtype=torch.float16
                )
                packed = canary[8:-8].view(8, 5120)
                arguments = [
                    packed.data_ptr().to_bytes(8, "little")
                    if kind == "packed"
                    else producer.args[index]
                    for kind, index in mapping
                ]
                changed, owned = clone_params(
                    producer.params, arguments, template=new_norm.params
                )
                edits.append((producer, changed))
                storage.extend([canary, packed, owned])
                norms_seen[producer.handle] = (packed, canary)
            packed, _ = norms_seen[producer.handle]
            arguments = list(projection.args)
            arguments[2] = packed.data_ptr().to_bytes(8, "little")
            changed, owned = clone_params(
                projection.params, arguments, template=template.params
            )
            edits.append((projection, changed))
            storage.append(owned)
        return LayoutEdit(graph, edits, storage, norms_seen)


class LayoutEdit:
    def __init__(self, graph, edits, storage, norms):
        self.graph, self.edits, self.storage, self.norms = graph, edits, storage, norms
        self.mode = "control"

    def switch(self, mode):
        assert mode in ("control", "candidate")
        # Call only between requests / isolated replays; no concurrent launches.
        torch.accelerator.synchronize()
        for original, changed in self.edits:
            set_params(
                self.graph,
                original.handle,
                changed if mode == "candidate" else original.params,
            )
        self.mode = mode
        return len(self.edits)

    def check_canaries(self):
        for _, canary in self.norms.values():
            assert bool((canary[:8] == -37).all() and (canary[-8:] == -37).all())
