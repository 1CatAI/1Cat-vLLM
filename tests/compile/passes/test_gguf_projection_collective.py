# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep live-M fallback and scratch aliases across the projection/norm rewrite."""

import operator

import pytest
import torch
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.distributed import parallel_state  # noqa: F401
from vllm.model_executor.layers.quantization import gguf_dmv  # noqa: F401
from vllm.model_executor.layers.quantization.gguf_projection_collective import (
    fuse_projection_collectives,
)


def trace(rows, functional, *, kind=5, k=4352, kw=4, extra_consumer=False):
    with FakeTensorMode():
        inputs = (
            torch.empty(rows, k, dtype=torch.float16),
            torch.empty(1, dtype=torch.uint8),
            torch.empty(1, dtype=torch.float32),
            torch.empty(1, dtype=torch.int32),
            torch.empty(rows, 5120, dtype=torch.float32),
            torch.empty(5120, dtype=torch.float32),
        )

        def model(x, codes, partials, counters, residual, weight):
            args: tuple = (
                x,
                [codes],
                [codes],
                [codes],
                [kind],
                [5120],
                kw,
                2,
                1,
                False,
                False,
                partials,
                counters,
                None,
                [],
                [codes],
                [codes],
                [None],
                [],
                [],
                [],
            )
            op = torch.ops.vllm.gguf_dmv_projection.default
            scratch = ()
            if functional is None:
                projected = op(*args)
            else:
                kwargs = dict(zip([a.name for a in op._schema.arguments], args))
                if functional == auto_functionalized_v2:
                    kwargs.pop("partials")
                    kwargs.pop("counters")
                    kwargs.update(
                        _all_bases=[partials, counters],
                        _partials_base_index=0,
                        _counters_base_index=1,
                    )
                result = functional(op, **kwargs)
                projected, scratch = result[0], result[1:]
            norm, new_residual = torch.ops.vllm.sm70_tp4_all_reduce_gemma_rms_norm(
                projected,
                residual,
                weight,
                9.999999974752427e-7,
                group_name="tp:0",
            )
            extra = (projected.relu(),) if extra_consumer else ()
            return norm, new_residual, *scratch, *extra

        return make_fx(model)(*inputs)


@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("kind", [0, 3, 5])
@pytest.mark.parametrize(
    "functional", [None, auto_functionalized, auto_functionalized_v2]
)
def test_rewrite_preserves_live_m_and_mutable_outputs(rows, kind, functional):
    module = trace(rows, functional, kind=kind)
    original_outputs = tuple(module.graph.nodes)[-1].args[0]
    shapes = [tuple(n.meta["val"].shape) for n in original_outputs]
    assert fuse_projection_collectives(module.graph) == 1
    module.graph.lint()
    module.recompile()
    outputs = tuple(module.graph.nodes)[-1].args[0]
    assert [tuple(n.meta["val"].shape) for n in outputs] == shapes
    if functional is not None:
        call = next(n for n in module.graph.nodes if n.target == functional)
        assert call.args[0] == torch.ops.vllm.gguf_projection_collective_norm.default
        assert [n.args[1] for n in outputs if n.target == operator.getitem] == [
            0,
            1,
            2,
            3,
        ]
        if functional == auto_functionalized_v2:
            assert call.kwargs["_partials_base_index"] == 0
            assert call.kwargs["_counters_base_index"] == 1
    # The exact GGUF metadata epsilon is retained rather than replaced by 1e-6.
    call = next(
        n
        for n in module.graph.nodes
        if n.target
        in (functional, torch.ops.vllm.gguf_projection_collective_norm.default)
    )
    assert (call.args[-2] if functional is None else call.kwargs["epsilon"]) == (
        9.999999974752427e-7
    )


@pytest.mark.parametrize(
    "functional", [None, auto_functionalized, auto_functionalized_v2]
)
@pytest.mark.parametrize(
    "change", [dict(kind=6), dict(k=5120), dict(kw=8), dict(extra_consumer=True)]
)
def test_reject_unmeasured_shapes_and_auxiliary_projection_consumers(
    functional, change
):
    module = trace(8, functional, **change)
    assert fuse_projection_collectives(module.graph) == 0
    module.graph.lint()
