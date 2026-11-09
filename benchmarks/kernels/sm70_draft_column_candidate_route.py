# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit, experimental q8 draft GEMMs with column-major FP16 weights.

Importing this module does not install a route. The candidate changes GEMM
arithmetic and is not admitted by the operator reference-error screen alone.
Only the draft's twenty captured query projections are eligible; context,
prefill and unsupported shapes retain their original implementation.
"""

import functools

import torch


def install_draft_column_candidate() -> None:
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    original_apply = UnquantizedLinearMethod.apply
    original_load = GPUModelRunner.load_model
    registry = {}
    captured = set()
    expected_shapes = {
        "qkv_proj": (1536, 5120),
        "o_proj": (5120, 1024),
        "gate_up_proj": (8704, 5120),
        "down_proj": (5120, 4352),
    }

    @torch.library.custom_op("quasar_draft_column::linear", mutates_args=())
    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        layer, column_weight = registry[prefix]
        if (
            x.ndim != 2
            or x.shape != (8, layer.weight.shape[1])
            or not x.is_contiguous()
            or x.dtype != torch.float16
            or not x.is_cuda
            or not torch.cuda.is_current_stream_capturing()
        ):
            return original_apply(layer.quant_method, layer, x, None)
        previous = torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        try:
            # This selects the strict kernel only during this captured call.
            # Restore the process setting before any other projection runs.
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
            output = torch.mm(x, column_weight.T)
        finally:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = previous
        if prefix not in captured:
            captured.add(prefix)
            print(
                f"DRAFT_COLUMN_CAPTURE rank={torch.distributed.get_rank()} "
                f"prefix={prefix} input={tuple(x.shape)} "
                f"weight={tuple(column_weight.shape)} "
                f"stride={column_weight.stride()} strict_reduction=1",
                flush=True,
            )
        return output

    @linear.register_fake
    def fake(x: torch.Tensor, prefix: str) -> torch.Tensor:
        return x.new_empty((*x.shape[:-1], registry[prefix][0].weight.shape[0]))

    @functools.wraps(original_apply)
    def apply(self, layer, x, bias=None):
        if layer.prefix in registry:
            assert bias is None
            return linear(x, layer.prefix)
        return original_apply(self, layer, x, bias)

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        assert torch.distributed.get_world_size() == 4
        assert torch.cuda.get_device_capability(self.device) == (7, 0)
        for name, layer in self.speculator.model.named_modules():
            leaf = name.rsplit(".", 1)[-1]
            if ".layers." not in "." + name or leaf not in expected_shapes:
                continue
            assert isinstance(layer.quant_method, UnquantizedLinearMethod)
            assert layer.bias is None and layer.weight.dtype == torch.float16
            assert tuple(layer.weight.shape) == expected_shapes[leaf]
            column_weight = layer.weight.T.contiguous().T
            assert torch.equal(layer.weight, column_weight)
            assert layer.prefix not in registry
            registry[layer.prefix] = (layer, column_weight)
        assert len(registry) == 20
        print(
            f"DRAFT_COLUMN_READY rank={torch.distributed.get_rank()} "
            f"layers={len(registry)} strict_reduction=1",
            flush=True,
        )
        return result

    UnquantizedLinearMethod.apply = apply
    GPUModelRunner.load_model = load
