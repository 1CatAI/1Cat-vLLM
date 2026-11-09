# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional NVFP4 projections alongside independent common DFlash2 schedules.

VLLM_SM70_DFLASH2_QPN2_MANIFEST selects hashed cap64/publisher libraries and
their matching all-reduce library. Layers qualify through their actual QPN2
representation and dimensions. Other quantizations retain their projections
while the common worker extension can still install its independent routes.
This experimental entry point does not change a serving default.
"""

import functools
import json
import os
from pathlib import Path

import torch

from benchmarks.kernels.sm70_dflash2_common_candidate_route import (
    CommonDFlash2Extension,
    _library,
)


def install_qpn2_routes(manifest: dict) -> None:
    from vllm.distributed import get_tp_group
    from vllm.model_executor.layers.linear import RowParallelLinear
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_w4a4_nvfp4 as quant,
    )
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if set(manifest) != {"schema_version", "capped", "publisher", "all_reduce"}:
        raise ValueError("Incomplete or unknown QPN2 manifest entries")
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported QPN2 manifest version")
    libraries = {
        key: _library(manifest[key]) for key in ("capped", "publisher", "all_reduce")
    }
    configured_ar = os.getenv("VLLM_SM70_CUSTOM_AR_LIBRARY")
    if not configured_ar or Path(configured_ar).resolve() != libraries["all_reduce"]:
        raise ValueError("QPN2 publication requires the matching all-reduce library")
    torch.ops.load_library(str(libraries["publisher"]))
    torch.ops.load_library(str(libraries["capped"]))
    columns = {}
    rows = {}
    seen = set()
    original_apply = quant.CompressedTensorsW4A4Fp4._apply_qpn2
    original_forward = RowParallelLinear.forward
    original_load = GPUModelRunner.load_model

    @torch.library.custom_op("quasar_capped::column", mutates_args=())
    def column(x: torch.Tensor, prefix: str, gated: bool) -> torch.Tensor:
        layer = columns[prefix]
        if (
            x.ndim == 2
            and x.shape[0] == 8
            and x.dtype == torch.float16
            and torch.cuda.is_current_stream_capturing()
        ):
            assert x.shape[1] == 5120 and x.is_contiguous()
            divisor = 2 if gated else 1
            logical = layer.output_size_per_partition // divisor
            physical = int(layer.sm70_nvfp4_qpn2_output_size) // divisor
            output = torch.empty((8, physical), device=x.device, dtype=x.dtype)
            split, nacc = quant._SM70_NVFP4_QPN2_CONFIGS[
                (5120, physical * divisor, gated)
            ]
            op = torch.ops._qpn2_capped.gated if gated else torch.ops._qpn2_capped.gemm
            op(
                output,
                x,
                layer.sm70_nvfp4_qpn2_codes,
                layer.sm70_nvfp4_qpn2_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                split,
                nacc,
            )
            if prefix not in seen:
                seen.add(prefix)
                print(f"QPN2_CAP64_CAPTURE prefix={prefix}", flush=True)
            return output[:, :logical]
        return original_apply(layer, x, None, gated_silu=gated)

    @column.register_fake
    def column_fake(x, prefix, gated):
        layer = columns[prefix]
        divisor = 2 if gated else 1
        logical = layer.output_size_per_partition // divisor
        physical = int(layer.sm70_nvfp4_qpn2_output_size) // divisor
        return x.new_empty((x.shape[0], physical))[:, :logical]

    def apply(layer, x, bias, *, gated_silu):
        if layer.prefix in columns:
            assert bias is None
            return column(x, layer.prefix, gated_silu)
        return original_apply(layer, x, bias, gated_silu=gated_silu)

    @torch.library.custom_op("quasar_qpn2::row_parallel", mutates_args=())
    def row_parallel(input_: torch.Tensor, prefix: str) -> torch.Tensor:
        layer, communicator = rows[prefix]
        if (
            input_.ndim == 2
            and input_.shape[0] == 8
            and input_.dtype == torch.float16
            and input_.is_contiguous()
            and torch.cuda.is_current_stream_capturing()
        ):
            peers = communicator.sm70_tp4_push_buffer_ptrs
            assert peers is not None and communicator.world_size == 4
            assert communicator.fully_connected
            projected = torch.empty(
                (8, 5120), device=input_.device, dtype=torch.float16
            )
            output = torch.empty_like(projected)
            torch.ops._qpn2_candidate.publish(
                projected,
                input_,
                layer.sm70_nvfp4_qpn2_codes,
                layer.sm70_nvfp4_qpn2_scales,
                float(layer.sm70_nvfp4_qpn2_global_scale),
                int(layer.sm70_nvfp4_qpn2_split_k),
                int(layer.sm70_nvfp4_qpn2_nacc),
                peers,
                layer.tp_rank,
            )
            torch.ops._qpn2_candidate.consume(projected, output, peers, layer.tp_rank)
            if prefix not in seen:
                seen.add(prefix)
                print(f"QPN2_PUBLISH_CAPTURE prefix={prefix}", flush=True)
            return output
        result = original_forward(layer, input_)
        return result[0] if isinstance(result, tuple) else result

    @row_parallel.register_fake
    def row_fake(input_, prefix):
        return input_.new_empty((*input_.shape[:-1], 5120))

    @functools.wraps(original_forward)
    def forward(self, input_):
        if self.prefix in rows:
            result = row_parallel(input_, self.prefix)
            return (result, None) if self.return_bias else result
        return original_forward(self, input_)

    @functools.wraps(original_load)
    def load(self, *args, **kwargs):
        result = original_load(self, *args, **kwargs)
        communicator = getattr(get_tp_group().device_communicator, "ca_comm", None)
        for _name, layer in self.model.named_modules():
            if not getattr(layer, "sm70_nvfp4_qpn2", False):
                continue
            if (
                getattr(layer, "input_size_per_partition", None) == 5120
                and layer.prefix.rsplit(".", 1)[-1]
                in ("in_proj_qkvz", "qkv_proj", "gate_up_proj")
                and layer.bias is None
                and layer.tp_size == 4
            ):
                physical = int(layer.sm70_nvfp4_qpn2_output_size)
                if physical in (4128, 3584, 8704):
                    columns[layer.prefix] = layer
            if (
                isinstance(layer, RowParallelLinear)
                and layer.input_is_parallel
                and layer.reduce_results
                and layer.tp_size == 4
                and layer.bias is None
                and layer.output_size_per_partition == 5120
                and layer.input_size_per_partition in (1536, 4352)
                and int(layer.sm70_nvfp4_qpn2_nacc) == 2
            ):
                assert communicator is not None and not communicator.disabled
                assert communicator.sm70_tp4_push_buffer_ptrs is not None
                rows[layer.prefix] = (layer, communicator)
        print(
            f"QPN2_ROUTES_READY rank={torch.distributed.get_rank()} "
            f"columns={len(columns)} rows={len(rows)}",
            flush=True,
        )
        return result

    quant.CompressedTensorsW4A4Fp4._apply_qpn2 = staticmethod(apply)
    RowParallelLinear.forward = forward
    GPUModelRunner.load_model = load


class DFlash2KernelExtension(CommonDFlash2Extension):
    """Optional QPN2 projections plus separately configured common schedules."""


if manifest_path := os.getenv("VLLM_SM70_DFLASH2_QPN2_MANIFEST"):
    install_qpn2_routes(json.loads(Path(manifest_path).read_text()))
