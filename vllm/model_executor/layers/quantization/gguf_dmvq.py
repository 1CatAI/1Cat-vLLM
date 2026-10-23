# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured temporary original-record readers for IQ2 down projections."""

import torch
from torch.nn import Parameter

from vllm.model_executor.kernels.gguf import GGUFOperatorCapability, decoder_family
from vllm.model_executor.layers.quantization.gguf_native_pair import _SOURCE_PACKERS
from vllm.model_executor.layers.quantization.gguf_turbomind import (
    _prepared_gguf_projection,
    prepared_projection_arguments,
)
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

_books = {}
# Cold-L2 graph ABBA on real TP4 shards: IQ2_XS 23.81 vs 32.32 us,
# IQ2_S 25.12 vs 33.91 us. Other source formats retain the prior route.
_CONFIGS = {17: (8, 1), 22: (8, 1)}


def _project(
    x: torch.Tensor,
    records: torch.Tensor,
    book: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    kind: int,
    kw: int,
    split: int,
    codes: list[torch.Tensor],
    stats: list[torch.Tensor],
    caches: list[torch.Tensor | None],
    descriptors: list[int],
    cache_bands: list[int],
    blas_bands: list[int],
) -> torch.Tensor:
    rows = x.reshape(-1, x.shape[-1]).contiguous()
    if rows.shape[0] == 8:
        out = rows.new_empty((8, 5120))
        torch.ops._C.gguf_dmvq_sm70_out(
            out, rows, records, partials, counters, book, kind, kw, split
        )
    else:
        out = _prepared_gguf_projection(
            rows,
            codes[0],
            stats[0],
            caches[0],
            descriptors[0],
            descriptors[1],
            descriptors[2],
            descriptors[3],
            descriptors[4],
            descriptors[5],
            descriptors[6],
            cache_bands,
            blas_bands,
        )
    return out.reshape(*x.shape[:-1], 5120)


def _project_fake(
    x,
    records,
    book,
    partials,
    counters,
    kind,
    kw,
    split,
    codes,
    stats,
    caches,
    descriptors,
    cache_bands,
    blas_bands,
):
    return x.new_empty((*x.shape[:-1], 5120))


direct_register_custom_op(
    "gguf_dmvq_projection",
    _project,
    fake_impl=_project_fake,
    mutates_args=["partials", "counters"],
)


def prepare_layer(layer, sources, projections, enabled):
    kind = sources[0][1] if len(sources) == 1 else -1
    reason = None
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif kind not in _CONFIGS:
        reason = "temporary_reader_not_faster_or_not_measured"
    elif not layer.prefix.endswith(".down_proj") or sources[0][0].shape[0] != 5120:
        reason = "temporary_reader_shape_not_measured"
    elif not (
        len(projections) == 1
        and projections[0].kernel is not None
        and projections[0].kernel.config.partition_weight_shape == (4352, 5120)
    ):
        reason = "canonical_fallback_unavailable"
    elif not all(
        hasattr(torch.ops._C, name)
        for name in ("gguf_dmvq_sm70_out", "gguf_dmvq_book_sm70_out")
    ):
        reason = "packaged_temporary_reader_missing"
    if reason is None:
        weight = sources[0][0]
        device = weight.device
        layer.register_parameter(
            "gguf_dmvq_records",
            Parameter(
                torch.from_numpy(
                    _SOURCE_PACKERS[kind](weight.detach().cpu().numpy())
                ).to(device),
                False,
            ),
        )
        if (device, kind) not in _books:
            book = torch.empty(65536, dtype=torch.uint8, device=device)
            torch.ops._C.gguf_dmvq_book_sm70_out(book, kind)
            _books[device, kind] = book
        layer.register_buffer("gguf_dmvq_book", _books[device, kind], persistent=False)
        layer.register_buffer(
            "gguf_dmvq_partials",
            torch.empty(160 * 256, dtype=torch.float32, device=device),
            persistent=False,
        )
        layer.register_buffer(
            "gguf_dmvq_counters",
            torch.zeros(160, dtype=torch.int32, device=device),
            persistent=False,
        )
        layer.gguf_dmvq_kind = kind
    capability = GGUFOperatorCapability(
        decoder_family(kind) if kind >= 0 else decoder_family(17),
        quant_type_name(kind) if kind >= 0 else "mixed",
        "gguf_dmvq_sm70_out",
        True,
        min_m=8,
        max_m=8,
        reason=reason,
    )
    from dataclasses import asdict

    return asdict(capability)


def apply_layer(layer, x):
    kw, split = _CONFIGS[layer.gguf_dmvq_kind]
    return torch.ops.vllm.gguf_dmvq_projection(
        x,
        layer.gguf_dmvq_records,
        layer.gguf_dmvq_book,
        layer.gguf_dmvq_partials,
        layer.gguf_dmvq_counters,
        layer.gguf_dmvq_kind,
        kw,
        split,
        *prepared_projection_arguments(layer.gguf_tm_projections),
    )
