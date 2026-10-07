# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Canonical expert banks and graph-compatible GGUF routing."""

from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.forward_context import get_forward_kernel_config_or_none
from vllm.logger import init_logger
from vllm.model_executor.kernels.gguf import (
    GGUFDecoderFamily,
    GGUFOperatorCapability,
    admit_moe_fallback,
    decoder_family,
    dp4a_expert_capabilities,
    lattice_grouped_capabilities,
    q8_intermediate_expert_capabilities,
    raw_grouped_gate_up_capabilities,
    select_lattice_grouped_capability,
    small_grouped_vector_capabilities,
)
from vllm.model_executor.layers.fused_moe.sm70_small_routing import (
    SM70_PREFILL_ROUTING,
    SM70_SMALL_ROUTING,
)
from vllm.model_executor.layers.quantization.gguf_lattice_transcode import (
    LATTICE_TYPES,
    LatticeGGUFProjection,
    transcode_lattice,
)
from vllm.model_executor.layers.quantization.gguf_lut_transcode import (
    LUT4_TYPES,
    Lut4GGUFProjection,
    transcode_lut4,
)
from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod
from vllm.model_executor.layers.quantization.gguf_native import pad_weight_tail
from vllm.model_executor.layers.quantization.gguf_raw import RawGGUFProjection
from vllm.model_executor.layers.quantization.gguf_transcode import (
    AffineGGUFProjection,
    transcode_affine,
)
from vllm.platforms import current_platform
from vllm.transformers_utils.gguf_tensor_reader import quant_type_name
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _expert_gate_up(
    x: torch.Tensor,
    offsets: torch.Tensor,
    ids: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_up: torch.Tensor,
    gate_ptrs: torch.Tensor,
    gate_stats: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_stats: torch.Tensor,
    source_type: int,
    experts: int,
    group: int,
    output_size: int,
    top_k: int,
    raw_batches: list[int],
    vector_bands: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Resolve original M inside the opaque op. Range compilation must not
    # freeze the prefill choice for subsequent MTP verification batches.
    gate = x.new_empty((x.shape[0], output_size))
    up = torch.empty_like(gate)
    if x.shape[0] // top_k in raw_batches:
        logger.info_once(
            "SM70 original-block joint GGUF gate/up enabled (type=%d, M=%d).",
            source_type,
            x.shape[0] // top_k,
        )
        torch.ops._C.gguf_lattice_raw_grouped_gate_up_sm70_out(
            gate, up, x, raw_gate, raw_up, offsets, ids, source_type, top_k
        )
    else:
        vector = any(
            x.shape[0] >= vector_bands[i]
            and (vector_bands[i + 1] < 0 or x.shape[0] <= vector_bands[i + 1])
            for i in range(0, len(vector_bands), 2)
        )
        op = (
            torch.ops._C.gguf_lut4_grouped_gemm_sm70_out
            if source_type in (20, 23)
            else torch.ops._C.gguf_lattice_grouped_vec_sm70_out
            if vector
            else torch.ops._C.gguf_lattice_grouped_gemm_sm70_out
        )
        for out, weights, stats in (
            (gate, gate_ptrs, gate_stats),
            (up, up_ptrs, up_stats),
        ):
            op(
                out,
                x,
                offsets,
                weights,
                stats,
                0 if source_type in (20, 23) else source_type,
                experts,
                group,
            )
    return gate, up


def _expert_gate_up_fake(
    x,
    offsets,
    ids,
    raw_gate,
    raw_up,
    gate_ptrs,
    gate_stats,
    up_ptrs,
    up_stats,
    source_type,
    experts,
    group,
    output_size,
    top_k,
    raw_batches,
    vector_bands,
):
    return x.new_empty((x.shape[0], output_size)), x.new_empty(
        (x.shape[0], output_size)
    )


direct_register_custom_op(
    op_name="gguf_expert_gate_up",
    op_func=_expert_gate_up,
    fake_impl=_expert_gate_up_fake,
)


def _expert_down(
    out: torch.Tensor,
    x: torch.Tensor,
    offsets: torch.Tensor,
    weight_ptrs: torch.Tensor,
    stat_ptrs: torch.Tensor,
    source_type: int,
    decoder: int,
    experts: int,
    group: int,
    vector_batches: list[int],
) -> None:
    # Inspect actual routed rows inside the opaque boundary, including graph
    # capture. Range compilation must not freeze the prefill fallback.
    if x.shape[0] in vector_batches:
        logger.info_once(
            "SM70 canonical GGUF down vectors enabled (type=%d, routed_rows=%d).",
            source_type,
            x.shape[0],
        )
        torch.ops._C.gguf_small_grouped_vec_sm70_out(
            out, x, offsets, weight_ptrs, stat_ptrs, source_type, experts, group
        )
    else:
        op = (
            torch.ops._C.gguf_lut4_grouped_gemm_sm70_out
            if source_type == 20
            else torch.ops._C.gguf_affine_grouped_gemm_sm70_out
        )
        op(out, x, offsets, weight_ptrs, stat_ptrs, decoder, experts, group)


def _expert_down_fake(
    out,
    x,
    offsets,
    weight_ptrs,
    stat_ptrs,
    source_type,
    decoder,
    experts,
    group,
    vector_batches,
):
    return None


direct_register_custom_op(
    op_name="gguf_expert_down",
    op_func=_expert_down,
    mutates_args=["out"],
    fake_impl=_expert_down_fake,
)


def _expert_dp4a(
    x: torch.Tensor,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
    raw_gate: torch.Tensor,
    raw_up: torch.Tensor,
    gate_ptrs: torch.Tensor,
    gate_stats: torch.Tensor,
    up_ptrs: torch.Tensor,
    up_stats: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_stats: torch.Tensor,
    source_type: int,
    down_type: int,
    down_decoder: int,
    experts: int,
    group: int,
    intermediate: int,
    raw_batches: list[int],
    vector_bands: list[int],
    down_vector_batches: list[int],
    dp4a_batches: list[int],
    q8_intermediate_batches: list[int],
) -> torch.Tensor:
    # Resolve actual M inside the opaque boundary so range compilation cannot
    # freeze a prefill choice into an MTP verification graph.
    m, top_k = ids.shape
    policy = get_forward_kernel_config_or_none()
    chunk_size = (
        policy.sm70_gguf.prefill_expert_chunk_size if policy is not None else 4096
    )
    if (
        chunk_size > 0
        and m > chunk_size
        and x.is_cuda
        and x.dtype == torch.float16
        and x.shape[1] == 2560
        and experts == 512
        and current_platform.is_device_capability(70)
    ):
        logger.info_once(
            "SM70 GGUF expert prefill row chunks selected (M=%d, chunk_rows=%d).",
            m,
            chunk_size,
        )
        # Expert FFNs are independent across tokens. Keep the scheduler's
        # large prefill batch while bounding gathered and down intermediates.
        output = torch.empty_like(x)
        for start in range(0, m, chunk_size):
            stop = min(start + chunk_size, m)
            part = _expert_dp4a(
                x[start:stop],
                ids[start:stop],
                probabilities[start:stop],
                raw_gate,
                raw_up,
                gate_ptrs,
                gate_stats,
                up_ptrs,
                up_stats,
                down_ptrs,
                down_stats,
                source_type,
                down_type,
                down_decoder,
                experts,
                group,
                intermediate,
                raw_batches,
                vector_bands,
                down_vector_batches,
                [],  # Prefill tails must retain FP16 activations.
                [],
            )
            output[start:stop].copy_(part)
            del part
        return output
    if (
        m in dp4a_batches
        and probabilities.dtype == torch.float32
        and x.is_contiguous()
        and ids.is_contiguous()
        and probabilities.is_contiguous()
    ):
        logger.info_once(
            "SM70 Q8_1 GGUF experts enabled (type=%d, down=%d, M=%d).",
            source_type,
            down_type,
            m,
        )
        q8 = torch.empty((m, x.shape[1] // 32, 36), dtype=torch.uint8, device=x.device)
        quantized_hidden = m in q8_intermediate_batches
        hidden = (
            torch.empty(
                (m, top_k, intermediate // 32, 36), dtype=torch.uint8, device=x.device
            )
            if quantized_hidden
            else x.new_empty((m, top_k, intermediate))
        )
        output = torch.empty_like(x)
        torch.ops._C.gguf_quantize_q8_1_sm70_out(q8, x)
        if quantized_hidden:
            logger.info_once(
                "SM70 GGUF routed Q8 intermediate enabled (type=%d, M=%d).",
                source_type,
                m,
            )
        if source_type in (20, 23):
            torch.ops._C.gguf_dp4a_lut4_gate_up_sm70_out(
                hidden,
                q8,
                ids,
                gate_ptrs,
                gate_stats,
                up_ptrs,
                up_stats,
                experts,
                4 if quantized_hidden else 16,
            )
        else:
            torch.ops._C.gguf_dp4a_gate_up_sm70_out(
                hidden, q8, ids, raw_gate, raw_up, source_type, True
            )
        torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
            output,
            hidden,
            ids,
            probabilities,
            down_ptrs,
            down_stats,
            down_type,
            experts,
        )
        return output
    routed, offsets, sorted_ids, inverse = torch.ops.vllm.sm70_small_expert_route(
        x, ids, experts
    )
    gate, up = _expert_gate_up(
        routed,
        offsets,
        sorted_ids,
        raw_gate,
        raw_up,
        gate_ptrs,
        gate_stats,
        up_ptrs,
        up_stats,
        source_type,
        experts,
        group,
        intermediate,
        top_k,
        raw_batches,
        vector_bands,
    )
    hidden = (torch.nn.functional.silu(gate) * up).contiguous()
    # These buffers are no longer consumed; release the large gathered matrix
    # before allocating down so 16K prefill can reuse its storage.
    del routed, gate, up
    down = x.new_empty((m * top_k, x.shape[1]))
    _expert_down(
        down,
        hidden,
        offsets,
        down_ptrs,
        down_stats,
        down_type,
        down_decoder,
        experts,
        32,
        down_vector_batches,
    )
    del hidden
    return torch.ops.vllm.sm70_small_expert_unroute(down, inverse, probabilities)


def _expert_dp4a_fake(
    x,
    ids,
    probabilities,
    raw_gate,
    raw_up,
    gate_ptrs,
    gate_stats,
    up_ptrs,
    up_stats,
    down_ptrs,
    down_stats,
    source_type,
    down_type,
    down_decoder,
    experts,
    group,
    intermediate,
    raw_batches,
    vector_bands,
    down_vector_batches,
    dp4a_batches,
    q8_intermediate_batches,
):
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="gguf_expert_dp4a",
    op_func=_expert_dp4a,
    fake_impl=_expert_dp4a_fake,
)


def _expert_mma(
    x: torch.Tensor,
    ids: torch.Tensor,
    probabilities: torch.Tensor,
    gate_codes: torch.Tensor,
    gate_scale: torch.Tensor,
    up_codes: torch.Tensor,
    up_scale: torch.Tensor,
    table: torch.Tensor,
    down_ptrs: torch.Tensor,
    down_stats: torch.Tensor,
    plane_format: int,
    down_type: int,
    experts: int,
    intermediate: int,
    kw: int,
    q8_batches: list[int],
) -> torch.Tensor:
    m, top_k = ids.shape
    hidden = (
        torch.empty(
            (m, top_k, intermediate // 32, 36), dtype=torch.uint8, device=x.device
        )
        if m in q8_batches
        else x.new_empty((m, top_k, intermediate))
    )
    ids32 = ids if ids.dtype == torch.int32 else ids.to(torch.int32)
    torch.ops._C.gguf_moe_gate_up_sm70_out(
        hidden,
        x,
        ids32.contiguous(),
        gate_codes,
        gate_scale,
        up_codes,
        up_scale,
        plane_format,
        table,
        kw,
    )
    output = torch.empty_like(x)
    torch.ops._C.gguf_dp4a_down_unroute_sm70_out(
        output,
        hidden,
        ids32,
        probabilities,
        down_ptrs,
        down_stats,
        down_type,
        experts,
    )
    return output


def _expert_mma_fake(
    x,
    ids,
    probabilities,
    gate_codes,
    gate_scale,
    up_codes,
    up_scale,
    table,
    down_ptrs,
    down_stats,
    plane_format,
    down_type,
    experts,
    intermediate,
    kw,
    q8_batches,
):
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="gguf_expert_mma",
    op_func=_expert_mma,
    fake_impl=_expert_mma_fake,
)

MMA_MAX_TOKENS = 20
MMA_TOP_K = 10
MMA_PLANE_KW = 4


def _expert_planes(raw: torch.Tensor, source_type: int, n: int, k: int):
    """Repack [E, N, stride] original IQ3 rows into dense_mv tile planes."""
    from vllm.model_executor.layers.quantization.gguf_moe_planes import (
        BLOCK_BYTES,
        expert_planes,
    )

    experts = raw.shape[0]
    payload = k // 256 * BLOCK_BYTES[source_type]
    rows = raw[:, :n, :payload].reshape(experts * n, payload).cpu().numpy()
    fmt, codes, meta = expert_planes(rows, source_type)
    return (
        fmt,
        torch.from_numpy(codes).to(raw.device).view(experts, -1),
        torch.from_numpy(meta).to(raw.device).view(experts, -1),
    )


class GGUFExpertBank(torch.nn.Module):
    def __init__(self, source_type, experts, device, dtype, retain_raw=False):
        super().__init__()
        self.source_type = source_type
        self.experts = experts
        self.device = device
        self.dtype = dtype
        self.family = decoder_family(source_type)
        self.pending: dict[int, Any] = {}
        self.capabilities: tuple[GGUFOperatorCapability, ...] = ()
        self.retain_raw = retain_raw
        self.raw_pending: dict[int, torch.Tensor] = {}
        self.raw_capabilities: tuple[GGUFOperatorCapability, ...] = ()

    def add(self, index, weight, rank, size, axis):
        if weight.ndim != 2 or not 0 <= index < self.experts:
            raise ValueError("GGUF expert requires a valid index and 2D projection")
        if index in self.pending:
            raise ValueError("Duplicate canonical GGUF expert")
        if self.family == GGUFDecoderFamily.FLOAT:
            if weight.shape[axis] % size or not 0 <= rank < size:
                raise ValueError("Floating GGUF expert has invalid TP boundary")
            local = weight.chunk(size, dim=axis)[rank]
            self.n, self.k = local.shape
            self.pending[index] = local.to(self.device).contiguous()
            return
        source = weight.detach().cpu().numpy()
        canonical: LatticeGGUFProjection | Lut4GGUFProjection | AffineGGUFProjection
        if self.source_type in LATTICE_TYPES:
            canonical = transcode_lattice(source, self.source_type)
        elif self.source_type in LUT4_TYPES:
            canonical = transcode_lut4(source, self.source_type)
        else:
            canonical = transcode_affine(source, self.source_type)
        canonical = canonical.tp_slice(rank, size, axis=axis)
        self.group = canonical.group_size
        self.n, self.k = (
            canonical.shape
            if isinstance(canonical, LatticeGGUFProjection)
            else canonical.codes.shape
        )
        if self.n % 32:
            raise ValueError("Canonical GGUF expert output cuts a 32-row pack")
        self.raw_capabilities = raw_grouped_gate_up_capabilities(
            self.source_type,
            self.k,
            self.n,
            self.experts,
            self.dtype,
            is_sm70=current_platform.is_device_capability(70),
            original_storage_available=self.retain_raw,
        )
        if self.retain_raw and any(c.reason is None for c in self.raw_capabilities):
            raw = RawGGUFProjection.from_rows(source, self.source_type).tp_slice(
                rank, size, axis=axis
            )
            self.raw_pending[index] = torch.from_numpy(raw.data).to(self.device)
        if isinstance(canonical, LatticeGGUFProjection):
            codes, stats = canonical.mma884_storage()
            stats = stats.view({2: np.int16, 4: np.int32, 8: np.int64}[stats.itemsize])
            prepared = torch.ops._C.gguf_lattice_sm70_prepare(
                torch.from_numpy(codes).to(self.device),
                torch.from_numpy(stats).to(self.device),
                self.source_type,
                self.group,
            )
        else:
            codes = torch.from_numpy(canonical.codes).to(self.device)
            scales = torch.from_numpy(canonical.scales).to(self.device)
            if isinstance(canonical, Lut4GGUFProjection):
                self.decoder = canonical.lut_id
                prepared = torch.ops._C.gguf_lut4_sm70_prepare(
                    codes, scales, self.decoder, self.group
                )
            else:
                assert isinstance(canonical, AffineGGUFProjection)
                self.decoder = canonical.bits
                prepared = torch.ops._C.gguf_affine_sm70_prepare(
                    codes,
                    scales,
                    torch.from_numpy(canonical.mins).to(self.device),
                    self.decoder,
                    self.group,
                )
        self.pending[index] = prepared

    def finalize(self):
        if set(self.pending) != set(range(self.experts)):
            raise ValueError("Incomplete canonical GGUF expert bank")
        prepared = [self.pending[i] for i in range(self.experts)]
        if self.family == GGUFDecoderFamily.FLOAT:
            weights = pad_weight_tail(torch.stack(prepared), self.source_type)
            self.register_buffer("weights", weights, persistent=False)
            self.capabilities = (
                admit_moe_fallback(weights, self.source_type, self.dtype),
            )
        else:
            metadata = prepared[0][2].tolist()
            if any(p[2].tolist() != metadata for p in prepared[1:]):
                raise ValueError("Canonical GGUF expert layouts disagree")
            weights = torch.stack([p[0] for p in prepared])
            stats = torch.stack([p[1] for p in prepared])
            self.register_buffer("weights", weights, persistent=False)
            self.register_buffer("stats", stats, persistent=False)
            wp, sp = torch.ops._C.awq_moe_build_strided_ptrs(
                weights, stats, *metadata, self.experts
            )
            self.register_buffer("weight_ptrs", wp, persistent=False)
            self.register_buffer("stat_ptrs", sp, persistent=False)
            if self.family == GGUFDecoderFamily.LATTICE:
                self.capabilities = lattice_grouped_capabilities(
                    self.source_type, self.k, self.n, self.experts, self.dtype
                )
            else:
                name = (
                    "gguf_lut4_grouped_gemm_sm70_out"
                    if self.family == GGUFDecoderFamily.LUT4
                    else "gguf_affine_grouped_gemm_sm70_out"
                )
                self.capabilities = (
                    GGUFOperatorCapability(
                        self.family,
                        quant_type_name(self.source_type),
                        name,
                        True,
                        reason=None
                        if hasattr(torch.ops._C, name)
                        else f"operator_missing:{name}",
                    ),
                )
        self.down_vector_batches: list[int] = []
        if self.source_type in (20, 42):
            vector = small_grouped_vector_capabilities(
                self.source_type,
                self.k,
                self.n,
                self.experts,
                self.dtype,
                is_sm70=current_platform.is_device_capability(70),
            )
            self.capabilities = (*self.capabilities, *vector)
            self.down_vector_batches = [c.min_m for c in vector if c.reason is None]
        self.pending.clear()
        if self.raw_pending:
            if set(self.raw_pending) != set(range(self.experts)):
                raise ValueError("Incomplete original-block GGUF expert bank")
            self.register_buffer(
                "raw_weights",
                torch.stack([self.raw_pending[i] for i in range(self.experts)]),
                persistent=False,
            )
            self.raw_pending.clear()
        if not any(c.reason is None for c in self.capabilities):
            raise ValueError("No admitted canonical GGUF expert operator")

    def forward(self, x, offsets, ids):
        output = torch.empty((x.shape[0], self.n), dtype=x.dtype, device=x.device)
        if self.family == GGUFDecoderFamily.FLOAT:
            op = getattr(torch.ops._C_gguf, self.capabilities[0].operator)
            return op(
                x,
                self.weights,
                ids[:, None].int().contiguous(),
                self.source_type,
                self.n,
                1,
                x.shape[0],
            )
        if self.source_type in (20, 42) and self.down_vector_batches:
            torch.ops.vllm.gguf_expert_down(
                output,
                x,
                offsets,
                self.weight_ptrs,
                self.stat_ptrs,
                self.source_type,
                self.decoder,
                self.experts,
                self.group,
                self.down_vector_batches,
            )
            return output
        capability = (
            select_lattice_grouped_capability(self.capabilities, x.shape[0])
            if self.family == GGUFDecoderFamily.LATTICE
            else self.capabilities[0]
        )
        if capability.reason or not capability.supports_m(x.shape[0]):
            raise ValueError("Canonical GGUF expert operator rejects routed rows")
        decoder = (
            self.source_type
            if self.family == GGUFDecoderFamily.LATTICE
            else self.decoder
        )
        getattr(torch.ops._C, capability.operator)(
            output,
            x,
            offsets,
            self.weight_ptrs,
            self.stat_ptrs,
            decoder,
            self.experts,
            self.group,
        )
        return output


class GGUFTurboMindMoEMethod(GGUFNativeMoEMethod):
    def __init__(self, quant_config, moe):
        super().__init__(quant_config, moe)
        self.builders: dict[str, GGUFExpertBank] = {}
        config = get_current_vllm_config_or_none()
        self.prefill_routing_enabled = self.native_enabled and (
            config.kernel_config.sm70_gguf.prefill_routing
            if config is not None
            else True
        )
        self.prefill_unroute_enabled = self.native_enabled and (
            config.kernel_config.sm70_gguf.prefill_unroute
            if config is not None
            else True
        )
        self.dp4a_enabled = self.native_enabled and (
            config.kernel_config.sm70_gguf.small_m_dp4a if config is not None else True
        )
        self.lut4_dp4a_enabled = self.dp4a_enabled and (
            config.kernel_config.sm70_gguf.lut4_expert_dp4a
            if config is not None
            else True
        )
        self.mma_enabled = bool(
            self.native_enabled
            and config is not None
            and config.kernel_config.sm70_gguf.grouped_mma_gate_up
        )
        self.mma_release_raw = bool(
            config is not None
            and config.kernel_config.sm70_gguf.grouped_mma_release_raw
        )
        self.mma_planes = False
        self.q8_intermediate_enabled = self.dp4a_enabled and (
            config.kernel_config.sm70_gguf.q8_expert_intermediate
            if config is not None
            else True
        )

    def load_expert(self, layer, param, weight, shard_id, expert_id):
        if param.is_gguf_weight_type:
            value = int(weight.item())
            if self.weight_types.setdefault(shard_id, value) != value:
                raise ValueError("GGUF projection has inconsistent expert formats")
            return
        if shard_id not in self.weight_types:
            raise ValueError("Missing GGUF expert type before payload")
        if shard_id not in self.builders:
            self.builders[shard_id] = GGUFExpertBank(
                self.weight_types[shard_id],
                self.num_experts,
                param.device,
                self.params_dtype,
                retain_raw=self.native_enabled
                and shard_id in ("w1", "w3")
                and self.weight_types[shard_id] in (18, 21, 22),
            )
        self.builders[shard_id].add(
            expert_id,
            weight,
            layer.tp_rank,
            layer.tp_size,
            1 if shard_id == "w2" else 0,
        )
        bank = self.builders[shard_id]
        expected = (
            (self.hidden_size, self.intermediate_size)
            if shard_id == "w2"
            else (self.intermediate_size, self.hidden_size)
        )
        actual = (bank.n, bank.k)
        if actual != expected:
            raise ValueError(f"GGUF expert shape {actual} != {expected}")
        self.loaded_experts[shard_id].add(expert_id)

    def process_weights_after_loading(self, layer):
        banks = torch.nn.ModuleDict()
        for shard in ("w1", "w3", "w2"):
            bank = self.builders[shard]
            bank.finalize()
            banks[shard] = bank
        layer.gguf_expert_banks = banks
        self.small_routing = bool(
            self.native_enabled
            and self.params_dtype == torch.float16
            and self.num_experts == 512
            and self.hidden_size == 2560
            and layer.ep_size == 1
        )
        gate, up = banks["w1"], banks["w3"]
        self.raw_gate_up = bool(
            layer.ep_size == 1
            and gate.source_type == up.source_type
            and hasattr(gate, "raw_weights")
            and hasattr(up, "raw_weights")
        )
        down = banks["w2"]
        if self.mma_enabled and not (
            self.raw_gate_up
            and gate.source_type in (18, 21, 22)
            and down.source_type in (20, 42)
        ):
            logger.info_once(
                "SM70 grouped-MMA GGUF expert gate/up not admitted "
                "(raw=%s, gate=%d, down=%d).",
                self.raw_gate_up,
                gate.source_type,
                down.source_type,
            )
        self.mma_planes = bool(
            self.mma_enabled
            and self.raw_gate_up
            and gate.source_type in (18, 21, 22)
            and down.source_type in (20, 42)
            and gate.k % 128 == 0
            and gate.n % 32 == 0
            and self.params_dtype == torch.float16
        )
        if self.mma_planes:
            from vllm.model_executor.layers.quantization.gguf_moe_planes import (
                expert_table,
            )

            fmt, gate.plane_codes, gate.plane_scale = _expert_planes(
                gate.raw_weights, gate.source_type, gate.n, gate.k
            )
            _, up.plane_codes, up.plane_scale = _expert_planes(
                up.raw_weights, up.source_type, up.n, up.k
            )
            self.plane_format = int(fmt)
            logger.info_once(
                "SM70 grouped-MMA GGUF expert gate/up planes enabled "
                "(type=%d, down=%d, raw released=%s).",
                gate.source_type,
                down.source_type,
                self.mma_release_raw,
            )
            gate.plane_table = torch.from_numpy(expert_table(gate.source_type)).to(
                gate.plane_codes.device
            )
            if self.mma_release_raw:
                self.raw_gate_up = False
        if not self.raw_gate_up:
            for bank in (gate, up):
                if hasattr(bank, "raw_weights"):
                    del bank.raw_weights
        self.raw_batches = (
            [c.min_m for c in gate.raw_capabilities if c.reason is None]
            if self.raw_gate_up
            else []
        )
        self.vector_bands = [
            bound
            for c in gate.capabilities[1:]
            if c.reason is None
            for bound in (c.min_m, c.max_m or -1)
        ]
        down = banks["w2"]
        canonical_iq4 = bool(
            layer.ep_size == 1
            and all(
                b.source_type in (20, 23) and b.decoder == 0 and b.group == 32
                for b in (gate, up)
            )
            and down.source_type == 20
            and down.decoder == 0
            and down.group == 32
        )
        dp4a_enabled = (
            self.lut4_dp4a_enabled
            if gate.source_type in (20, 23)
            else self.dp4a_enabled
        )
        self.dp4a_capabilities = dp4a_expert_capabilities(
            gate.source_type,
            down.source_type,
            gate.k,
            gate.n,
            self.num_experts,
            self.params_dtype,
            is_sm70=current_platform.is_device_capability(70),
            enabled=dp4a_enabled,
            original_storage_available=self.raw_gate_up or self.mma_planes,
            canonical_storage_available=canonical_iq4,
        )
        self.dp4a_batches = [
            c.min_m for c in self.dp4a_capabilities if c.reason is None
        ]
        q8_capabilities = q8_intermediate_expert_capabilities(
            gate.source_type,
            down.source_type,
            gate.k,
            gate.n,
            self.num_experts,
            self.params_dtype,
            is_sm70=current_platform.is_device_capability(70),
            enabled=self.q8_intermediate_enabled and dp4a_enabled,
            original_storage_available=self.raw_gate_up or self.mma_planes,
            canonical_storage_available=canonical_iq4,
        )
        self.q8_intermediate_batches = [
            c.min_m for c in q8_capabilities if c.reason is None
        ]
        self.native_admission = {
            "enabled": True,
            "tp_size": layer.tp_size,
            "ep_size": layer.ep_size,
            "small_m_dp4a": {
                "enabled": bool(self.dp4a_batches),
                "activation_format": "Q8_1",
                "accumulation": "FP32",
                "operators": [asdict(c) for c in self.dp4a_capabilities],
                "outside_m_band": "canonical_grouped_operator",
            },
            "q8_intermediate": {
                "enabled": bool(self.q8_intermediate_batches),
                "format": "Q8_1",
                "lanes_per_row": 4 if canonical_iq4 else 16,
                "operators": [asdict(c) for c in q8_capabilities],
                "outside_m_band": "fp16_intermediate",
            },
            "joint_gate_up": {
                "enabled": self.raw_gate_up,
                "original_batches": self.raw_batches,
                "outside_m_band": "canonical_grouped_operator",
                "reason": None
                if self.raw_gate_up
                else "original_expert_bank_not_retained"
                if gate.source_type == up.source_type and layer.ep_size == 1
                else "requires_matching_original_block_formats_without_ep",
            },
            "routing": {
                **asdict(SM70_SMALL_ROUTING),
                "enabled": self.small_routing,
                "reason": None
                if self.small_routing
                else "requires_fp16_512_experts_hidden2560_without_ep",
                "outside_m_band": "legacy_alignment_and_fp32_weighted_sum",
            },
            "prefill_routing": {
                **asdict(SM70_PREFILL_ROUTING),
                "enabled": self.small_routing and self.prefill_routing_enabled,
                "outside_m_band": "legacy_alignment",
                "reason": "unsupported_expert_geometry"
                if not self.small_routing
                else "disabled_by_policy"
                if not self.prefill_routing_enabled
                else None,
            },
            "prefill_unroute": {
                **asdict(SM70_PREFILL_ROUTING),
                "operator": "sm70_small_expert_unroute",
                "enabled": self.small_routing and self.prefill_unroute_enabled,
                "outside_m_band": "legacy_fp32_weighted_sum",
                "reason": "unsupported_expert_geometry"
                if not self.small_routing
                else "disabled_by_policy"
                if not self.prefill_unroute_enabled
                else None,
            },
            "projections": {
                name: {
                    "source_type": quant_type_name(bank.source_type),
                    "shape": [bank.experts, bank.n, bank.k],
                    "operators": [asdict(c) for c in bank.capabilities],
                    "raw_gate_up_operators": [asdict(c) for c in bank.raw_capabilities],
                    "raw_storage_bytes": bank.raw_weights.numel()
                    if hasattr(bank, "raw_weights")
                    else 0,
                }
                for name, bank in banks.items()
            },
        }
        self.builders.clear()

    def apply(
        self, layer, x, topk_weights, topk_ids, shared_experts, shared_experts_input
    ):
        from vllm.model_executor.layers.fused_moe import MoEActivation

        if layer.apply_router_weight_on_input or layer.activation != MoEActivation.SILU:
            raise ValueError(
                "Canonical GGUF experts require output-weighted SiLU routing"
            )
        if x.shape[0] == 0:
            return torch.empty_like(x)
        if (
            self.mma_planes
            and layer.expert_map is None
            and x.shape[0] <= MMA_MAX_TOKENS
            and topk_ids.shape[1] == MMA_TOP_K
            and topk_weights.dtype == torch.float32
            and x.is_contiguous()
            and topk_weights.is_contiguous()
        ):
            bank = layer.gguf_expert_banks
            gate, up, down = bank["w1"], bank["w3"], bank["w2"]
            return torch.ops.vllm.gguf_expert_mma(
                x,
                topk_ids,
                topk_weights,
                gate.plane_codes,
                gate.plane_scale,
                up.plane_codes,
                up.plane_scale,
                gate.plane_table,
                down.weight_ptrs,
                down.stat_ptrs,
                self.plane_format,
                down.source_type,
                self.num_experts,
                self.intermediate_size,
                MMA_PLANE_KW,
                self.q8_intermediate_batches,
            )
        if self.dp4a_batches and layer.expert_map is None:
            bank = layer.gguf_expert_banks
            gate, up, down = bank["w1"], bank["w3"], bank["w2"]
            return torch.ops.vllm.gguf_expert_dp4a(
                x,
                topk_ids,
                topk_weights,
                getattr(gate, "raw_weights", gate.weight_ptrs),
                getattr(up, "raw_weights", up.weight_ptrs),
                gate.weight_ptrs,
                gate.stat_ptrs,
                up.weight_ptrs,
                up.stat_ptrs,
                down.weight_ptrs,
                down.stat_ptrs,
                gate.source_type,
                down.source_type,
                down.decoder,
                self.num_experts,
                gate.group,
                self.intermediate_size,
                self.raw_batches,
                self.vector_bands,
                down.down_vector_batches,
                self.dp4a_batches,
                self.q8_intermediate_batches,
            )
        ids = topk_ids
        mask = None
        if layer.expert_map is not None:
            ids = layer.expert_map[ids]
            mask = ids >= 0
            ids = ids.clamp_min(0)
        tokens, top_k = ids.shape
        small_routing = self.small_routing and layer.expert_map is None
        if small_routing:
            routed, offsets, sorted_ids, inverse = (
                torch.ops.vllm.sm70_small_expert_route(x, topk_ids, self.num_experts)
            )
        else:
            ids = ids.long()
            sorted_ids, order = ids.reshape(-1).sort()
            boundaries = torch.arange(
                self.num_experts + 1, device=x.device, dtype=torch.int64
            )
            offsets = torch.searchsorted(sorted_ids, boundaries).to(torch.int32)
            routed = x[torch.div(order, top_k, rounding_mode="floor")].contiguous()
        bank = layer.gguf_expert_banks
        if self.raw_gate_up and layer.expert_map is None:
            gate_bank, up_bank = bank["w1"], bank["w3"]
            gate, up = torch.ops.vllm.gguf_expert_gate_up(
                routed,
                offsets,
                sorted_ids,
                gate_bank.raw_weights,
                up_bank.raw_weights,
                gate_bank.weight_ptrs,
                gate_bank.stat_ptrs,
                up_bank.weight_ptrs,
                up_bank.stat_ptrs,
                gate_bank.source_type,
                self.num_experts,
                gate_bank.group,
                gate_bank.n,
                top_k,
                self.raw_batches,
                self.vector_bands,
            )
        else:
            gate = bank["w1"](routed, offsets, sorted_ids)
            up = bank["w3"](routed, offsets, sorted_ids)
        hidden = torch.nn.functional.silu(gate) * up
        down = bank["w2"](hidden.contiguous(), offsets, sorted_ids)
        if small_routing:
            return torch.ops.vllm.sm70_small_expert_unroute(down, inverse, topk_weights)
        restored = down[order.argsort()].view(tokens, top_k, self.hidden_size)
        if mask is not None:
            restored = torch.where(mask[..., None], restored, 0)
        return (restored.float() * topk_weights[..., None].float()).sum(1).to(x.dtype)
