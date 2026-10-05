# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Research QSA phase: both norm/rope paths, cache publication and output init.

No model dispatch is installed on import. Attach only inside a leased whole
layer benchmark. Attention core, selection and projection retain native paths.
"""

from types import MethodType

import torch
import torch.nn.functional as F

from vllm.forward_context import get_forward_context
from vllm.models.qwen4_exp.nvidia.ops.qsa_pre_indexer import _qsa_pre_indexer_kernel
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
)


@triton.jit
def _joint_prepare(
    q_ptr,
    q_stride_token,
    k_ptr,
    k_stride_token,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    cos_sin_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    q_out_ptr,
    q_out_stride_token,
    q_out_stride_head,
    state_cache_ptr,
    state_cache_stride_block,
    state_cache_stride_token,
    state_slots_ptr,
    state_table_ptr,
    state_table_stride_req,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    k_work_metadata_ptr,
    compressed_cache_ptr,
    compressed_cache_stride_block,
    compressed_cache_stride_token,
    num_tokens,
    num_state_blocks,
    num_compressed_blocks,
    num_k_work,
    MainQKV,
    MainQW,
    MainKW,
    MainCos,
    MainQOut,
    MainKOut,
    MainGate,
    MainSlots,
    MainKCache,
    MainVCache,
    AttnOutput,
    HQ: tl.constexpr,
    D: tl.constexpr,
    TILE_T_Q: tl.constexpr,
    TILE_H_Q: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE_SIZE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    IS_K_MROPE: tl.constexpr,
    CACHE_HAS_ROPE_POS: tl.constexpr,
    CACHE_IS_FP16: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
    BaseGrid: tl.constexpr,
    MainTokens: tl.constexpr,
    MainRow: tl.constexpr,
    MainCacheBlock: tl.constexpr,
    MainCacheToken: tl.constexpr,
    MainPage: tl.constexpr,
    MainCosRows: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < BaseGrid:
        _qsa_pre_indexer_kernel(
            q_ptr,
            q_stride_token,
            k_ptr,
            k_stride_token,
            pos_ptr,
            pos_stride_axis,
            pos_stride_token,
            cos_sin_ptr,
            q_norm_weight_ptr,
            k_norm_weight_ptr,
            eps,
            q_out_ptr,
            q_out_stride_token,
            q_out_stride_head,
            state_cache_ptr,
            state_cache_stride_block,
            state_cache_stride_token,
            state_slots_ptr,
            state_table_ptr,
            state_table_stride_req,
            query_start_loc_ptr,
            logical_positions_ptr,
            compressed_slots_ptr,
            k_work_metadata_ptr,
            compressed_cache_ptr,
            compressed_cache_stride_block,
            compressed_cache_stride_token,
            num_tokens,
            num_state_blocks,
            num_compressed_blocks,
            num_k_work,
            HQ,
            D,
            TILE_T_Q,
            TILE_H_Q,
            COMPRESS_RATIO,
            STATE_SIZE,
            COMP_PAGE_SIZE,
            IS_2D_POSITIONS,
            IS_K_MROPE,
            CACHE_HAS_ROPE_POS,
            CACHE_IS_FP16,
            MROPE_H,
            MROPE_W,
        )
    else:
        token = (pid - BaseGrid) // 7
        head = (pid - BaseGrid) % 7
        col = tl.arange(0, 256)
        if head < 6:
            source = MainQKV + token * MainRow + head * 512
            weight = MainQW
            destination = MainQOut + token * 1536 + head * 256
            gate = tl.load(MainQKV + token * MainRow + head * 512 + 256 + col)
            tl.store(MainGate + token * 1536 + head * 256 + col, gate)
            tl.store(AttnOutput + token * 1536 + head * 256 + col, 0)
        else:
            source = MainQKV + token * MainRow + 3072
            weight = MainKW
            destination = MainKOut + token * 256
        value = tl.load(source + col).to(tl.float32)
        inverse = tl.rsqrt(tl.sum(value * value, 0) / 256.0 + eps)
        normalized = value * inverse * (1.0 + tl.load(weight + col).to(tl.float32))
        partner = tl.where(col < 32, col + 32, tl.where(col < 64, col - 32, col))
        other = tl.load(source + partner).to(tl.float32)
        other = other * inverse * (1.0 + tl.load(weight + partner).to(tl.float32))
        frequency = col % 32
        plane = tl.full((256,), 0, tl.int32)
        if IS_2D_POSITIONS:
            plane = tl.where((frequency % 3 == 1) & (frequency < 33), 1, plane)
            plane = tl.where((frequency % 3 == 2) & (frequency < 30), 2, plane)
        position = tl.load(pos_ptr + plane * pos_stride_axis + token * pos_stride_token)
        position = tl.where(position < 0, position + MainCosRows, position)
        cosine = tl.load(MainCos + position * 64 + frequency).to(tl.float32)
        sine = tl.load(MainCos + position * 64 + 32 + frequency).to(tl.float32)
        second = other * sine
        second = tl.where(col < 32, -second, second)
        rotated = tl.fma(normalized, cosine, second)
        processed = tl.where(col < 64, rotated, normalized).to(tl.float16)
        tl.store(destination + col, processed)
        slot = tl.load(MainSlots + token)
        if slot >= 0:
            offset = (
                (slot // MainPage) * MainCacheBlock
                + (slot % MainPage) * MainCacheToken
                + col
            )
            if head == 6:
                tl.store(MainKCache + offset, processed)
            elif head == 0:
                tl.store(
                    MainVCache + offset, tl.load(MainQKV + token * MainRow + 3328 + col)
                )


def _transaction(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    joint_projection: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.qsa import (
        qsa_select_paged_tokens,
        qsa_sparse_paged_attention,
    )

    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    indexer = layer.indexer
    metadata = get_forward_context().attn_metadata[layer.layer_name]
    raw, compressed = indexer._metadata()
    count = metadata.num_actual_tokens
    qkv = joint_projection[:, :3584]
    assert count == qkv.shape[0] and count in (1, 5)
    assert layer.kv_cache.dtype == qkv.dtype == torch.float16
    assert not layer.qsa_dcp_sharded
    projected = joint_projection[:, 3584:]
    query, raw_keys = projected.split((512, 128), dim=-1)
    index_query = query.new_empty((count, 4, 128))
    main_query = qkv.new_empty((count, 1536))
    main_key = qkv.new_empty((count, 256))
    main_gate = qkv.new_empty((count, 1536))
    state = indexer.raw_key_cache.kv_cache
    compressed_cache = indexer.compressed_key_cache.kv_cache
    main_k, main_v = layer.kv_cache.unbind(1)
    main_k = canonicalize_singleton_dim_strides(main_k)
    main_v = canonicalize_singleton_dim_strides(main_v)
    is_2d = positions.ndim == 2
    axis_stride = positions.stride(0) if is_2d else 0
    token_stride = positions.stride(-1)
    section = tuple(indexer.rotary_emb.mrope_section)
    key_work = compressed.k_work_metadata.shape[0]
    base_grid = key_work + triton.cdiv(count, 2) * 2
    rope_positions = indexer.raw_key_cache.rope_position_cache is not None
    _joint_prepare[(base_grid + count * 7,)](
        query,
        query.stride(0),
        raw_keys,
        raw_keys.stride(0),
        positions,
        axis_stride,
        token_stride,
        indexer.rotary_emb.cos_sin_cache,
        indexer.q_layernorm.weight,
        indexer.k_layernorm.weight,
        indexer.q_layernorm.variance_epsilon,
        index_query,
        index_query.stride(0),
        index_query.stride(1),
        state,
        state.stride(0),
        state.stride(1),
        raw.slot_mapping,
        raw.block_table,
        raw.block_table.stride(0),
        raw.query_start_loc,
        raw.logical_positions,
        compressed.slot_mapping,
        compressed.k_work_metadata,
        compressed_cache,
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        count,
        state.shape[0],
        compressed_cache.shape[0],
        key_work,
        qkv,
        layer.q_norm.weight,
        layer.k_norm.weight,
        layer.rotary_emb.cos_sin_cache,
        main_query,
        main_key,
        main_gate,
        metadata.slot_mapping,
        main_k,
        main_v,
        output,
        HQ=4,
        D=128,
        TILE_T_Q=2,
        TILE_H_Q=2,
        COMPRESS_RATIO=4,
        STATE_SIZE=state.shape[1],
        COMP_PAGE_SIZE=compressed_cache.shape[1],
        IS_2D_POSITIONS=is_2d,
        IS_K_MROPE=True,
        CACHE_HAS_ROPE_POS=rope_positions,
        CACHE_IS_FP16=True,
        MROPE_H=section[1],
        MROPE_W=section[2],
        BaseGrid=base_grid,
        MainTokens=count,
        MainRow=qkv.stride(0),
        MainCacheBlock=main_k.stride(0),
        MainCacheToken=main_k.stride(1),
        MainPage=main_k.shape[1],
        MainCosRows=layer.rotary_emb.cos_sin_cache.shape[0],
        num_warps=2,
    )
    selected = layer.topk_indices_buffer[:count]
    if not indexer.skip_topk:
        qsa_select_paged_tokens(
            index_query,
            compressed_cache,
            compressed.block_table,
            compressed.token_to_req,
            compressed.logical_positions,
            compressed.seq_lens,
            2048,
            4,
            selected,
            query_start_loc_cpu=compressed.query_start_loc_cpu,
        )
    qsa_sparse_paged_attention(
        main_query.view(count, 6, 256),
        main_k,
        main_v,
        selected,
        metadata.block_table,
        raw.token_to_req[:count],
        output[:count],
        kv_cache_dtype=layer.kv_cache_dtype,
        k_scale=layer._k_scale_float,
        v_scale=layer._v_scale_float,
        output_gate=main_gate.view(count, 6, 256),
        query_positions=raw.logical_positions[:count],
        sequence_lengths=raw.seq_lens,
    )


def _fake(
    hidden: torch.Tensor,
    positions: torch.Tensor,
    joint_projection: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    pass


direct_register_custom_op(
    op_name="sm70_qsa_jointprep_research",
    op_func=_transaction,
    fake_impl=_fake,
    mutates_args=["output"],
)


def _joint_projection(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    from vllm.models.qwen4_exp.nvidia.sm70_fp16_gemv import (
        _qwen38_fp16_row_gemv_kernel,
    )

    if hidden.shape == (1, 2560):
        output = hidden.new_empty((1, 4224))
        _qwen38_fp16_row_gemv_kernel[(4224, 1)](
            hidden,
            weight,
            output,
            K=2560,
            BLOCK_K=512,
            LOAD_POLICY=0,
            N=4224,
            num_warps=2,
        )
        return output
    return F.linear(hidden, weight)


def _joint_projection_fake(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], 4224))


direct_register_custom_op(
    op_name="sm70_qsa_joint_projection_research",
    op_func=_joint_projection,
    fake_impl=_joint_projection_fake,
    mutates_args=[],
)


def attach(layer) -> None:
    original = layer.forward
    layer._qsa_jointprep_original = original
    layer._qsa_jointprep_enabled = False
    assert layer.qkv_proj.weight.shape == (3584, 2560)
    assert layer.indexer.index_qk_proj.weight.shape == (640, 2560)
    layer.register_buffer(
        "_qsa_jointprep_weight",
        torch.cat((layer.qkv_proj.weight, layer.indexer.index_qk_proj.weight), 0),
        persistent=False,
    )

    def forward(self, positions, output, hidden_states):
        if not self._qsa_jointprep_enabled:
            return self._qsa_jointprep_original(positions, output, hidden_states)
        projected = torch.ops.vllm.sm70_qsa_joint_projection_research(
            hidden_states, self._qsa_jointprep_weight
        )
        attention = hidden_states.new_empty((hidden_states.shape[0], 6, 256))
        torch.ops.vllm.sm70_qsa_jointprep_research(
            hidden_states,
            positions,
            projected,
            attention,
            _encode_layer_name(self.layer_name),
        )
        result, _ = self.o_proj(attention.flatten(1))
        if output is not None:
            output.copy_(result)
        return result

    layer.forward = MethodType(forward, layer)
