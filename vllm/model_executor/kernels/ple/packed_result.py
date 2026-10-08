# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transport selected IQ4_NL rows and decode them on the consumer stream."""

import torch

from vllm.config import get_current_vllm_config
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


def packed_result_capability(
    *, enabled, source_type, row_width, heads, fp16, sm70, offloaded, local_tables
):
    reason = None
    if not enabled:
        reason = "disabled_by_kernel_config"
    elif source_type != 20:
        reason = "requires_iq4nl_source"
    elif row_width <= 0 or row_width % 32 or heads <= 0:
        reason = "requires_complete_iq4nl_rows"
    elif not fp16 or not sm70:
        reason = "requires_sm70_fp16_output"
    elif not offloaded:
        reason = "cpu_result_transport_not_active"
    elif local_tables:
        reason = "local_table_decode_takes_precedence"
    return dict(
        enabled=reason is None,
        reason=reason,
        operator="ple_decode_iq4nl_result",
        source_type=source_type,
        row_width=row_width,
        heads=heads,
        packed_width=heads * row_width // 32 * 18,
        output_dtype="float16",
        graph_safe=True,
    )


def prepare_packed_gguf_results(config, tensors, names):
    import vllm.envs as envs
    from vllm.model_executor.kernels.ple.gguf_pinned import pinned_decode_active
    from vllm.model_executor.layers.ple_offload_layer import ple_offload_enabled
    from vllm.platforms import current_platform

    policy = config.kernel_config
    text = config.model_config.hf_text_config
    for raw, name in names.items():
        if not name.endswith(".ple_embedding.ngram_embedding.weight"):
            continue
        tensor = tensors[raw]
        k = int(tensor.shape[0])
        embedding_dim = int(text.ple_embed_dim)
        heads = embedding_dim // k if k > 0 and embedding_dim % k == 0 else 0
        policy.ple_packed_result_decoders[
            name.removesuffix(".ngram_embedding.weight")
        ] = packed_result_capability(
            enabled=policy.ple_packed_gguf_results,
            source_type=int(tensor.tensor_type),
            row_width=k,
            heads=heads,
            fp16=config.model_config.dtype == torch.float16,
            sm70=current_platform.is_device_capability(70),
            offloaded=ple_offload_enabled(),
            local_tables=(
                envs.VLLM_SM70_QWEN38_HYBRID_PLE
                or pinned_decode_active(config)
                or policy.ple_disk_cascade_active
            ),
        )


def packed_result_layout(layer_name):
    statuses = getattr(
        get_current_vllm_config().kernel_config, "ple_packed_result_decoders", {}
    )
    status = statuses.get(layer_name)
    return status if status and status["enabled"] else None


@triton.jit
def _decode(
    PACKET, BOOK, OUTPUT, K: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < K
    base = PACKET + row * WIDTH + (col // 32) * 18
    lo = tl.load(base, valid, other=0).to(tl.uint16)
    hi = tl.load(base + 1, valid, other=0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
    packed = tl.load(base + 2 + col % 16, valid, other=0)
    code = (packed >> (4 * ((col % 32) // 16))) & 15
    value = tl.load(BOOK + code.to(tl.int32)).to(tl.float32)
    tl.store(OUTPUT + row * K + col, scale * value, valid)


def _result_shape(packet: torch.Tensor, codebook: torch.Tensor, row_width: int):
    if row_width <= 0 or row_width % 32:
        raise ValueError("Packed PLE result requires complete IQ4_NL rows")
    width = row_width // 32 * 18
    if (
        packet.ndim != 2
        or packet.dtype != torch.uint8
        or not packet.is_cuda
        or not packet.is_contiguous()
        or packet.shape[1] <= 0
        or packet.shape[1] % width
        or codebook.shape != (16,)
        or codebook.dtype != torch.float32
        or not codebook.is_contiguous()
        or codebook.device != packet.device
    ):
        raise ValueError(
            "Packed PLE result has incompatible packet or codebook storage"
        )
    return packet.shape[0], packet.shape[1] // width * row_width


def decode_iq4nl_result_out(
    packet: torch.Tensor,
    codebook: torch.Tensor,
    row_width: int,
    output: torch.Tensor,
) -> None:
    shape = _result_shape(packet, codebook, row_width)
    if (
        output.shape != shape
        or output.dtype != torch.float16
        or output.device != packet.device
        or not output.is_contiguous()
    ):
        raise ValueError("Packed PLE decoded output has incompatible storage")
    if not packet.shape[0]:
        return
    heads = shape[1] // row_width
    width = row_width // 32 * 18
    _decode[(packet.shape[0] * heads,)](
        packet,
        codebook,
        output,
        row_width,
        width,
        triton.next_power_of_2(row_width),
        num_warps=4,
    )


def decode_iq4nl_result(
    packet: torch.Tensor, codebook: torch.Tensor, row_width: int
) -> torch.Tensor:
    output = torch.empty(
        _result_shape(packet, codebook, row_width),
        device=packet.device,
        dtype=torch.float16,
    )
    decode_iq4nl_result_out(packet, codebook, row_width, output)
    return output


def _decode_fake(
    packet: torch.Tensor, codebook: torch.Tensor, row_width: int
) -> torch.Tensor:
    heads = packet.shape[1] // (row_width // 32 * 18)
    return torch.empty(
        (packet.shape[0], heads * row_width), device=packet.device, dtype=torch.float16
    )


direct_register_custom_op(
    op_name="ple_decode_iq4nl_result",
    op_func=decode_iq4nl_result,
    fake_impl=_decode_fake,
)


def _decode_out_fake(
    packet: torch.Tensor,
    codebook: torch.Tensor,
    row_width: int,
    output: torch.Tensor,
) -> None:
    return


direct_register_custom_op(
    op_name="ple_decode_iq4nl_result_out",
    op_func=decode_iq4nl_result_out,
    mutates_args=["output"],
    fake_impl=_decode_out_fake,
)
