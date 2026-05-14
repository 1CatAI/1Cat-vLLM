import torch
import traceback
import os
import flash_attn_v100_cuda
from typing import Optional, Sequence, Tuple, Union

DEFAULT_DECODE_PARTITION_SIZE = 256
LONG_CONTEXT_DECODE_PARTITION_SIZE = 512
LONG_CONTEXT_DECODE_PARTITION_THRESHOLD = 20480
VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
_decode_workspace_cache = {}

def maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _get_decode_workspace(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
):
    batch_capacity = block_table.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    max_seq_capacity = block_table.shape[1] * k_cache.shape[1]
    partition_size = _get_decode_partition_size(max_seq_capacity)
    max_num_partitions = (max_seq_capacity + partition_size - 1) // partition_size
    device_index = q.device.index if q.device.index is not None else -1
    key = (device_index, batch_capacity, num_heads, head_dim, max_num_partitions, partition_size)

    workspace = _decode_workspace_cache.get(key)
    if workspace is None:
        workspace = (
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions, head_dim),
                dtype=torch.float16,
                device=q.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=q.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=q.device,
            ),
        )
        _decode_workspace_cache[key] = workspace

    return workspace, partition_size


def _get_decode_partition_size(max_seq_capacity: int) -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    if raw is None:
        if max_seq_capacity > LONG_CONTEXT_DECODE_PARTITION_THRESHOLD:
            return LONG_CONTEXT_DECODE_PARTITION_SIZE
        return DEFAULT_DECODE_PARTITION_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    if value not in VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{VALID_DECODE_PARTITION_SIZES}, got {value}"
        )
    return value


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor,
    return_softmax: bool,
) -> tuple:
    q, k, v = map(maybe_contiguous, (q, k, v))
    out = maybe_contiguous(out)
    if out is None:
        out = torch.zeros_like(q)
    lse = torch.zeros(q.shape[0] * q.shape[1] * q.shape[2], dtype=torch.float32, device=q.device)
    outputs = flash_attn_v100_cuda.fwd(
        q, k, v,
        out, alibi_slopes,
        dropout_p, softmax_scale, causal,
        window_size_left, window_size_right,
        softcap, return_softmax, None
    )
    return outputs[0], outputs[1], None, None

def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    softcap: float,
    alibi_slopes: torch.Tensor,
    deterministic: bool,
    rng_state: torch.Tensor = None,
) -> torch.Tensor:
    dout, q, k, v, out = map(maybe_contiguous, (dout, q, k, v, out))
    grads = flash_attn_v100_cuda.bwd(
        dout, q, k, v, out, softmax_lse,
        dq, dk, dv,
        alibi_slopes,
        dropout_p, softmax_scale, causal,
        window_size_left, window_size_right,
        softcap, deterministic, None, rng_state
    )
    return grads[0], grads[1], grads[2]

class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dropout_p: float,
        softmax_scale: float,
        causal: bool,
        window_size: tuple,
        softcap: float,
        alibi_slopes: torch.Tensor,
        deterministic: bool,
        return_softmax: bool,
        is_grad_enabled: bool,
        out: Optional[torch.Tensor],
    ):

        q_ = q.permute(0, 2, 1, 3).contiguous()
        k_ = k.permute(0, 2, 1, 3).contiguous()
        v_ = v.permute(0, 2, 1, 3).contiguous()

        B, M, H, D = q.shape
        _, N, _, _ = k.shape

        if D % 8 != 0:
            raise ValueError(f"head_dim={D} must be divisible by 8 for Volta kernel")

        if dropout_p != 0.0:
            raise NotImplementedError("dropout_p != 0.0 not supported")

        if alibi_slopes is not None:
            raise NotImplementedError("alibi_slopes not supported")

        if softcap != 0.0:
            raise NotImplementedError("softcap != 0.0 not supported")

        if q_.shape[1] % k_.shape[1] != 0:
            raise ValueError(
                f"invalid head mapping: q has {q_.shape[1]} heads, "
                f"k has {k_.shape[1]} heads"
            )
        if k_.shape[1] != v_.shape[1]:
            raise ValueError(
                f"k/v head mismatch: k has {k_.shape[1]}, v has {v_.shape[1]}"
            )

        window_size_left, window_size_right = window_size
        if causal and (window_size_left != -1 or window_size_right != -1):
            if window_size_left > 0 and window_size_right > 0:
                window_size_left, window_size_right = -1, -1
            else:
                raise NotImplementedError(f"Unsupported window_size={window_size} with causal=True")

        out_, lse_, _, rng_state = _flash_attn_forward(
            q_, k_, v_,
            out.permute(0, 2, 1, 3).contiguous() if out is not None else None,
            dropout_p, softmax_scale, causal,
            window_size_left, window_size_right,
            softcap, alibi_slopes, return_softmax
        )

        out = out_.permute(0, 2, 1, 3).contiguous()

        if is_grad_enabled and q.requires_grad:
            ctx.save_for_backward(q_, k_, v_, out_, lse_, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.softcap = softcap
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        return out if not return_softmax else (out, lse_, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q_, k_, v_, out_, lse_, rng_state = ctx.saved_tensors

        dout_ = dout.permute(0, 2, 1, 3).contiguous()

        dq_ = torch.empty_like(q_)
        dk_ = torch.empty_like(k_)
        dv_ = torch.empty_like(v_)

        _flash_attn_backward(
            dout_, q_, k_, v_, out_, lse_,
            dq_, dk_, dv_,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.softcap,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state,
        )

        dq = dq_.permute(0, 2, 1, 3)
        dk = dk_.permute(0, 2, 1, 3)
        dv = dv_.permute(0, 2, 1, 3)

        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None

def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    out: Optional[torch.Tensor] = None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    try:
        return FlashAttnFunc.apply(
            q, k, v,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            softcap,
            alibi_slopes,
            deterministic,
            return_attn_probs,
            torch.is_grad_enabled(),
            out,
        )
    except Exception as e:
        print("VOLTA FA2 FAILED in flash_attn_func")
        print(f"  q.shape = {list(q.shape)}, dtype = {q.dtype}, device = {q.device}, contiguous = {q.is_contiguous()}")
        print(f"  k.shape = {list(k.shape)}, dtype = {k.dtype}, device = {k.device}, contiguous = {k.is_contiguous()}")
        print(f"  v.shape = {list(v.shape)}, dtype = {v.dtype}, device = {v.device}, contiguous = {v.is_contiguous()}")
        print(f"  causal = {causal}, window_size = {window_size}, softmax_scale = {softmax_scale}")
        print(f"  Exception type: {type(e).__name__}")
        print(f"  Exception message: {e}")
        traceback.print_exc()
        raise

def flash_attn_decode_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    (tmp_out, max_logits, exp_sums), partition_size = _get_decode_workspace(
        q, k_cache, block_table
    )

    return flash_attn_v100_cuda.decode_paged_fwd(
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        softmax_scale,
        partition_size,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
    )

def flash_attn_prefill_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = True,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)

    q_ = q.permute(0, 2, 1, 3).contiguous()
    out_ = out.permute(0, 2, 1, 3).contiguous() if out is not None else None

    out_ = flash_attn_v100_cuda.prefill_paged_fwd(
        q_,
        k_cache,
        v_cache,
        out_,
        block_table,
        seq_lens,
        softmax_scale,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        causal,
    )
    return out_.permute(0, 2, 1, 3).contiguous()


__all__ = ["flash_attn_func", "flash_attn_decode_paged", "flash_attn_prefill_paged"]
