import torch
import logging
import os
try:
    from . import flash_attn_v100_cuda
except ImportError:
    import flash_attn_v100_cuda
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

try:
    from torch._subclasses.fake_tensor import FakeTensor
except ImportError:
    FakeTensor = None

DEFAULT_DECODE_PARTITION_SIZE = 256
VALID_DECODE_PARTITION_SIZES = (256, 512, 1024)
DECODE_MQ_Q_MAX = 8
_decode_plan_cache = {}
_decode_workspace_cache = {}
_decode_mq_workspace_cache = {}
_turboquant_decode_workspace_cache = {}
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _DecodePlan:
    partition_size: int
    actual_num_partitions: int
    launch_num_partitions: int
    workspace_num_partitions: int


@dataclass
class _DecodeWorkspace:
    tmp_out: torch.Tensor
    max_logits: torch.Tensor
    exp_sums: torch.Tensor
    active_num_partitions: torch.Tensor
    max_num_partitions: int


def maybe_contiguous(x):
    return x.contiguous() if x is not None and not x.is_contiguous() else x


def _copy_bhmd_to_bmhd_out(
    out_bhmd: torch.Tensor,
    out_bmhd: Optional[torch.Tensor],
) -> torch.Tensor:
    out = out_bhmd.permute(0, 2, 1, 3).contiguous()
    if out_bmhd is not None:
        out_bmhd.copy_(out)
        return out_bmhd
    return out


def _is_fake_tensor(x: torch.Tensor) -> bool:
    return FakeTensor is not None and isinstance(x, FakeTensor)


def _can_cache_workspace(x: torch.Tensor) -> bool:
    return (
        not torch.compiler.is_compiling()
        and not x.is_meta
        and not _is_fake_tensor(x)
    )


def _workspace_stream_id(device: torch.device) -> int:
    if device.type != "cuda":
        return -1
    if torch.compiler.is_compiling():
        return 0
    return int(torch.cuda.current_stream(device).cuda_stream)


def _round_decode_partition_capacity(required_num_partitions: int) -> int:
    if required_num_partitions <= 1:
        return 1
    return 1 << (required_num_partitions - 1).bit_length()


def _decode_dynamic_partitions_enabled() -> bool:
    return os.getenv("VLLM_FLASH_V100_DECODE_DYNAMIC_PARTITIONS", "1") != "0"


def _cuda_graph_capture_active() -> bool:
    is_capturing = getattr(torch.cuda, "is_current_stream_capturing", None)
    if is_capturing is None:
        return False
    try:
        return bool(is_capturing())
    except RuntimeError:
        return False


def _allocate_decode_workspace(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    max_num_partitions: int,
) -> _DecodeWorkspace:
    return _DecodeWorkspace(
        tmp_out=torch.empty(
            (batch_capacity, num_heads, max_num_partitions, head_dim),
            dtype=torch.float16,
            device=q.device,
        ),
        max_logits=torch.empty(
            (batch_capacity, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        exp_sums=torch.empty(
            (batch_capacity, num_heads, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        active_num_partitions=torch.empty(
            (1,),
            dtype=torch.int32,
            device=q.device,
        ),
        max_num_partitions=max_num_partitions,
    )


def _get_decode_plan(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    max_seq_len_hint: Optional[int] = None,
    batch_size_hint: Optional[int] = None,
    workspace_seq_capacity_hint: Optional[int] = None,
    active_num_partitions: Optional[torch.Tensor] = None,
    default_partition_size: Optional[int] = None,
) -> _DecodePlan:
    batch_capacity = batch_size_hint or block_table.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    max_seq_capacity = block_table.shape[1] * k_cache.shape[1]
    effective_max_seq_len = int(max_seq_len_hint or max_seq_capacity)
    effective_workspace_seq_capacity = int(
        workspace_seq_capacity_hint or effective_max_seq_len
    )
    effective_max_seq_len = max(1, effective_max_seq_len)
    effective_workspace_seq_capacity = max(
        1,
        effective_workspace_seq_capacity,
        effective_max_seq_len,
    )
    if (
        workspace_seq_capacity_hint is not None
        and active_num_partitions is None
        and _cuda_graph_capture_active()
    ):
        # CUDA graph replay replays the captured fill_ into active_num_partitions
        # instead of rerunning Python planning for the runtime sequence length.
        # Capture the full workspace envelope so runtime seq_lens, not a stale
        # short capture hint, bounds the effective decode range inside kernels.
        effective_max_seq_len = max(
            effective_max_seq_len,
            effective_workspace_seq_capacity,
        )
    partition_size = _get_decode_partition_size(
        max_seq_capacity=max_seq_capacity,
        head_dim=head_dim,
        num_q_heads=num_heads,
        num_kv_heads=k_cache.shape[2],
        max_seq_len_hint=effective_max_seq_len,
        batch_size_hint=batch_capacity,
        default_partition_size=default_partition_size,
    )
    runtime_num_partitions = max(
        1,
        (effective_max_seq_len + partition_size - 1) // partition_size,
    )
    workspace_num_partitions = max(
        1,
        (effective_workspace_seq_capacity + partition_size - 1) // partition_size,
    )
    plan = _DecodePlan(
        partition_size=partition_size,
        actual_num_partitions=runtime_num_partitions,
        launch_num_partitions=(
            workspace_num_partitions
            if workspace_seq_capacity_hint is not None
            else runtime_num_partitions
        ),
        workspace_num_partitions=workspace_num_partitions,
    )
    device_index = q.device.index if q.device.index is not None else -1
    key = (
        device_index,
        batch_capacity,
        num_heads,
        head_dim,
        q.dtype,
        plan.partition_size,
        plan.actual_num_partitions,
        plan.launch_num_partitions,
        plan.workspace_num_partitions,
    )
    if _can_cache_workspace(q) and workspace_seq_capacity_hint is None:
        cached = _decode_plan_cache.get(key)
        if cached is not None:
            return cached
        _decode_plan_cache[key] = plan
    return plan


def _get_decode_workspace_for_plan(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    plan: _DecodePlan,
    active_num_partitions: Optional[torch.Tensor] = None,
):
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = _workspace_stream_id(q.device)
    key = (
        device_index,
        stream_id,
        batch_capacity,
        num_heads,
        head_dim,
        plan.partition_size,
    )

    workspace = _decode_workspace_cache.get(key) if _can_cache_workspace(q) else None
    if (
        workspace is None
        or workspace.max_num_partitions < plan.workspace_num_partitions
    ):
        workspace = _allocate_decode_workspace(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            max_num_partitions=_round_decode_partition_capacity(
                plan.workspace_num_partitions
            ),
        )
        if _can_cache_workspace(q):
            _decode_workspace_cache[key] = workspace

    if active_num_partitions is None:
        workspace.active_num_partitions.fill_(plan.actual_num_partitions)
        active_num_partitions = workspace.active_num_partitions
    return (
        workspace.tmp_out[:, :, :workspace.max_num_partitions, :],
        workspace.max_logits[:, :, :workspace.max_num_partitions],
        workspace.exp_sums[:, :, :workspace.max_num_partitions],
        active_num_partitions,
    )


@dataclass
class _DecodeWorkspaceMq:
    tmp_out: torch.Tensor
    max_logits: torch.Tensor
    exp_sums: torch.Tensor
    active_num_partitions: torch.Tensor
    max_num_partitions: int


def _allocate_decode_workspace_mq(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    q_max: int,
    max_num_partitions: int,
) -> _DecodeWorkspaceMq:
    return _DecodeWorkspaceMq(
        tmp_out=torch.empty(
            (batch_capacity, num_heads, q_max, max_num_partitions, head_dim),
            dtype=torch.float16,
            device=q.device,
        ),
        max_logits=torch.empty(
            (batch_capacity, num_heads, q_max, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        exp_sums=torch.empty(
            (batch_capacity, num_heads, q_max, max_num_partitions),
            dtype=torch.float32,
            device=q.device,
        ),
        active_num_partitions=torch.empty(
            (1,),
            dtype=torch.int32,
            device=q.device,
        ),
        max_num_partitions=max_num_partitions,
    )


def _get_decode_workspace_mq_for_plan(
    q: torch.Tensor,
    *,
    batch_capacity: int,
    num_heads: int,
    head_dim: int,
    plan: _DecodePlan,
    active_num_partitions: Optional[torch.Tensor] = None,
):
    device_index = q.device.index if q.device.index is not None else -1
    stream_id = _workspace_stream_id(q.device)
    key = (
        "mq",
        device_index,
        stream_id,
        batch_capacity,
        num_heads,
        head_dim,
        DECODE_MQ_Q_MAX,
        plan.partition_size,
    )

    workspace = (
        _decode_mq_workspace_cache.get(key) if _can_cache_workspace(q) else None
    )
    if (
        workspace is None
        or workspace.max_num_partitions < plan.workspace_num_partitions
    ):
        workspace = _allocate_decode_workspace_mq(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            q_max=DECODE_MQ_Q_MAX,
            max_num_partitions=_round_decode_partition_capacity(
                plan.workspace_num_partitions
            ),
        )
        if _can_cache_workspace(q):
            _decode_mq_workspace_cache[key] = workspace

    if active_num_partitions is None:
        workspace.active_num_partitions.fill_(plan.actual_num_partitions)
        active_num_partitions = workspace.active_num_partitions
    return (
        workspace.tmp_out[:, :, :, :workspace.max_num_partitions, :],
        workspace.max_logits[:, :, :, :workspace.max_num_partitions],
        workspace.exp_sums[:, :, :, :workspace.max_num_partitions],
        active_num_partitions,
    )


def _get_turboquant_decode_workspace(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    num_kv_splits: int,
):
    batch_capacity = block_table.shape[0]
    num_heads = q_rot.shape[1]
    head_dim = q_rot.shape[2]
    max_seq_capacity = block_table.shape[1] * kv_cache.shape[1]
    per_split_capacity = (max_seq_capacity + num_kv_splits - 1) // num_kv_splits
    partition_size = next(
        (size for size in VALID_DECODE_PARTITION_SIZES
         if per_split_capacity <= size),
        None,
    )
    if partition_size is None:
        raise ValueError(
            "TurboQuant Flash-V100 decode cannot cover max_seq_capacity="
            f"{max_seq_capacity} with num_kv_splits={num_kv_splits}; "
            f"largest split tile is {VALID_DECODE_PARTITION_SIZES[-1]}"
        )

    device_index = q_rot.device.index if q_rot.device.index is not None else -1
    key = (
        "turboquant",
        device_index,
        batch_capacity,
        num_heads,
        head_dim,
        num_kv_splits,
        partition_size,
    )

    workspace = _turboquant_decode_workspace_cache.get(key)
    if workspace is None:
        workspace = (
            torch.empty(
                (batch_capacity, num_heads, num_kv_splits, head_dim),
                dtype=torch.float32,
                device=q_rot.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, num_kv_splits),
                dtype=torch.float32,
                device=q_rot.device,
            ),
            torch.empty(
                (batch_capacity, num_heads, num_kv_splits),
                dtype=torch.float32,
                device=q_rot.device,
            ),
        )
        _turboquant_decode_workspace_cache[key] = workspace

    return workspace, partition_size


def _get_decode_partition_size(
    max_seq_capacity: int,
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    max_seq_len_hint: Optional[int] = None,
    batch_size_hint: Optional[int] = None,
    default_partition_size: Optional[int] = None,
) -> int:
    raw = os.getenv("VLLM_FLASH_V100_DECODE_PARTITION_SIZE")
    if raw is None:
        if default_partition_size is not None:
            return _validate_decode_partition_size(
                default_partition_size,
                "default_partition_size",
            )
        return _select_default_decode_partition_size(max_seq_len_hint)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_FLASH_V100_DECODE_PARTITION_SIZE must be one of "
            f"{VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
        ) from exc
    return _validate_decode_partition_size(
        value,
        "VLLM_FLASH_V100_DECODE_PARTITION_SIZE",
    )


def _select_default_decode_partition_size(
    max_seq_len_hint: Optional[int],
) -> int:
    if max_seq_len_hint is None:
        return DEFAULT_DECODE_PARTITION_SIZE

    seq_len = max(1, int(max_seq_len_hint))
    if seq_len >= 32768:
        return 1024
    return DEFAULT_DECODE_PARTITION_SIZE


def _validate_decode_partition_size(value: int, name: str) -> int:
    if value not in VALID_DECODE_PARTITION_SIZES:
        raise ValueError(
            f"{name} must be one of {VALID_DECODE_PARTITION_SIZES}, got {value}"
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
        if window_size_left < -1 or window_size_right < -1:
            raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")

        out_, lse_, _, rng_state = _flash_attn_forward(
            q_, k_, v_,
            out.permute(0, 2, 1, 3).contiguous() if out is not None else None,
            dropout_p, softmax_scale, causal,
            window_size_left, window_size_right,
            softcap, alibi_slopes, return_softmax
        )

        out = _copy_bhmd_to_bmhd_out(out_, out)

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
    except Exception:
        logger.debug(
            "FlashAttention-V100 flash_attn_func failed "
            "(q=%s/%s/%s, k=%s/%s/%s, v=%s/%s/%s, causal=%s, "
            "window_size=%s, softmax_scale=%s)",
            list(q.shape),
            q.dtype,
            q.device,
            list(k.shape),
            k.dtype,
            k.device,
            list(v.shape),
            v.dtype,
            v.device,
            causal,
            window_size,
            softmax_scale,
            exc_info=True,
        )
        raise


def flash_attn_bhmd_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float = None,
    causal: bool = False,
    window_size: tuple = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes: torch.Tensor = None,
    return_attn_probs: bool = False,
    out: Optional[torch.Tensor] = None,
):
    """Forward-only Flash-V100 dense attention for [B, H, T, D] tensors."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if dropout_p != 0.0:
        raise NotImplementedError("dropout_p != 0.0 not supported")
    if softcap != 0.0:
        raise NotImplementedError("softcap != 0.0 not supported")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes not supported")

    window_size_left, window_size_right = window_size
    out, lse, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        out,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        alibi_slopes,
        return_attn_probs,
    )
    return out if not return_attn_probs else (out, lse, None)


def flash_attn_qk_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
):
    """Debug-only Flash-V100 QK score dump before softmax."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    k = maybe_contiguous(k)
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    return flash_attn_v100_cuda.qk_scores_fwd(q_, k_, softmax_scale, causal)


def flash_attn_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
):
    """Debug-only Flash-V100 softmax LSE dump."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q, k, v = map(maybe_contiguous, (q, k, v))
    q_ = q.permute(0, 2, 1, 3).contiguous()
    k_ = k.permute(0, 2, 1, 3).contiguous()
    v_ = v.permute(0, 2, 1, 3).contiguous()
    _, lse, _, _ = _flash_attn_forward(
        q_,
        k_,
        v_,
        None,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
    )
    return lse


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
    window_size: tuple = (-1, -1),
    max_seq_len_hint: Optional[int] = None,
    workspace_seq_capacity_hint: Optional[int] = None,
    active_num_partitions: Optional[torch.Tensor] = None,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")
    batch_capacity = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    if not _decode_dynamic_partitions_enabled():
        max_seq_len_hint = None
        workspace_seq_capacity_hint = None
        active_num_partitions = None
    plan = _get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len_hint,
        batch_size_hint=batch_capacity,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        active_num_partitions=active_num_partitions,
    )
    tmp_out, max_logits, exp_sums, active_num_partitions = (
        _get_decode_workspace_for_plan(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            plan=plan,
            active_num_partitions=active_num_partitions,
        )
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
        active_num_partitions,
        softmax_scale,
        plan.partition_size,
        plan.launch_num_partitions,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        int(window_size_left),
        int(window_size_right),
    )


def flash_attn_decode_paged_xqa_available() -> bool:
    return hasattr(flash_attn_v100_cuda, "decode_paged_xqa_fwd")


def flash_attn_decode_paged_xqa(
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
    window_size: tuple = (-1, -1),
    max_seq_len_hint: Optional[int] = None,
    workspace_seq_capacity_hint: Optional[int] = None,
    active_num_partitions: Optional[torch.Tensor] = None,
):
    if not flash_attn_decode_paged_xqa_available():
        raise RuntimeError("flash_attn_v100 CUDA extension lacks XQA decode")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")
    batch_capacity = q.shape[0]
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    if not _decode_dynamic_partitions_enabled():
        max_seq_len_hint = None
        workspace_seq_capacity_hint = None
        active_num_partitions = None
    plan = _get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len_hint,
        batch_size_hint=batch_capacity,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        active_num_partitions=active_num_partitions,
    )
    tmp_out, max_logits, exp_sums, active_num_partitions = (
        _get_decode_workspace_for_plan(
            q,
            batch_capacity=batch_capacity,
            num_heads=num_heads,
            head_dim=head_dim,
            plan=plan,
            active_num_partitions=active_num_partitions,
        )
    )

    return flash_attn_v100_cuda.decode_paged_xqa_fwd(
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        active_num_partitions,
        softmax_scale,
        plan.partition_size,
        plan.launch_num_partitions,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        int(window_size_left),
        int(window_size_right),
    )


def flash_attn_decode_paged_mq_available() -> bool:
    return hasattr(flash_attn_v100_cuda, "decode_paged_mq_fwd")


def _mq_default_partition_size() -> Optional[int]:
    raw = os.getenv("VLLM_FLASH_V100_MQ_PARTITION_SIZE")
    if raw is not None:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(
                "VLLM_FLASH_V100_MQ_PARTITION_SIZE must be one of "
                f"{VALID_DECODE_PARTITION_SIZES}, got {raw!r}"
            ) from exc
        return _validate_decode_partition_size(
            value,
            "VLLM_FLASH_V100_MQ_PARTITION_SIZE",
        )
    # Default: match the legacy single-query partition sizing so multi-query
    # outputs stay bit-identical to the folded path (the partition merge
    # order is the only numeric difference between the two). Set
    # VLLM_FLASH_V100_MQ_PARTITION_SIZE=512 for ~10-15% more kernel speed at
    # 128k at the cost of ~1e-4 output drift vs the folded path.
    return None


def flash_attn_decode_paged_mq(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    window_size: tuple = (-1, -1),
    max_seq_len_hint: Optional[int] = None,
    workspace_seq_capacity_hint: Optional[int] = None,
    active_num_partitions: Optional[torch.Tensor] = None,
    q_len_max: Optional[int] = None,
):
    """Multi-query paged decode: each request contributes q_len query tokens.

    ``q``/``out`` have shape [total_tokens, H, D]; ``query_start_loc`` has
    shape [num_reqs + 1] and gives per-request token offsets; ``seq_lens``
    holds per-request total lengths (including the new tokens). Query token j
    of a request attends to KV positions [0, seq_len - q_len + j] inclusive.
    Per-request q_len must be <= q_len_max <= DECODE_MQ_Q_MAX. Requests with
    q_len == 0 or seq_len == 0 are skipped (their rows are left untouched /
    zeroed). ``q_len_max`` should be the batch max query length (host-side);
    it selects the kernel instantiation and defaults to DECODE_MQ_Q_MAX.
    """
    if not flash_attn_decode_paged_mq_available():
        raise RuntimeError(
            "flash_attn_v100 CUDA extension lacks multi-query decode"
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    if q_len_max is None:
        q_len_max = DECODE_MQ_Q_MAX
    q_len_max = max(1, min(int(q_len_max), DECODE_MQ_Q_MAX))

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    query_start_loc = maybe_contiguous(query_start_loc)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")
    num_reqs = query_start_loc.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    if not _decode_dynamic_partitions_enabled():
        max_seq_len_hint = None
        workspace_seq_capacity_hint = None
        active_num_partitions = None
    plan = _get_decode_plan(
        q,
        k_cache,
        block_table,
        max_seq_len_hint=max_seq_len_hint,
        batch_size_hint=num_reqs,
        workspace_seq_capacity_hint=workspace_seq_capacity_hint,
        active_num_partitions=active_num_partitions,
        default_partition_size=_mq_default_partition_size(),
    )
    tmp_out, max_logits, exp_sums, active_num_partitions = (
        _get_decode_workspace_mq_for_plan(
            q,
            batch_capacity=num_reqs,
            num_heads=num_heads,
            head_dim=head_dim,
            plan=plan,
            active_num_partitions=active_num_partitions,
        )
    )

    return flash_attn_v100_cuda.decode_paged_mq_fwd(
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        seq_lens,
        query_start_loc,
        tmp_out,
        max_logits,
        exp_sums,
        active_num_partitions,
        softmax_scale,
        plan.partition_size,
        plan.launch_num_partitions,
        q_len_max,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        int(window_size_left),
        int(window_size_right),
    )


def flash_attn_decode_paged_wmma(
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
    """Single-query decode using the paged-prefill WMMA compute order.

    ``q`` has the same [B, H, D] shape as ``flash_attn_decode_paged``.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)

    return flash_attn_v100_cuda.decode_paged_wmma_fwd(
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        seq_lens,
        softmax_scale,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
    )


def flash_attn_decode_qk_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
):
    """Debug-only scalar paged decode QK score dump before softmax."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    max_seq_capacity = block_table.shape[1] * k_cache.shape[1]
    partition_size = _get_decode_partition_size(
        max_seq_capacity=max_seq_capacity,
        head_dim=q.shape[2],
        num_q_heads=q.shape[1],
        num_kv_heads=k_cache.shape[2],
    )
    return flash_attn_v100_cuda.decode_qk_scores_fwd(
        q,
        k_cache,
        block_table,
        seq_lens,
        softmax_scale,
        partition_size,
        kv_cache_dtype,
        float(k_scale),
    )


def flash_attn_turboquant_decode_paged_available() -> bool:
    return hasattr(flash_attn_v100_cuda, "decode_turboquant_paged_fwd")


def flash_attn_turboquant_decode_paged(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    centroids: torch.Tensor,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    mse_bits: int = 4,
    value_quant_bits: int = 4,
    norm_correction: bool = True,
    num_kv_splits: int = 32,
):
    if not flash_attn_turboquant_decode_paged_available():
        raise RuntimeError("flash_attn_v100 CUDA extension lacks TurboQuant decode")
    if softmax_scale is None:
        softmax_scale = q_rot.shape[-1] ** -0.5

    q_rot = maybe_contiguous(q_rot)
    kv_cache = maybe_contiguous(kv_cache)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    centroids = maybe_contiguous(centroids)
    out = maybe_contiguous(out)
    (tmp_out, max_logits, exp_sums), partition_size = (
        _get_turboquant_decode_workspace(
            q_rot, kv_cache, block_table, int(num_kv_splits)
        )
    )

    return flash_attn_v100_cuda.decode_turboquant_paged_fwd(
        q_rot,
        kv_cache,
        out,
        block_table,
        seq_lens,
        tmp_out,
        max_logits,
        exp_sums,
        centroids,
        softmax_scale,
        partition_size,
        int(mse_bits),
        int(value_quant_bits),
        bool(norm_correction),
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
    window_size: tuple = (-1, -1),
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out_original = out
    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")

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
        int(window_size_left),
        int(window_size_right),
    )
    return _copy_bhmd_to_bmhd_out(out_, out_original)


def flash_attn_prefill_paged_bfla(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    bfla_block_mask: torch.Tensor,
    bfla_mask_block_n: int,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    causal: bool = True,
    window_size: tuple = (-1, -1),
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out_original = out
    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    bfla_block_mask = maybe_contiguous(bfla_block_mask)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")

    q_ = q.permute(0, 2, 1, 3).contiguous()
    out_ = out.permute(0, 2, 1, 3).contiguous() if out is not None else None

    out_ = flash_attn_v100_cuda.prefill_paged_bfla_fwd(
        q_,
        k_cache,
        v_cache,
        out_,
        block_table,
        seq_lens,
        bfla_block_mask,
        int(bfla_mask_block_n),
        softmax_scale,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        causal,
        int(window_size_left),
        int(window_size_right),
    )
    return _copy_bhmd_to_bmhd_out(out_, out_original)


def flash_attn_prefill_paged_splitkv(
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
    window_size: tuple = (-1, -1),
    split_kv_tokens: int = 32768,
    max_seq_len_hint: int = 0,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    out_original = out
    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")

    q_ = q.permute(0, 2, 1, 3).contiguous()
    out_ = out.permute(0, 2, 1, 3).contiguous() if out is not None else None

    out_ = flash_attn_v100_cuda.prefill_paged_splitkv_fwd(
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
        int(window_size_left),
        int(window_size_right),
        int(split_kv_tokens),
        int(max_seq_len_hint),
    )
    return _copy_bhmd_to_bmhd_out(out_, out_original)


def flash_attn_prefill_paged_bhmd(
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
    window_size: tuple = (-1, -1),
):
    """Paged prefill entry for tensors already laid out as [B, H, M, D]."""
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5

    q = maybe_contiguous(q)
    block_table = maybe_contiguous(block_table)
    seq_lens = maybe_contiguous(seq_lens)
    out = maybe_contiguous(out)
    window_size_left, window_size_right = window_size
    if window_size_left < -1 or window_size_right < -1:
        raise ValueError(f"Invalid window_size={window_size}; values must be >= -1")

    return flash_attn_v100_cuda.prefill_paged_fwd(
        q,
        k_cache,
        v_cache,
        out,
        block_table,
        seq_lens,
        softmax_scale,
        kv_cache_dtype,
        float(k_scale),
        float(v_scale),
        causal,
        int(window_size_left),
        int(window_size_right),
    )


__all__ = [
    "flash_attn_func",
    "flash_attn_lse",
    "flash_attn_qk_scores",
    "flash_attn_decode_paged",
    "flash_attn_decode_paged_xqa",
    "flash_attn_decode_paged_xqa_available",
    "flash_attn_decode_paged_wmma",
    "flash_attn_decode_qk_scores",
    "flash_attn_turboquant_decode_paged",
    "flash_attn_turboquant_decode_paged_available",
    "flash_attn_prefill_paged",
    "flash_attn_prefill_paged_bfla",
    "flash_attn_prefill_paged_splitkv",
    "flash_attn_prefill_paged_bhmd",
]
