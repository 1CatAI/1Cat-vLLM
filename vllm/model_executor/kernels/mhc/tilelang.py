# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _require_mhc_activation_dtype(
    tensor: torch.Tensor,
    *same_dtype_tensors: tuple[str, torch.Tensor | None],
) -> torch.dtype:
    """Validate the mHC activation dtype without changing numerical format."""
    dtype = tensor.dtype
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"mHC requires float16 or bfloat16 activations, got {dtype}.")
    for name, other in same_dtype_tensors:
        if other is not None and other.dtype != dtype:
            raise ValueError(
                f"mHC {name} must use the activation dtype {dtype}, got {other.dtype}."
            )
    if dtype == torch.bfloat16 and current_platform.is_cuda():
        capability = current_platform.get_device_capability()
        if capability is None or capability.to_int() < 80:
            raise RuntimeError(
                "mHC BF16 execution requires NVIDIA SM80 or newer. SM70 has no "
                "native BF16 instructions; load the model with an explicit FP16 "
                "dtype (for example, --dtype half). mHC does not cast BF16 "
                "tensors to FP16."
            )
    return dtype


def _prepare_mhc_norm_weight(
    norm_weight: torch.Tensor | None,
    activation_dtype: torch.dtype,
) -> torch.Tensor | None:
    """Preserve native-BF16 behavior while requiring explicit FP16 inputs."""
    if norm_weight is None:
        return None
    if norm_weight.dtype != activation_dtype:
        if activation_dtype == torch.bfloat16:
            # This is the pre-existing native-BF16 behavior. The caller has
            # already rejected BF16 on SM70 before this conversion can happen.
            norm_weight = norm_weight.to(torch.bfloat16)
        else:
            raise ValueError(
                "mHC FP16 requires norm_weight to use torch.float16; "
                f"got {norm_weight.dtype}."
            )
    return norm_weight.contiguous()


def _torch_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
) -> None:
    assert out.shape[0] == 1
    assert sqrsum.shape[0] == 1
    x_float = x.float()
    out[0].copy_(x_float @ fn.t())
    sqrsum[0].copy_(x_float.square().sum(dim=-1))


def _tilelang_hc_prenorm_gemm(
    x: torch.Tensor,
    fn: torch.Tensor,
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    hidden_size: int,
    hc_mult: int,
    tile_n: int = 12,
    n_thr: int = 512,
    n_splits: int = 1,
) -> None:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        hc_prenorm_gemm_block_m_tilelang,
        hc_prenorm_gemm_tilelang,
    )

    activation_dtype = _require_mhc_activation_dtype(x)
    use_fp16 = activation_dtype == torch.float16

    assert out.shape[0] == n_splits
    assert sqrsum.shape[0] == n_splits
    assert x.shape[1] == hc_mult * hidden_size
    assert x.shape[1] % n_splits == 0
    assert (x.shape[1] // n_splits) % n_thr == 0
    use_default_config = tile_n == 12 and n_thr == 512
    if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
        hc_prenorm_gemm_block_m_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            n_thr,
            tile_n,
            2,
            use_fp16=use_fp16,
        )
        return
    if (
        n_splits == 1
        and use_default_config
        and x.shape[0] < 128
        and x.shape[1] % 1024 == 0
    ):
        hc_prenorm_gemm_tilelang(
            x,
            fn,
            out,
            sqrsum,
            hidden_size,
            hc_mult,
            fn.shape[0],
            1024,
            4,
            n_splits,
            use_fp16=use_fp16,
        )
        return
    hc_prenorm_gemm_tilelang(
        x,
        fn,
        out,
        sqrsum,
        hidden_size,
        hc_mult,
        fn.shape[0],
        n_thr,
        tile_n,
        n_splits,
        use_fp16=use_fp16,
    )


def mhc_pre_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.float16 or
            torch.bfloat16. BF16 requires native BF16 CUDA support.
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;
        norm_weight: optional RMSNorm weight, shape (hidden_size,). FP16 requires
            FP16 weights; native BF16 preserves the existing BF16 conversion.
            When provided, RMSNorm is fused into the layer_input write path.
        norm_eps: epsilon for the fused RMSNorm; only consulted when
            norm_weight is given.

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), same dtype as ``residual``
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
    )
    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm
    from vllm.utils.math_utils import cdiv

    activation_dtype = _require_mhc_activation_dtype(residual)
    use_fp16 = activation_dtype == torch.float16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        norm_weight = _prepare_mhc_norm_weight(norm_weight, activation_dtype)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    if use_deep_gemm:
        # these numbers are from deepgemm kernel impl
        block_k = 64
        block_m = 64
        n_splits = compute_num_split(block_k, hc_hidden_size, cdiv(num_tokens, block_m))
    else:
        n_splits = 1

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=activation_dtype, device=residual.device
    )

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    residual_2d = residual_flat.view(num_tokens, hc_mult * hidden_size)
    if use_deep_gemm:
        tf32_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
    else:
        _tilelang_hc_prenorm_gemm(
            residual_2d,
            fn,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            hc_mult,
        )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            use_fp16=use_fp16,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            use_fp16=use_fp16,
        )

    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def _mhc_pre_tilelang_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=residual.dtype,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_pre_broadcast_tilelang(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
    fn_broadcast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the first mHC pre block from a single residual stream.

    SM70 uses the explicit FP16 TileLang prenorm path and never casts BF16
    input. FP16 requires FP16 RMSNorm weights; native BF16 keeps its existing
    RMSNorm-weight conversion.
    """
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_pre_big_fuse_broadcast_with_norm_tilelang,
    )

    if norm_weight is None:
        raise ValueError("Broadcast mHC pre requires a fused RMSNorm weight.")
    activation_dtype = _require_mhc_activation_dtype(residual)
    use_fp16 = activation_dtype == torch.float16
    assert residual.dim() == 2
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hidden_size = residual.shape[-1]
    hc_mult = fn.shape[1] // hidden_size
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    assert fn.shape == (hc_mult3, hc_mult * hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)
    assert fn_broadcast is not None
    assert fn_broadcast.dtype == torch.float32
    assert fn_broadcast.shape == (hc_mult3, hidden_size)
    assert norm_weight.shape == (hidden_size,)
    norm_weight = _prepare_mhc_norm_weight(norm_weight, activation_dtype)
    assert norm_weight is not None

    num_tokens = residual.shape[0]
    residual_out = torch.empty(
        num_tokens,
        hc_mult,
        hidden_size,
        dtype=activation_dtype,
        device=residual.device,
    )
    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=activation_dtype, device=residual.device
    )

    if use_fp16:
        # The SM70 path uses the same FP32 accumulation as the regular
        # TileLang prenorm kernel. Split-K is deliberately disabled: its
        # V100 shape must satisfy the fixed 512-thread reduction contract.
        n_splits = 1
    else:
        from vllm.utils.math_utils import cdiv

        n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))

    gemm_out_mul = torch.empty(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
    )
    gemm_out_sqrsum = torch.empty(
        n_splits, num_tokens, dtype=torch.float32, device=residual.device
    )

    if use_fp16:
        _tilelang_hc_prenorm_gemm(
            residual,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
            hidden_size,
            1,
            n_splits=n_splits,
        )
    else:
        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual,
            fn_broadcast,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )

    mhc_pre_big_fuse_broadcast_with_norm_tilelang(
        gemm_out_mul,
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual,
        residual_out,
        post_mix,
        comb_mix,
        layer_input,
        norm_weight,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_eps,
        n_splits,
        hc_mult,
        use_fp16=use_fp16,
    )
    return (
        residual_out,
        post_mix.unsqueeze(-1),
        comb_mix.view(num_tokens, hc_mult, hc_mult),
        layer_input,
    )


def mhc_post_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_post_tilelang as _mhc_post_kernel,
    )

    activation_dtype = _require_mhc_activation_dtype(residual, ("x", x))
    out = torch.empty_like(residual)
    _mhc_post_kernel(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
        use_fp16=activation_dtype == torch.float16,
    )
    return out


def mhc_fused_post_pre_tilelang(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run one MHC post block followed by the next MHC pre block.

    When ``norm_weight`` is provided, the layer_input_cur output is the
    RMSNorm'd activation (fused into the kernel); otherwise it is the
    raw pre-norm activation as before.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        compute_num_split,
        mhc_fused_tilelang,
        mhc_post_tilelang,
        mhc_pre_big_fuse_tilelang,
        mhc_pre_big_fuse_with_norm_tilelang,
        sm70_mhc_dot_from_fp32_stage_tilelang,
        sm70_mhc_post_fp32_stage_tilelang,
    )
    from vllm.utils.math_utils import cdiv

    activation_dtype = _require_mhc_activation_dtype(residual, ("x", x))
    use_fp16 = activation_dtype == torch.float16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    if norm_weight is not None:
        assert norm_weight.shape == (hidden_size,)
        norm_weight = _prepare_mhc_norm_weight(norm_weight, activation_dtype)

    assert n_splits in (1, 2, 4, 8)
    assert hidden_size % n_splits == 0

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_layer_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_res_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    from vllm.utils.deep_gemm import is_deep_gemm_supported

    use_deep_gemm = is_deep_gemm_supported()
    use_small_fma = num_tokens <= 16
    if use_small_fma:
        # TODO(gnovack): investigate autotuning these heuristics
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        if use_deep_gemm:
            # these number are from deepgemm kernel impl
            block_k = 64
            block_m = 64
            n_splits = compute_num_split(
                block_k, hc_hidden_size, cdiv(num_tokens, block_m)
            )
        else:
            n_splits = 1

    capability = (
        current_platform.get_device_capability() if current_platform.is_cuda() else None
    )
    use_sm70_fp32_stage = (
        envs.VLLM_SM70_DSV4_MHC_FP32_STAGE
        and use_small_fma
        and use_fp16
        and capability is not None
        and capability.to_int() == 70
        and num_tokens == 1
        and hidden_size == 4096
        and hc_mult == 4
        and hc_mult3 == 24
        and tile_n == 2
        and n_splits == 8
    )

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)
    residual_cur_fp32 = (
        torch.empty_like(residual_flat, dtype=torch.float32)
        if use_sm70_fp32_stage
        else None
    )
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        num_tokens,
        hidden_size,
        dtype=activation_dtype,
        device=residual.device,
    )

    if use_small_fma:
        if use_sm70_fp32_stage:
            assert residual_cur_fp32 is not None
            logger.info_once("SM70 DeepSeek V4 mHC FP32 staging decode path enabled.")
            sm70_mhc_post_fp32_stage_tilelang(
                comb_res_mix_flat,
                residual_flat,
                post_layer_mix_flat,
                x_flat,
                residual_cur_fp32,
                residual_cur,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
                n_splits=n_splits,
            )
            sm70_mhc_dot_from_fp32_stage_tilelang(
                residual_cur_fp32,
                fn.view(hc_mult3, hc_mult, hidden_size),
                gemm_out_mul,
                hidden_size,
                hc_mult,
                hc_mult3,
                tile_n=tile_n,
                n_splits=n_splits,
            )
        else:
            mhc_fused_tilelang(
                comb_res_mix_flat,
                residual_flat,
                post_layer_mix_flat,
                x_flat,
                fn.view(hc_mult3, hc_mult, hidden_size),
                gemm_out_mul,
                gemm_out_sqrsum,
                residual_cur,
                hc_mult,
                hidden_size,
                hc_mult3,
                tile_n=tile_n,
                n_splits=n_splits,
                use_fp16=use_fp16,
            )
    else:
        mhc_post_tilelang(
            comb_res_mix_flat,
            residual_flat,
            post_layer_mix_flat,
            x_flat,
            residual_cur,
            residual.shape[-2],
            residual.shape[-1],
            use_fp16=use_fp16,
        )

        residual_cur_2d = residual_cur.view(num_tokens, hc_mult * hidden_size)
        if use_deep_gemm:
            from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm

            tf32_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                n_splits,
            )
        else:
            _tilelang_hc_prenorm_gemm(
                residual_cur_2d,
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                hidden_size,
                hc_mult,
            )

    if norm_weight is None:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            use_fp16=use_fp16,
        )
    else:
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            use_fp16=use_fp16,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=residual.dtype,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


def _mhc_post_tilelang_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def hc_head_fused_kernel_tilelang(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """Apply the fused hc_head kernel and preserve the activation dtype."""
    activation_dtype = _require_mhc_activation_dtype(hs_flat)
    num_tokens, hc_mult, hidden_size = hs_flat.shape
    out = torch.empty(
        num_tokens, hidden_size, dtype=activation_dtype, device=hs_flat.device
    )
    if num_tokens == 0:
        return out
    from vllm.model_executor.kernels.mhc.tilelang_kernels import hc_head_fuse_tilelang

    hc_head_fuse_tilelang(
        hs_flat,
        fn,
        hc_scale,
        hc_base,
        out,
        hidden_size,
        rms_eps,
        hc_eps,
        hc_mult,
        use_fp16=activation_dtype == torch.float16,
    )
    return out


def _hc_head_fused_kernel_tilelang_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = hs_flat.shape
    return torch.empty(
        num_tokens, hidden_size, dtype=hs_flat.dtype, device=hs_flat.device
    )


direct_register_custom_op(
    op_name="mhc_pre_tilelang",
    op_func=mhc_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_pre_tilelang_fake,
)
direct_register_custom_op(
    op_name="mhc_post_tilelang",
    op_func=mhc_post_tilelang,
    mutates_args=[],
    fake_impl=_mhc_post_tilelang_fake,
)

direct_register_custom_op(
    op_name="mhc_fused_post_pre_tilelang",
    op_func=mhc_fused_post_pre_tilelang,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_tilelang_fake,
)

direct_register_custom_op(
    op_name="hc_head_fused_kernel_tilelang",
    op_func=hc_head_fused_kernel_tilelang,
    mutates_args=[],
    fake_impl=_hc_head_fused_kernel_tilelang_fake,
)
