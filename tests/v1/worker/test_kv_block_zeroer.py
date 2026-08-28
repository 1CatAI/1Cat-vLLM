# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KVBlockZeroer must zero the right blocks under a block-major interleaved KV
layout (n_segs segments one cell apart, so the block stride is
n_segs * page_size_el). Build such a layout directly and check that zeroing
block 1 clears only block 1's cells; the old single-page_size_el stride zeros
the wrong cells, so this fails before the fix and passes after.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.worker.utils import (
    KVBlockZeroer,
    _infer_segment_block_strides,
    _resolve_zeroer_kernel_layout,
)

CELL_ELS = 8  # int32 elements zeroed per (block, segment)
CELL_BYTES = CELL_ELS * 4
N_LAYERS = 3  # -> n_segs = 3 interleaved segments
N_BLOCKS = 4


class _BlockFirstAttentionBackend:
    @staticmethod
    def get_kv_cache_block_dim(*args, **kwargs) -> int:
        return 0


@pytest.mark.parametrize(
    ("spec", "group_kernel_bs", "expected"),
    [
        (
            FullAttentionSpec(
                block_size=784,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
            ),
            16,
            (16, 49),
        ),
        (
            MLAAttentionSpec(
                block_size=784,
                num_kv_heads=1,
                head_size=128,
                dtype=torch.bfloat16,
                compress_ratio=8,
            ),
            16,
            (98, 1),
        ),
    ],
)
@pytest.mark.skip_global_cleanup
def test_resolve_zeroer_kernel_layout(
    spec: FullAttentionSpec,
    group_kernel_bs: int,
    expected: tuple[int, int],
) -> None:
    assert _resolve_zeroer_kernel_layout(spec, group_kernel_bs) == expected


@pytest.mark.skip_global_cleanup
def test_resolve_zeroer_kernel_layout_rejects_nondivisible_size() -> None:
    spec = FullAttentionSpec(
        block_size=784,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="must be divisible"):
        _resolve_zeroer_kernel_layout(spec, 30)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_block_ids_compressed_attention_storage_page() -> None:
    device = torch.device("cuda")
    num_blocks = 8
    spec = MLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.bfloat16,
        compress_ratio=2,
    )
    kv = torch.ones(
        (num_blocks, spec.storage_block_size, 1, spec.head_size),
        dtype=spec.dtype,
        device=device,
    )
    layer_name = "qsa.compressed"
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=[layer_name],
        backend=_BlockFirstAttentionBackend,
    )
    static_forward_context = {layer_name: SimpleNamespace(kv_cache=kv)}

    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        [group],
        kernel_block_sizes=[2],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_forward_context,
    )

    target = 1
    zeroer.zero_block_ids([target])
    torch.accelerator.synchronize()

    pages = kv.view(num_blocks, -1)
    zeroed = {index for index in range(num_blocks) if bool((pages[index] == 0).all())}
    assert zeroed == {target}


@pytest.mark.parametrize(
    ("seg_addrs", "page_size_el", "expected"),
    [
        ([0x1000], 8, [8]),
        ([0x1000, 0x2000], 8, [8, 8]),
        ([0x1000, 0x1020, 0x1040], 8, [24, 24, 24]),
        # Two independent two-segment interleaved pools. Input order differs
        # from address order to prove strides stay bound to their segments.
        ([0x3020, 0x1000, 0x3000, 0x1020], 8, [16, 16, 16, 16]),
    ],
)
def test_infer_segment_block_strides(
    seg_addrs: list[int],
    page_size_el: int,
    expected: list[int],
) -> None:
    assert _infer_segment_block_strides(seg_addrs, page_size_el) == expected


def test_infer_segment_block_strides_rejects_duplicate_addresses() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        _infer_segment_block_strides([0x1000, 0x1000], 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_block_ids_block_major_interleaved() -> None:
    device = torch.device("cuda")

    # Block-major interleaved buffer: cell k holds (block B, layer i) where
    # k == B * N_LAYERS + i. Fill with a sentinel so any write is visible.
    flat = torch.ones(N_BLOCKS * N_LAYERS * CELL_ELS, dtype=torch.int32, device=device)

    # One mamba spec whose page is exactly one cell; page_size_padded lets us
    # pin page_size_bytes without depending on real state shapes.
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.int32,),
        page_size_padded=CELL_BYTES,
    )
    layer_names = [f"mamba.{i}" for i in range(N_LAYERS)]
    group = SimpleNamespace(
        kv_cache_spec=spec,
        kv_cache_group_id=0,
        layer_names=layer_names,
        backend=None,  # unused on the mamba path
    )
    # Layer i's state view starts one cell into the buffer, so the segment base
    # addresses are exactly one cell apart -> the interleaved layout.
    static_forward_context = {
        name: SimpleNamespace(kv_cache=[flat[i * CELL_ELS :]])
        for i, name in enumerate(layer_names)
    }

    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        [group],
        kernel_block_sizes=[spec.block_size],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_forward_context,
    )

    # Zero block 1 (block 0 is correct under both old and new strides, so it
    # does not discriminate; block 1 does).
    target = 1
    zeroer.zero_block_ids([target])
    torch.accelerator.synchronize()

    cells = flat.view(N_BLOCKS * N_LAYERS, CELL_ELS)
    zeroed = {int(k) for k in range(cells.shape[0]) if bool((cells[k] == 0).all())}
    expected = {target * N_LAYERS + i for i in range(N_LAYERS)}
    assert zeroed == expected, (
        f"expected only block {target}'s cells {sorted(expected)} to be "
        f"zeroed, got {sorted(zeroed)}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_block_ids_multiple_interleaved_pools() -> None:
    device = torch.device("cuda")
    layers_per_pool = 2
    buffers = [
        torch.ones(
            N_BLOCKS * layers_per_pool * CELL_ELS,
            dtype=torch.int32,
            device=device,
        )
        for _ in range(2)
    ]
    spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.int32,),
        page_size_padded=CELL_BYTES,
    )
    groups = []
    static_forward_context = {}
    for pool_index, buffer in enumerate(buffers):
        layer_names = [
            f"mamba.{pool_index}.{layer_index}"
            for layer_index in range(layers_per_pool)
        ]
        groups.append(
            SimpleNamespace(
                kv_cache_spec=spec,
                kv_cache_group_id=pool_index,
                layer_names=layer_names,
                backend=None,
            )
        )
        static_forward_context.update(
            {
                name: SimpleNamespace(kv_cache=[buffer[index * CELL_ELS :]])
                for index, name in enumerate(layer_names)
            }
        )

    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        groups,
        kernel_block_sizes=[spec.block_size] * len(groups),
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_forward_context,
    )

    target = 1
    zeroer.zero_block_ids([target])
    torch.accelerator.synchronize()

    expected = {target * layers_per_pool + index for index in range(layers_per_pool)}
    for buffer in buffers:
        cells = buffer.view(N_BLOCKS * layers_per_pool, CELL_ELS)
        zeroed = {
            index for index in range(cells.shape[0]) if bool((cells[index] == 0).all())
        }
        assert zeroed == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zero_block_ids_nonuniform_page_sizes() -> None:
    device = torch.device("cuda")
    page_sizes = [8, 16]
    buffers = [
        torch.ones(N_BLOCKS * page_size, dtype=torch.int32, device=device)
        for page_size in page_sizes
    ]
    groups = []
    static_forward_context = {}
    for index, (page_size, buffer) in enumerate(zip(page_sizes, buffers)):
        spec = MambaSpec(
            block_size=16,
            shapes=((1,),),
            dtypes=(torch.int32,),
            page_size_padded=page_size * 4,
        )
        layer_name = f"mamba.nonuniform.{index}"
        groups.append(
            SimpleNamespace(
                kv_cache_spec=spec,
                kv_cache_group_id=index,
                layer_names=[layer_name],
                backend=None,
            )
        )
        static_forward_context[layer_name] = SimpleNamespace(kv_cache=[buffer])

    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer.init_meta(
        groups,
        kernel_block_sizes=[16, 16],
        cache_dtype="auto",
        runner_only_attn_layers=set(),
        static_forward_context=static_forward_context,
    )

    target = 2
    zeroer.zero_block_ids([target])
    torch.accelerator.synchronize()

    for page_size, buffer in zip(page_sizes, buffers):
        pages = buffer.view(N_BLOCKS, page_size)
        zeroed = {index for index in range(N_BLOCKS) if bool((pages[index] == 0).all())}
        assert zeroed == {target}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_zeroer_supports_heterogeneous_page_sizes() -> None:
    device = torch.device("cuda")
    small = torch.ones((3, 8), dtype=torch.int32, device=device)
    large = torch.ones((3, 20), dtype=torch.int32, device=device)
    zeroer = KVBlockZeroer(device, pin_memory=False)
    zeroer._id_cap = 8
    zeroer._ids_pinned = torch.empty(8, dtype=torch.int64)
    zeroer._ids_gpu = torch.empty(8, dtype=torch.int64, device=device)
    zeroer._meta = (
        torch.tensor(
            [small.data_ptr(), large.data_ptr()], dtype=torch.uint64, device=device
        ),
        torch.tensor([8, 20], dtype=torch.int64, device=device),
        torch.tensor([8, 20], dtype=torch.int64, device=device),
        3,
        8,
        2,
    )

    zeroer.zero_block_ids([1])
    torch.accelerator.synchronize()

    torch.testing.assert_close(small[0], torch.ones_like(small[0]))
    torch.testing.assert_close(small[1], torch.zeros_like(small[1]))
    torch.testing.assert_close(small[2], torch.ones_like(small[2]))
    torch.testing.assert_close(large[0], torch.ones_like(large[0]))
    torch.testing.assert_close(large[1], torch.zeros_like(large[1]))
    torch.testing.assert_close(large[2], torch.ones_like(large[2]))
