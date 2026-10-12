# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Borrowed KV views. Allocation and lifetime stay with the cache owner.

All strides/offsets in the layout contract are bytes. Tensor views expose the
element strides expected by existing kernels. Binding never writes storage,
including after prefix restore, offload or AOT address rebinding.
"""

from dataclasses import dataclass

import torch

from vllm.v1.attention.kv_codecs import KVCodec


def _align16(size: int) -> int:
    return (size + 15) // 16 * 16


@dataclass(frozen=True)
class KVPageLayout:
    """Planar token/head cache inside one physical kernel page."""

    block_size: int
    num_heads: int
    head_size: int
    head_size_v: int
    element_bytes: int = 1

    def __post_init__(self) -> None:
        if (
            min(
                self.block_size,
                self.num_heads,
                self.head_size,
                self.head_size_v,
                self.element_bytes,
            )
            <= 0
        ):
            raise ValueError("KV page dimensions must be positive")

    @property
    def value_offset(self) -> int:
        return _align16(
            self.block_size * self.num_heads * self.head_size * self.element_bytes
        )

    @property
    def key_scale_offset(self) -> int:
        return self.value_offset + _align16(
            self.block_size * self.num_heads * self.head_size_v * self.element_bytes
        )

    @property
    def value_scale_offset(self) -> int:
        return self.key_scale_offset + _align16(self.block_size * self.num_heads * 4)

    @property
    def page_size_bytes(self) -> int:
        return self.value_scale_offset + _align16(self.block_size * self.num_heads * 4)

    def scheduler_page_stride(self, padded_bytes: int | None) -> int:
        """Honor the allocator's exact page budget, including hybrid padding.

        Subdivision must also satisfy ``bind_planar_kv_cache``. Changing an
        allocator-selected common page size here would break hybrid ownership.
        """
        if padded_bytes is None:
            return self.page_size_bytes
        if padded_bytes < self.page_size_bytes:
            raise ValueError("Padded KV page is smaller than its payload and scales")
        if padded_bytes % 16:
            raise ValueError("Planar KV scheduler pages must be sixteen-byte aligned")
        return padded_bytes


@dataclass(frozen=True)
class KVPageMapping:
    """Logical subpage IDs; payload/scale views retain physical owner strides."""

    owner_page_span: int
    subpage_tokens: int
    owner_tokens: int

    @classmethod
    def bind_payload(
        cls, key: torch.Tensor, value: torch.Tensor, subpage_tokens: int
    ) -> "KVPageMapping":
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.dtype != value.dtype
            or key.device != value.device
            or subpage_tokens <= 0
            or key.shape[1] % subpage_tokens
        ):
            raise ValueError("Invalid KV subpage geometry")
        row_elements = key.shape[2] * key.shape[3]
        if any(
            payload.stride(3) != 1
            or payload.stride(2) != payload.shape[3]
            or payload.stride(1) != row_elements
            or payload.stride(0) * payload.element_size() % 16
            or payload.stride(0) < key.shape[1] * row_elements
            for payload in (key, value)
        ) or key.stride(0) != value.stride(0):
            raise ValueError("KV subpages require aligned contiguous payload rows")
        return cls(key.shape[1] // subpage_tokens, subpage_tokens, key.shape[1])


@dataclass(frozen=True)
class MLAPageLayout:
    """Grouped latent payload with a separate scale plane and optional RoPE."""

    latent_dim: int
    rope_dim: int = 0
    group_size: int = 64
    scale_bytes: int = 1
    scale_alignment: int = 8

    def __post_init__(self) -> None:
        if (
            self.latent_dim <= 0
            or self.group_size <= 0
            or self.latent_dim % self.group_size
            or self.rope_dim < 0
            or self.scale_bytes not in (1, 4)
            or self.scale_alignment <= 0
        ):
            raise ValueError("Invalid grouped MLA page layout")

    @property
    def groups(self) -> int:
        return self.latent_dim // self.group_size

    @property
    def scale_slots(self) -> int:
        return (
            (self.groups + self.scale_alignment - 1)
            // self.scale_alignment
            * self.scale_alignment
        )

    @property
    def data_row_bytes(self) -> int:
        return self.latent_dim + 2 * self.rope_dim

    @property
    def slot_bytes(self) -> int:
        return self.data_row_bytes + self.scale_slots * self.scale_bytes

    def page_size_bytes(self, tokens: int) -> int:
        return tokens * self.slot_bytes


@dataclass(frozen=True)
class KVCacheView:
    codec: KVCodec
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None
    value_scale: torch.Tensor | None
    page_stride_bytes: int
    page_size_bytes: int
    unquantized: tuple[tuple[str, torch.Tensor], ...] = ()

    def unquantized_field(self, name: str) -> torch.Tensor:
        for field, tensor in self.unquantized:
            if field == name:
                return tensor
        raise ValueError(f"Cache view has no unquantized field {name!r}")

    @classmethod
    def bind_mla(
        cls,
        cache: torch.Tensor,
        codec: KVCodec,
        latent_dim: int,
        rope_dim: int = 0,
        rope_dtype: torch.dtype = torch.bfloat16,
        *,
        page_tokens: int | None = None,
    ) -> "KVCacheView":
        """Borrow grouped latent data, scale plane and optional unquantized RoPE.

        The carrier reserves physical page bytes; its token stride is not the
        payload stride. Eight-slot scale alignment preserves the legacy G64
        formats' padding group, with FP32 scales for INT8.
        """
        row_scaled = codec.representation == "mla_concat"
        group_size = latent_dim if row_scaled else codec.group_size
        valid_codec = (
            codec.native_id == 3 and codec.layout == "mla_row_planar" and not rope_dim
            if row_scaled
            else codec.representation in ("mla_latent", "mla_mixed")
            and codec.layout == "group_planar"
            and codec.group_size == 64
        )
        if (
            not valid_codec
            or latent_dim <= 0
            or latent_dim % group_size
            or rope_dim < 0
            or rope_dtype not in (torch.float16, torch.bfloat16)
            or codec.unquantized_fields != (("rope",) if rope_dim else ())
        ):
            raise ValueError("Invalid MLA grouped representation")
        if (codec.scale_encoding, codec.scale_dtype) not in (
            ("fp32", torch.float32),
            ("ue8m0", torch.uint8),
        ):
            raise ValueError("Unsupported MLA scale encoding")
        layout = MLAPageLayout(
            latent_dim,
            rope_dim,
            group_size,
            4 if codec.scale_encoding == "fp32" else 1,
            1 if row_scaled else 8,
        )
        if cache.ndim == 2:
            if page_tokens is None or page_tokens <= 0 or cache.stride(1) != 1:
                raise ValueError("Flattened MLA pages need an explicit token capacity")
            if cache.shape[1] < layout.page_size_bytes(page_tokens):
                raise ValueError("Flattened MLA page does not reserve payload/scales")
            cache = cache.as_strided(
                (cache.shape[0], page_tokens, layout.slot_bytes),
                (cache.stride(0), layout.slot_bytes, 1),
            )
        if (
            cache.ndim != 3
            or cache.dtype != codec.storage_dtype
            or cache.stride(-1) != 1
            or cache.stride(1) != cache.shape[2]
            or (page_tokens is not None and page_tokens != cache.shape[1])
        ):
            raise ValueError("MLA carrier requires block-major byte pages")
        pages, tokens, capacity = cache.shape
        groups, scale_slots = layout.groups, layout.scale_slots
        scale_bytes, row_bytes = layout.scale_bytes, layout.data_row_bytes
        page_size = layout.page_size_bytes(tokens)
        stride = cache.stride(0)
        base = cache.storage_offset()
        if (
            capacity < layout.slot_bytes
            or stride < page_size
            or base % scale_bytes
            or stride % scale_bytes
            or (rope_dim and ((base + latent_dim) % 2 or stride % 2))
        ):
            raise ValueError("MLA carrier does not reserve aligned data and scales")
        if pages and base + (pages - 1) * stride + page_size > (
            cache.untyped_storage().nbytes()
        ):
            raise ValueError("MLA regions exceed the allocation")
        key = cache.as_strided(
            (pages, tokens, 1, latent_dim), (stride, row_bytes, latent_dim, 1)
        )
        scales = torch.empty(0, dtype=codec.scale_dtype, device=cache.device).set_(
            cache.untyped_storage(),
            (base + tokens * row_bytes) // scale_bytes,
            (pages, tokens, groups),
            (stride // scale_bytes, scale_slots, 1),
        )
        fields: tuple[tuple[str, torch.Tensor], ...] = ()
        if rope_dim:
            rope = torch.empty(0, dtype=rope_dtype, device=cache.device).set_(
                cache.untyped_storage(),
                (base + latent_dim) // 2,
                (pages, tokens, 1, rope_dim),
                (stride // 2, row_bytes // 2, rope_dim, 1),
            )
            fields = (("rope", rope),)
        return cls(codec, key, key, scales, scales, stride, page_size, fields)

    @classmethod
    def bind_mla_concat(
        cls,
        cache: torch.Tensor,
        codec: KVCodec,
        head_size: int,
        scale: torch.Tensor,
    ) -> "KVCacheView":
        """Concatenated MLA shares a row scale across latent and quantized RoPE."""
        if codec.representation != "mla_concat":
            raise ValueError("Concatenated MLA requires its representation codec")
        if codec.native_id == 3:
            return cls.bind_mla(cache, codec, head_size)
        if cache.ndim != 3 or cache.shape[-1] != head_size or cache.stride(-1) != 1:
            raise ValueError("Concatenated MLA requires block-major payload pages")
        if cache.stride(1) != head_size or cache.stride(0) < cache.shape[1] * head_size:
            raise ValueError("Concatenated MLA payload rows must not overlap")
        if codec.native_id == 1:
            if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
                raise ValueError("E4M3 MLA requires E4M3 payload bytes")
            if (
                scale.dtype != torch.float32
                or scale.numel() != 1
                or scale.device != cache.device
            ):
                raise ValueError("E4M3 MLA requires its FP32 tensor scale")
            key = cache.view(torch.uint8).unsqueeze(2)
            scales = scale
        else:
            if cache.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("Dense MLA requires FP16 or BF16 payloads")
            key, scales = cache.unsqueeze(2), None
        size = cache.element_size()
        return cls(
            codec,
            key,
            key,
            scales,
            scales,
            cache.stride(0) * size,
            cache.shape[1] * head_size * size,
        )

    def subpage_mapping(self, page_tokens: int) -> KVPageMapping:
        return KVPageMapping.bind_payload(self.key, self.value, page_tokens)

    @property
    def scale_kwargs(self) -> dict[str, torch.Tensor]:
        """Explicit native row-scale arguments; tensor-scale codecs use none."""
        if not self.codec.per_token_head:
            return {}
        if self.key_scale is None or self.value_scale is None:
            raise ValueError("Per-token KV view is missing its scale planes")
        return {"k_scale_cache": self.key_scale, "v_scale_cache": self.value_scale}

    @property
    def inter_page_gap_bytes(self) -> int:
        """Bytes between this member's pages, including any packed peers."""
        return self.page_stride_bytes - self.page_size_bytes

    @classmethod
    def bind_history(
        cls, cache: torch.Tensor, scales: torch.Tensor, codec: KVCodec
    ) -> "KVCacheView":
        """Borrow history payload and its independently owned FP32 scale plane."""
        if codec.representation != "history" or codec.layout != "history_planes":
            raise ValueError("History binding requires the history representation")
        if (
            cache.ndim != 5
            or cache.shape[1] != 2
            or cache.shape[3] != 1
            or cache.dtype != codec.storage_dtype
            or not cache.is_contiguous()
            or scales.dtype != torch.float32
            or not scales.is_contiguous()
            or scales.device != cache.device
        ):
            raise ValueError("Invalid history payload or scale layout")
        pages, _, tokens, _, dim = cache.shape
        expected = (pages * tokens if codec.quantized else 1, 2)
        if scales.shape != expected:
            raise ValueError("History scales must match every physical token")
        key, value = cache.unbind(1)
        if codec.quantized:
            planes = scales.view(pages, tokens, 2)
            key_scale = planes[..., 0, None]
            value_scale = planes[..., 1, None]
        else:
            key_scale = value_scale = None
        occupied = 2 * tokens * dim * cache.element_size()
        return cls(
            codec,
            key,
            value,
            key_scale,
            value_scale,
            cache.stride(0) * cache.element_size(),
            occupied,
        )

    @classmethod
    def bind_indexer(
        cls, cache: torch.Tensor, codec: KVCodec, head_size: int
    ) -> "KVCacheView":
        """Borrow block-major payload/group scales, including padded pages.

        The indexer has one key plane. Its value and scale aliases preserve
        the public view contract without allocating or duplicating that plane.
        """
        if codec.representation != "indexer" or codec.layout != "group_planar":
            raise ValueError("Indexer binding requires its group representation")
        if (
            cache.ndim != 3
            or cache.dtype not in (torch.uint8, torch.int8)
            or cache.stride(-1) != 1
            or head_size <= 0
            or codec.group_size <= 0
            or head_size % codec.group_size
        ):
            raise ValueError("Expected byte indexer pages with complete groups")
        pages, tokens, capacity = cache.shape
        groups = head_size // codec.group_size
        page_size = tokens * (head_size + groups * 4)
        page_stride = cache.stride(0)
        base = cache.storage_offset()
        if capacity < head_size + groups * 4 or page_stride < page_size:
            raise ValueError("Indexer carrier does not reserve payload and scales")
        if base % 4 or page_stride % 4:
            raise ValueError("Indexer scale addresses require four-byte alignment")
        if (
            pages
            and base + (pages - 1) * page_stride + page_size
            > cache.untyped_storage().nbytes()
        ):
            raise ValueError("Indexer regions exceed the allocation")
        key = cache.view(codec.storage_dtype).as_strided(
            (pages, tokens, 1, head_size),
            (page_stride, head_size, head_size, 1),
        )
        scales = torch.empty(0, dtype=torch.float32, device=cache.device).set_(
            cache.untyped_storage(),
            (base + tokens * head_size) // 4,
            (pages, tokens, groups),
            (page_stride // 4, groups, 1),
        )
        return cls(codec, key, key, scales, scales, page_stride, page_size)

    @classmethod
    def bind(cls, cache: torch.Tensor, codec: KVCodec, head_size: int) -> "KVCacheView":
        """Bind a runner-owned ``[page, 2, token, head, storage]`` carrier.

        For planar codecs the carrier's trailing dimensions reserve bytes;
        only this view describes their interpretation. Dense and legacy inline
        layouts use the carrier's actual strides, including HND and page padding.
        """
        if codec.representation != "ordinary":
            raise ValueError("Ordinary KV binding requires an ordinary representation")
        if cache.ndim != 5 or cache.shape[1] != 2 or cache.stride(-1) != 1:
            raise ValueError("Expected a paged KV carrier with contiguous head storage")
        if cache.dtype != codec.storage_dtype:
            raise ValueError(f"{codec.name} requires {codec.storage_dtype} storage")
        n, _, b, h, stored_dim = cache.shape
        item = cache.element_size()
        page_stride = cache.stride(0) * item
        base_offset = cache.storage_offset() * item
        if head_size <= 0 or stored_dim < head_size:
            raise ValueError("Invalid logical KV head dimension")

        def scale_view(offset: int, strides: tuple[int, int, int]) -> torch.Tensor:
            if any(x % 4 for x in (offset, *strides)):
                raise ValueError("FP32 KV scales require four-byte aligned addresses")
            return torch.empty(0, dtype=torch.float32, device=cache.device).set_(
                cache.untyped_storage(),
                offset // 4,
                (n, b, h),
                tuple(x // 4 for x in strides),
            )

        if codec.layout == "planar_scales":
            layout = KVPageLayout(b, h, head_size, head_size, item)
            if base_offset % 16 or page_stride % 16:
                raise ValueError("Planar KV pages require sixteen-byte alignment")
            if page_stride < layout.page_size_bytes:
                raise ValueError("KV carrier does not reserve a complete physical page")
            if (
                n
                and base_offset + (n - 1) * page_stride + layout.page_size_bytes
                > cache.untyped_storage().nbytes()
            ):
                raise ValueError("Planar KV regions exceed the allocation")
            strides = (cache.stride(0), h * head_size, head_size, 1)
            key = cache.as_strided((n, b, h, head_size), strides)
            value = cache.as_strided(
                (n, b, h, head_size),
                strides,
                (base_offset + layout.value_offset) // item,
            )
            scale_strides = (page_stride, h * 4, 4)
            return cls(
                codec,
                key,
                value,
                scale_view(base_offset + layout.key_scale_offset, scale_strides),
                scale_view(base_offset + layout.value_scale_offset, scale_strides),
                page_stride,
                layout.page_size_bytes,
            )

        key, value = cache.unbind(1)
        if codec.layout == "inline_scale":
            if stored_dim != head_size + 4 // item:
                raise ValueError(
                    "Inline KV carrier must reserve one FP32 scale per head"
                )
            inline_scale_strides = (
                key.stride(0) * item,
                key.stride(1) * item,
                key.stride(2) * item,
            )
            return cls(
                codec,
                key[..., :head_size],
                value[..., :head_size],
                scale_view(base_offset + head_size * item, inline_scale_strides),
                scale_view(
                    value.storage_offset() * item + head_size * item,
                    inline_scale_strides,
                ),
                page_stride,
                2 * b * h * stored_dim * item,
            )
        if stored_dim != head_size:
            raise ValueError("Dense KV head dimension does not match the codec view")
        return cls(
            codec, key, value, None, None, page_stride, 2 * b * h * head_size * item
        )


def bind_planar_kv_cache(
    raw: torch.Tensor,
    *,
    num_blocks: int,
    scheduler_block_size: int,
    kernel_block_size: int,
    num_heads: int,
    head_size: int,
    page_size_bytes: int,
    member_index: int = 0,
    member_count: int = 1,
) -> torch.Tensor:
    """Make a carrier over existing allocation; no device allocation or copy.

    Packed members interleave at kernel-page granularity, matching the existing
    V2 owner. Each scheduler block still contains all bytes for every member.
    """
    if scheduler_block_size % kernel_block_size or kernel_block_size % 16:
        raise ValueError("Planar KV kernel pages must divide scheduler blocks in 16s")
    subdivisions = scheduler_block_size // kernel_block_size
    if page_size_bytes % (16 * subdivisions):
        raise ValueError("Padded KV page cannot be split into aligned kernel pages")
    if not 0 <= member_index < member_count:
        raise ValueError("Invalid packed KV member")
    layout = KVPageLayout(kernel_block_size, num_heads, head_size, head_size)
    kernel_bytes = page_size_bytes // subdivisions
    if kernel_bytes < layout.page_size_bytes:
        raise ValueError("KV allocation is smaller than the declared codec layout")
    shape = (num_blocks * subdivisions, 2, kernel_block_size, num_heads, head_size + 4)
    row = head_size + 4
    stride = (
        kernel_bytes * member_count,
        kernel_block_size * num_heads * row,
        num_heads * row,
        row,
        1,
    )
    return raw.view(torch.int8).as_strided(
        shape,
        stride,
        raw.storage_offset() * raw.element_size() + kernel_bytes * member_index,
    )
