# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense GLM5Next mmproj inversion into the native GLM5Next vision tower.

The GGUF projector is ``glm5v``: temporal convolution is stored as two
spatial slices, and SwiGLU gate/up projections are stored separately. Packed
QKV and the spatial downsample convolution retain their PyTorch order.
"""

from collections.abc import Iterator, Mapping

import gguf
import torch

from .common import decode_dense_tensor, require_tensor_shape


class Glm5NextVisionGGUF:
    def __init__(self, config, tensors: Mapping[str, gguf.ReaderTensor]):
        self.tensors = tensors
        self.name_map: dict[str, str] = {}
        self.raw_shapes: dict[str, tuple[int, ...]] = {}
        self.output_shapes: dict[str, tuple[int, ...]] = {}
        self.groups: dict[str, tuple[tuple[str, ...], int]] = {}
        for name in (
            "hidden_size",
            "intermediate_size",
            "projection_intermediate_size",
            "out_hidden_size",
            "depth",
            "num_heads",
            "in_channels",
            "patch_size",
            "spatial_merge_size",
        ):
            value = getattr(config, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"GLM5Next vision requires positive integer {name}")
        if config.temporal_patch_size != 2:
            raise ValueError("GLM5Next mmproj requires temporal_patch_size=2")
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        output = config.out_hidden_size
        projection = config.projection_intermediate_size
        if hidden % config.num_heads:
            raise ValueError("GLM5Next vision hidden_size must divide into num_heads")

        def add(raw, target, shape, output_shape=None):
            target = "visual." + target
            self.name_map[raw] = target
            self.raw_shapes[raw] = shape
            self.output_shapes[target] = output_shape or shape

        def gate_up(raw_prefix, target_prefix, size, input_size, *, bias):
            shapes: dict[str, tuple[int, ...]] = {"weight": (size, input_size)}
            if bias:
                shapes["bias"] = (size,)
            for suffix, shape in shapes.items():
                names = tuple(
                    raw_prefix + name + "." + suffix for name in ("gate", "up")
                )
                target = target_prefix + "gate_up_proj." + suffix
                for raw in names:
                    add(raw, target, shape, (2 * size, *shape[1:]))
                self.groups["visual." + target] = (names, 0)

        for index in range(config.depth):
            raw_prefix = f"v.blk.{index}."
            target_prefix = f"blocks.{index}."
            for raw, target, shape in (
                ("attn_qkv", "attn.qkv", (3 * hidden, hidden)),
                ("attn_out", "attn.proj", (hidden, hidden)),
                ("ffn_down", "mlp.down_proj", (hidden, intermediate)),
            ):
                add(
                    raw_prefix + raw + ".weight",
                    target_prefix + target + ".weight",
                    shape,
                )
                add(
                    raw_prefix + raw + ".bias",
                    target_prefix + target + ".bias",
                    shape[:1],
                )
            for raw, target, size in (
                ("ln1", "norm1", hidden),
                ("ln2", "norm2", hidden),
                ("attn_q_norm", "attn.q_norm", hidden // config.num_heads),
                ("attn_k_norm", "attn.k_norm", hidden // config.num_heads),
            ):
                add(
                    raw_prefix + raw + ".weight",
                    target_prefix + target + ".weight",
                    (size,),
                )
            gate_up(
                raw_prefix + "ffn_",
                target_prefix + "mlp.",
                intermediate,
                hidden,
                bias=True,
            )

        add("v.post_ln.weight", "post_layernorm.weight", (hidden,))
        add(
            "mm.patch_merger.weight",
            "downsample.weight",
            (output, hidden, config.spatial_merge_size, config.spatial_merge_size),
        )
        add("mm.patch_merger.bias", "downsample.bias", (output,))
        add("mm.model.fc.weight", "merger.proj.weight", (output, output))
        for suffix in ("weight", "bias"):
            add(
                "mm.post_norm." + suffix,
                "merger.post_projection_norm." + suffix,
                (output,),
            )
        add("mm.down.weight", "merger.down_proj.weight", (output, projection))
        gate_up("mm.", "merger.", projection, output, bias=False)

        add("v.patch_embd.bias", "patch_embed.proj.bias", (hidden,))
        patch_shape = (hidden, config.in_channels, config.patch_size, config.patch_size)
        patch_output = (*patch_shape[:2], 2, *patch_shape[2:])
        patch_names = ("v.patch_embd.weight", "v.patch_embd.weight.1")
        for raw in patch_names:
            add(raw, "patch_embed.proj.weight", patch_shape, patch_output)
        self.groups["visual.patch_embed.proj.weight"] = (patch_names, 2)

        missing = self.name_map.keys() - tensors.keys()
        extra = tensors.keys() - self.name_map.keys()
        if missing or extra:
            raise ValueError(
                f"Incomplete GLM5Next mmproj: missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        for raw, expected in self.raw_shapes.items():
            require_tensor_shape(tensors[raw], expected, raw)
            if tensors[raw].tensor_type not in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.BF16,
            ):
                raise ValueError(f"GLM5Next dense mmproj required: {raw}")

    @property
    def unquantized_modules(self) -> list[str]:
        return sorted(
            {
                name.removesuffix(".weight")
                for name in self.output_shapes
                if name.endswith(".weight")
            }
        )

    def weights(
        self,
        dtype: torch.dtype,
        name_map: Mapping[str, str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # Validate full structure in __init__, but read only selected payloads.
        # PP ranks without a tower can therefore skip this complete checkpoint.
        selected = self.name_map if name_map is None else name_map
        if any(self.name_map.get(raw) != target for raw, target in selected.items()):
            raise ValueError("GLM5Next filtered mapping contains unknown/changed names")
        for names, _ in self.groups.values():
            if selected.keys() & set(names) and not set(names) <= selected.keys():
                raise ValueError(
                    f"GLM5Next filtered mapping must retain complete group: {names}"
                )

        def dense(raw):
            return decode_dense_tensor(self.tensors[raw], dtype, raw)

        emitted = set()
        for raw, target in selected.items():
            if target in emitted:
                continue
            if target in self.groups:
                names, axis = self.groups[target]
                values = [dense(name) for name in names]
                # Temporal slices add an axis; SwiGLU concatenates existing rows.
                weight = (
                    torch.stack(values, dim=axis)
                    if axis == 2
                    else torch.cat(values, dim=axis)
                )
            else:
                weight = dense(raw)
            if tuple(weight.shape) != self.output_shapes[target]:
                raise ValueError(f"GLM5Next reconstructed shape mismatch: {target}")
            emitted.add(target)
            yield target, weight
