# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense Qwen3-VL and Qwen4Exp mmproj GGUF inverse adapter.

Oracle: local llama.cpp conversion/qwen3vl.py Qwen3VLVisionModel,
also inherited unchanged by Qwen4ExpVisionModel in qwen4exp.py.
Only dense F16/F32/BF16 tensors are admitted. No implicit quantization.
"""

from collections.abc import Iterator, Mapping

import gguf
import torch


class Qwen3VisionGGUF:
    def __init__(self, config, tensors: Mapping[str, gguf.ReaderTensor]):
        self.config = config
        self.tensors = tensors
        self.name_map: dict[str, str] = {}
        self.raw_shapes: dict[str, tuple[int, ...]] = {}
        self.output_shapes: dict[str, tuple[int, ...]] = {}
        hidden = config.hidden_size
        intermediate = config.intermediate_size
        merged = hidden * config.spatial_merge_size**2
        if config.temporal_patch_size != 2:
            raise ValueError("Qwen3-VL mmproj requires temporal_patch_size=2")

        def add(raw, target, shape, output_shape=None):
            self.name_map[raw] = "visual." + target
            self.raw_shapes[raw] = shape
            self.output_shapes["visual." + target] = output_shape or shape

        shape: tuple[int, ...]
        for index in range(config.depth):
            raw_prefix = f"v.blk.{index}."
            target_prefix = f"blocks.{index}."
            for raw, target, shape in (
                ("attn_qkv", "attn.qkv", (3 * hidden, hidden)),
                ("attn_out", "attn.proj", (hidden, hidden)),
                ("ffn_up", "mlp.linear_fc1", (intermediate, hidden)),
                ("ffn_down", "mlp.linear_fc2", (hidden, intermediate)),
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
            for raw, target in (("ln1", "norm1"), ("ln2", "norm2")):
                for suffix in ("weight", "bias"):
                    add(
                        raw_prefix + raw + "." + suffix,
                        target_prefix + target + "." + suffix,
                        (hidden,),
                    )

        for suffix in ("weight", "bias"):
            add("v.post_ln." + suffix, "merger.norm." + suffix, (hidden,))
        for raw, target, shape in (
            ("mm.0", "merger.linear_fc1", (merged, merged)),
            ("mm.2", "merger.linear_fc2", (config.out_hidden_size, merged)),
        ):
            add(raw + ".weight", target + ".weight", shape)
            add(raw + ".bias", target + ".bias", shape[:1])
        for merger_index, block_index in enumerate(config.deepstack_visual_indexes):
            for raw, target, shape in (
                ("norm", "norm", (merged,)),
                ("fc1", "linear_fc1", (merged, merged)),
                ("fc2", "linear_fc2", (config.out_hidden_size, merged)),
            ):
                prefix = f"v.deepstack.{block_index}.{raw}."
                target = f"deepstack_merger_list.{merger_index}.{target}."
                add(prefix + "weight", target + "weight", shape)
                add(prefix + "bias", target + "bias", shape[:1])

        add("v.patch_embd.bias", "patch_embed.proj.bias", (hidden,))
        patch_shape = (hidden, config.in_channels, config.patch_size, config.patch_size)
        patch_output_shape = (*patch_shape[:2], 2, *patch_shape[2:])
        for raw in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
            add(raw, "patch_embed.proj.weight", patch_shape, patch_output_shape)
        add(
            "v.position_embd.weight",
            "pos_embed.weight",
            (config.num_position_embeddings, hidden),
        )

        missing = self.name_map.keys() - tensors.keys()
        extra = tensors.keys() - self.name_map.keys()
        if missing or extra:
            raise ValueError(
                f"Incomplete Qwen3-VL mmproj: missing={sorted(missing)}, "
                f"unexpected={sorted(extra)}"
            )
        for raw, expected in self.raw_shapes.items():
            actual = tuple(map(int, tensors[raw].shape[::-1]))
            if actual != expected:
                raise ValueError(f"Qwen3-VL {raw}: shape {actual}, expected {expected}")
            if tensors[raw].tensor_type not in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.BF16,
            ):
                raise ValueError(f"Qwen3-VL dense mmproj required: {raw}")

    @property
    def unquantized_modules(self) -> list[str]:
        return sorted(
            {
                name.removesuffix(".weight")
                for name in self.name_map.values()
                if name.endswith(".weight")
            }
        )

    def weights(
        self,
        dtype: torch.dtype,
        name_map: Mapping[str, str] | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        # Full-file structure is validated once in __init__. Filtering only
        # controls payload loading (for PP ranks without a vision tower).
        selected = self.name_map if name_map is None else name_map
        if any(self.name_map.get(raw) != target for raw, target in selected.items()):
            raise ValueError("Qwen3-VL filtered mapping contains unknown/changed names")
        patch_names = {"v.patch_embd.weight", "v.patch_embd.weight.1"}
        if selected.keys() & patch_names and not patch_names <= selected.keys():
            raise ValueError("Qwen3-VL filtered mapping must retain both patch slices")

        def dense(raw):
            tensor = self.tensors[raw]
            array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
            # Own the output storage: GGUF arrays can be read-only mmaps and
            # downstream loaders may transform parameters after assignment.
            weight = torch.tensor(array, dtype=dtype)
            if not torch.isfinite(weight).all():
                raise ValueError(f"Qwen3-VL nonfinite or {dtype} overflow: {raw}")
            return weight

        for raw, target in selected.items():
            if raw == "v.patch_embd.weight.1":
                continue
            if raw == "v.patch_embd.weight":
                weight = torch.stack([dense(raw), dense(raw + ".1")], dim=2)
            else:
                weight = dense(raw)
            if tuple(weight.shape) != self.output_shapes[target]:
                raise ValueError(f"Qwen3-VL reconstructed shape mismatch: {target}")
            yield target, weight
