# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Qwen3.5 MoE mapping and stacked expert transport.

Name rules follow vllm-gguf-plugin qwen3_5.py at
e2b8ad532b8b5ea175100202c30430c1d2b5e6a8 (Apache-2.0). GDN transformations
are inherited from the dense adapter. Expert byte transport is shared with
Qwen4Exp; quantized tensors retain their source format until kernel admission.
"""

import gguf
import numpy as np
import torch

from vllm.model_executor.layers.quantization.gguf_repack import q2_0_to_q4_1
from vllm.transformers_utils.gguf_tensor_reader import dequantize

from .qwen35 import _LAYERS, Qwen35Adapter

_EXPERTS = {
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
    "ffn_gate_exps.weight": "mlp.experts.gate_proj.weight",
    "ffn_up_exps.weight": "mlp.experts.up_proj.weight",
    "ffn_down_exps.weight": "mlp.experts.down_proj.weight",
}


class Qwen35MoeAdapter(Qwen35Adapter):
    native_expert_storage = True
    architecture_label = "Qwen3.5 MoE"
    layer_names = {**_LAYERS, **_EXPERTS}

    @staticmethod
    def is_linear(name):
        return Qwen35Adapter.is_linear(name) and not name.endswith(
            (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")
        )

    def restore(self, name, weight):
        if name.endswith(".mlp.shared_expert_gate.weight") and weight.ndim == 1:
            return weight.unsqueeze(0)
        return super().restore(name, weight)

    def needs_dense_fallback(self, name, tensor):
        if ".mlp.experts." in name:
            return False
        return super().needs_dense_fallback(name, tensor)

    @staticmethod
    def _dense(tensor, dtype):
        if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            raw = torch.from_numpy(tensor.data.view(np.uint16).copy())
            value = raw.view(torch.bfloat16).float()
        elif tensor.tensor_type in (
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.F32,
        ):
            value = torch.from_numpy(tensor.data.copy())
        else:
            value = torch.from_numpy(dequantize(tensor.data, tensor.tensor_type))
        converted = value.to(dtype)
        if torch.any(torch.isfinite(value) & ~torch.isfinite(converted)):
            raise ValueError("GGUF projection overflows target dtype")
        return converted

    def weights(self, tensors, name_map, dtype):
        regular = {
            raw: name for raw, name in name_map.items() if ".mlp.experts." not in name
        }
        experts = {
            raw: name for raw, name in name_map.items() if ".mlp.experts." in name
        }
        yield from super().weights(tensors, regular, dtype)
        yield from self._expert_weights(tensors, experts, dtype)

    def _expert_weights(self, tensors, name_map, dtype):
        for raw, name in name_map.items():
            tensor = tensors[raw]
            if len(tensor.shape) != 3 or tensor.shape[2] != self.config.num_experts:
                raise ValueError(f"Invalid stacked GGUF expert shape: {raw}")
            prefix, projection, _ = name.rsplit(".", 2)
            floating = tensor.tensor_type in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.F32,
                gguf.GGMLQuantizationType.BF16,
            )
            repack = (
                int(tensor.tensor_type) == 42
                and not getattr(self, "canonical_expert_storage", False)
                and projection == "down_proj"
                and int(tensor.shape[0]) % (64 * self.tp_size) != 0
            )
            storage_type = (
                int(
                    {
                        torch.float16: gguf.GGMLQuantizationType.F16,
                        torch.float32: gguf.GGMLQuantizationType.F32,
                        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
                    }[dtype]
                )
                if floating
                else int(gguf.GGMLQuantizationType.Q4_1)
                if repack
                else int(tensor.tensor_type)
            )
            if repack:
                self.fallback_reasons[prefix + ".down_proj"] = (
                    "lossless_Q2_0_to_Q4_1_before_tp:"
                    f"local_K={int(tensor.shape[0]) // self.tp_size},block=32"
                )
            # Keep one type per logical gate/up/down projection. The expert
            # method handles local expert admission and TP storage boundaries.
            for expert in range(self.config.num_experts):
                module = f"{prefix}.{expert}.{projection}"
                yield module + ".qweight_type", torch.tensor(storage_type)
            packed = (
                self._dense(tensor, dtype)
                if floating
                else torch.from_numpy(tensor.data)
            )
            for expert in range(self.config.num_experts):
                weight = (
                    torch.from_numpy(q2_0_to_q4_1(tensor.data[expert]))
                    if repack
                    else packed[expert]
                )
                yield f"{prefix}.{expert}.{projection}.qweight", weight
