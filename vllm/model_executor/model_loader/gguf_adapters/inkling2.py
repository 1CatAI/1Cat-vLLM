# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 GGUF (llama.cpp / Unsloth) tensors -> official checkpoint tensors.

The GGUF route serves the same model as the official checkpoint, so every GGUF
tensor is yielded under its official name, shape and storage format, and
`Inkling2LoopForConditionalGeneration.load_weights` loads it unchanged:

* F32 / F16 / BF16 tensors keep their values (a few are transposed or reshaped
  to the official layout).
* Q8_0 dense projections and embeddings are dequantized (the model's dense
  layers run in FP16).
* NVFP4 expert banks are repacked exactly into the official NVFP4 layout
  (`vllm.models.inkling2.gguf_weights`), with the GGUF per-expert scale as the
  global scale. GGUF stores ``ffn_gate_up_exps`` as all gate rows then all up
  rows; the official ``w13`` interleaves them (gate_0, up_0, gate_1, ...).
* BF16 / Q8_0 expert banks (the layers the official checkpoint stores as
  MXFP8) are yielded as dense ``w13_weight`` / ``w2_weight``; the model keeps
  them in FP16, as it already does after dequantizing MXFP8.

``blk.N`` is physical layer N (GGUF ``block_count`` counts layer visits).
"""

from collections.abc import Iterator

import gguf
import numpy as np
import regex as re
import torch

from vllm.models.inkling2.gguf_weights import ggml_nvfp4_to_official
from vllm.transformers_utils.gguf_tensor_reader import dequantize

Q = gguf.GGMLQuantizationType
_FLOAT = (Q.F32, Q.F16, Q.BF16)
_P = "model.llm."

_GLOBALS = {
    "token_embd.weight": "model.llm.embed.weight",
    "token_embd_norm.weight": "model.llm.embed_norm.weight",
    "output.weight": "model.llm.unembed.weight",
    "output_norm.weight": "model.llm.norm.weight",
    "output_res_query.weight": "model.llm.attn_res.output_query",
    "loop_res_query.weight": "model.llm.middle_cycle_cross_pass_queries",
    "loop_kv_res_query.weight": "model.llm.middle_cycle_cross_pass_kv_queries",
}
# GGUF blk.N suffix -> official suffix under model.llm.layers.N.
_LAYER = {
    "attn_q.weight": "attn.wq_du.weight",
    "attn_gate.weight": "attn.gate_attn.weight",
    "attn_rel.weight": "attn.wr_du.weight",
    "attn_k.weight": "attn.wk_dv.weight",
    "attn_v.weight": "attn.wv_dv.weight",
    "attn_output.weight": "attn.wo_ud.weight",
    "attn_sink_proj.weight": "attn.attn_sink_proj.weight",
    "attn_q_clip.weight": "attn.q_l2_clip.scale",
    "attn_rel_clip.weight": "attn.rel_l2_clip.scale",
    "attn_rel_bias.weight": "attn.rel_logits_proj.proj",
    "attn_k_conv_proj.weight": "attn.k_sconv.weight_proj",
    "attn_v_conv_proj.weight": "attn.v_sconv.weight_proj",
    "attn_post_conv.weight": "attn_sconv.weight",
    "ffn_post_conv.weight": "mlp_sconv.weight",
    "attn_norm.weight": "attn_norm.weight",
    "attn_q_inp_norm.weight": "q_input_norm.weight",
    "attn_k_inp_norm.weight": "k_input_norm.weight",
    "attn_v_inp_norm.weight": "v_input_norm.weight",
    "ffn_norm.weight": "mlp_norm.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "exp_probs_b.bias": "mlp.gate.bias",
    "ffn_gate_up_exps.weight": "mlp.experts.w13_weight",
    "ffn_down_exps.weight": "mlp.experts.w2_weight",
    **{
        f"loop_{m}_{w}.weight": f"controllers.{m}.{w}_weight"
        for m in ("attn", "k", "v", "mlp")
        for w in ("down", "gain", "forget")
    },
}
# Expert global scales travel with their bank (see `weights`).
_EXPERT_SCALES = ("ffn_gate_up_exps.scale", "ffn_down_exps.scale")


def _float_tensor(t) -> torch.Tensor:
    """F32 / F16 / BF16 GGUF tensor in PyTorch (reversed ``ne``) order."""
    shape = tuple(int(x) for x in reversed(t.shape))
    if t.tensor_type == Q.BF16:
        raw = np.ascontiguousarray(t.data).view(np.uint16)
        return torch.from_numpy(raw.copy()).view(torch.bfloat16).reshape(shape)
    return torch.from_numpy(np.array(t.data)).reshape(shape)


def _interleave_gate_up(w: torch.Tensor) -> torch.Tensor:
    """GGUF ``[E, gate rows ; up rows, ...]`` -> official ``w13`` row order
    ``gate_0, up_0, gate_1, up_1, ...``."""
    half = w.shape[1] // 2
    return torch.stack((w[:, :half], w[:, half:]), dim=2).reshape(w.shape)


def _dense_tensor(t) -> torch.Tensor:
    if t.tensor_type in _FLOAT:
        return _float_tensor(t)
    shape = tuple(int(x) for x in reversed(t.shape))
    return torch.from_numpy(dequantize(t.data, t.tensor_type)).reshape(shape)


class Inkling2Adapter:
    architecture_label = "Inkling2"
    native_expert_storage = False
    language_model_prefix = ""
    packed_embeddings = False

    def __init__(self, config, tp_size: int = 1) -> None:
        self.config = config  # text config
        self.tp_size = tp_size
        self.fallback_reasons: dict[str, str] = {}
        self.local_layers = set(config.local_layer_ids)

    def build_name_map(self, tensors) -> dict[str, str]:
        result = {}
        n_layers = self.config.num_hidden_layers
        for raw in tensors:
            if raw in _GLOBALS:
                result[raw] = _GLOBALS[raw]
                continue
            m = re.fullmatch(r"blk\.(\d+)\.(.+)", raw)
            if m is None or int(m[1]) >= n_layers:
                raise ValueError(f"Unmapped Inkling2 GGUF tensor: {raw}")
            layer, suffix = int(m[1]), m[2]
            if suffix in _EXPERT_SCALES:
                result[raw] = f"{_P}layers.{layer}.mlp.experts.{suffix}"
            elif suffix in ("attn_k_conv.weight", "attn_v_conv.weight"):
                kv = suffix[5]
                tail = "weight" if layer in self.local_layers else "static_w"
                result[raw] = f"{_P}layers.{layer}.attn.{kv}_sconv.{tail}"
            elif suffix == "attn_res_query.weight":
                if layer == 0:
                    raise ValueError("Inkling2 layer 0 has no attention-input query")
                result[raw] = f"{_P}attn_res.attn_queries.{layer - 1}"
            elif suffix == "ffn_res_query.weight":
                result[raw] = f"{_P}attn_res.mlp_queries.{layer}"
            elif suffix in _LAYER:
                result[raw] = f"{_P}layers.{layer}.{_LAYER[suffix]}"
            else:
                raise ValueError(f"Unmapped Inkling2 GGUF tensor: {raw}")
        return result

    def is_linear(self, name: str) -> bool:
        return False  # dense layers are dequantized: the model runs them in FP16

    def needs_dense_fallback(self, name, tensor) -> bool:
        return False

    def linear_layouts(self, name_map) -> dict:
        return {}

    def weights(
        self, tensors, name_map: dict[str, str], dtype: torch.dtype
    ) -> Iterator[tuple[str, torch.Tensor]]:
        for raw, name in name_map.items():
            if raw.endswith(_EXPERT_SCALES):
                continue  # emitted with the expert bank
            t = tensors[raw]
            if name.endswith(("experts.w13_weight", "experts.w2_weight")):
                yield from self._experts(tensors, raw, name, t)
            else:
                yield name, self._layout(raw, name, _dense_tensor(t))

    @staticmethod
    def _layout(raw: str, name: str, w: torch.Tensor) -> torch.Tensor:
        if name.endswith(".rel_logits_proj.proj"):  # [rel_extent, d_rel] -> T
            return w.t().contiguous()
        if name.endswith("_sconv.static_w"):
            # full layers: GGUF [C, taps] holds the taps newest-first; the
            # official static_w is [taps, C] oldest-first
            return w.flip(1).t().contiguous()
        if name.endswith(("_sconv.weight", "attn_sconv.weight", "mlp_sconv.weight")):
            return w.reshape(w.shape[0], 1, w.shape[1])  # depthwise conv [C, 1, taps]
        return w

    @staticmethod
    def _experts(tensors, raw: str, name: str, t) -> Iterator[tuple[str, torch.Tensor]]:
        if t.tensor_type == Q.NVFP4:
            scale_raw = raw.removesuffix(".weight") + ".scale"
            if scale_raw not in tensors:
                raise ValueError(f"{raw}: NVFP4 experts without a global scale")
            data = torch.from_numpy(np.array(t.data))  # copy: the mmap is read-only
            packed, scales = ggml_nvfp4_to_official(data)
            if name.endswith("w13_weight"):
                packed = _interleave_gate_up(packed)
                scales = _interleave_gate_up(scales.view(torch.uint8)).view(
                    torch.float8_e4m3fn
                )
            amax = torch.from_numpy(np.array(tensors[scale_raw].data, dtype=np.float32))
            yield name, packed
            yield name + "_scale", scales
            yield name + "_amax", amax.reshape(-1)
            yield name + "_outer_scale_denominator", torch.ones(1, dtype=torch.float32)
        elif t.tensor_type in (*_FLOAT, Q.Q8_0):
            w = _dense_tensor(t)
            yield name, _interleave_gate_up(w) if name.endswith("w13_weight") else w
        else:
            raise ValueError(f"{raw}: unsupported expert type {t.tensor_type.name}")
