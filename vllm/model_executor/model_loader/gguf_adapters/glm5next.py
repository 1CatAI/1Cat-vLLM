# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Restore GLM5-Next GGUF tensors to the native GLM checkpoint contract.

Layout rules follow llama.cpp conversion/glm.py: the exporter splits MLA KV-B
per head and transposes its K part; KDA convolutions gain singleton dimensions;
A_log becomes -exp(A_log). These are inverted explicitly, without guessing
transposes from coincidentally compatible shapes. Routed experts retain their
original GGML blocks through the shared stacked-expert transport.
"""

import copy

import gguf
import regex as re
import torch

from vllm.logger import init_logger

from .common import decode_dense_tensor, require_tensor_shape
from .qwen35 import _dequantize_embedding
from .qwen35_moe import Qwen35MoeAdapter

logger = init_logger(__name__)

_TensorSpec = tuple[str, tuple[int, ...]]


class Glm5NextAdapter(Qwen35MoeAdapter):
    native_expert_storage = True
    packed_embeddings = False
    architecture_label = "GLM5-Next"

    def __init__(self, config, tp_size=1):
        self.config = copy.copy(config)
        self.tp_size = tp_size
        self.layout = None
        self.fallback_reasons = {}
        self.excluded_tensor_names: dict[str, str] = {}
        if config.n_routed_experts != getattr(
            config, "num_experts", config.n_routed_experts
        ):
            raise ValueError("GLM5-Next expert config aliases disagree")
        self.config.num_experts = config.n_routed_experts

    def tensor_specs(self) -> tuple[dict[str, _TensorSpec], dict[str, _TensorSpec]]:
        """Return required and optional raw names, destinations and logical shapes."""
        c = self.config
        hidden, vocab = c.hidden_size, c.vocab_size
        heads, kv = c.num_attention_heads, c.kv_lora_rank
        qk, value, q_rank = c.qk_nope_head_dim, c.v_head_dim, c.q_lora_rank
        linear_dim = c.linear_num_heads * c.linear_head_dim
        streams = c.mhc_num_residual_streams
        mix = (streams + 2) * streams
        specs: dict[str, _TensorSpec] = {
            "token_embd.weight": ("model.embed_tokens.weight", (vocab, hidden)),
            "output_norm.weight": ("model.norm.weight", (hidden,)),
            "output.weight": ("lm_head.weight", (vocab, hidden)),
        }
        optional: dict[str, _TensorSpec] = {}
        shape: tuple[int, ...]
        for block in range(c.num_hidden_layers + c.num_nextn_predict_layers):
            base = f"model.layers.{block}."
            layer: dict[str, _TensorSpec] = {
                "attn_norm.weight": ("input_layernorm.weight", (hidden,)),
                "ffn_norm.weight": ("post_attention_layernorm.weight", (hidden,)),
            }
            if block < c.num_hidden_layers:
                for part in ("attn", "ffn"):
                    for suffix, shape in (
                        ("fn", (mix, streams * hidden)),
                        ("base", (mix,)),
                        ("scale", (3,)),
                    ):
                        layer[f"hc_{part}_{suffix}.weight"] = (
                            f"hc_{part}_{suffix}",
                            shape,
                        )
            if c.is_kda_layer(block):
                for projection in ("q", "k", "v"):
                    layer[f"attn_{projection}.weight"] = (
                        f"self_attn.{projection}_proj.weight",
                        (linear_dim, hidden),
                    )
                    layer[f"ssm_conv1d_{projection}.weight"] = (
                        f"self_attn.{projection}_conv1d.weight",
                        (linear_dim, 1, c.linear_conv_kernel_dim),
                    )
                layer.update(
                    {
                        "attn_output.weight": (
                            "self_attn.o_proj.weight",
                            (hidden, linear_dim),
                        ),
                        "ssm_beta.weight": (
                            "self_attn.b_proj.weight",
                            (c.linear_num_heads, hidden),
                        ),
                        "ssm_f_a.weight": (
                            "self_attn.f_a_proj.weight",
                            (c.linear_head_dim, hidden),
                        ),
                        "ssm_f_b.weight": (
                            "self_attn.f_b_proj.weight",
                            (linear_dim, c.linear_head_dim),
                        ),
                        "ssm_g_a.weight": (
                            "self_attn.g_a_proj.weight",
                            (c.linear_head_dim, hidden),
                        ),
                        "ssm_g_b.weight": (
                            "self_attn.g_b_proj.weight",
                            (linear_dim, c.linear_head_dim),
                        ),
                        "ssm_a": ("self_attn.A_log", (c.linear_num_heads,)),
                        "ssm_dt.bias": ("self_attn.dt_bias", (linear_dim,)),
                        "ssm_norm.weight": (
                            "self_attn.o_norm.weight",
                            (c.linear_head_dim,),
                        ),
                    }
                )
            else:
                layer.update(
                    {
                        "attn_q_a.weight": (
                            "self_attn.q_a_proj.weight",
                            (q_rank, hidden),
                        ),
                        "attn_q_b.weight": (
                            "self_attn.q_b_proj.weight",
                            (heads * qk, q_rank),
                        ),
                        "attn_q_a_norm.weight": (
                            "self_attn.q_a_layernorm.weight",
                            (q_rank,),
                        ),
                        "attn_kv_a_mqa.weight": (
                            "self_attn.kv_a_proj_with_mqa.weight",
                            (kv, hidden),
                        ),
                        "attn_kv_a_norm.weight": (
                            "self_attn.kv_a_layernorm.weight",
                            (kv,),
                        ),
                        "attn_k_b.weight": (
                            "self_attn.kv_b_proj.weight",
                            (heads, kv, qk),
                        ),
                        "attn_v_b.weight": (
                            "self_attn.kv_b_proj.weight",
                            (heads, value, kv),
                        ),
                        "attn_output.weight": (
                            "self_attn.o_proj.weight",
                            (hidden, heads * value),
                        ),
                        "indexer.attn_q_b.weight": (
                            "self_attn.indexer.wq_b.weight",
                            (c.index_n_heads * c.index_head_dim, q_rank),
                        ),
                        "indexer.attn_k.weight": (
                            "self_attn.indexer.wk.weight",
                            (c.index_head_dim, hidden),
                        ),
                        "indexer.proj.weight": (
                            "self_attn.indexer.weights_proj.weight",
                            (c.index_n_heads, hidden),
                        ),
                        "indexer.k_norm.weight": (
                            "self_attn.indexer.k_norm.weight",
                            (c.index_head_dim,),
                        ),
                        "indexer.k_norm.bias": (
                            "self_attn.indexer.k_norm.bias",
                            (c.index_head_dim,),
                        ),
                        "indexer_compressor_gate.weight": (
                            "self_attn.indexer.index_kpool_compress_gate",
                            (c.index_head_dim, hidden),
                        ),
                        "indexer_compressor_ape.weight": (
                            "self_attn.indexer.index_kpool_compress_ape",
                            (c.index_kpool, c.index_head_dim),
                        ),
                    }
                )
            dense = block < c.num_hidden_layers and c.mlp_layer_types[block] == "dense"
            if dense:
                for projection in ("gate", "up", "down"):
                    shape = (
                        (hidden, c.intermediate_size)
                        if projection == "down"
                        else (c.intermediate_size, hidden)
                    )
                    layer[f"ffn_{projection}.weight"] = (
                        f"mlp.{projection}_proj.weight",
                        shape,
                    )
            else:
                layer["ffn_gate_inp.weight"] = (
                    "mlp.gate.weight",
                    (c.n_routed_experts, hidden),
                )
                layer["exp_probs_b.bias"] = (
                    "mlp.gate.e_score_correction_bias",
                    (c.n_routed_experts,),
                )
                for projection in ("gate", "up", "down"):
                    expert_shape = (
                        (hidden, c.moe_intermediate_size)
                        if projection == "down"
                        else (c.moe_intermediate_size, hidden)
                    )
                    shared_intermediate = c.moe_intermediate_size * c.n_shared_experts
                    shared_shape = (
                        (hidden, shared_intermediate)
                        if projection == "down"
                        else (shared_intermediate, hidden)
                    )
                    layer[f"ffn_{projection}_exps.weight"] = (
                        f"mlp.experts.{projection}_proj.weight",
                        (c.n_routed_experts, *expert_shape),
                    )
                    layer[f"ffn_{projection}_shexp.weight"] = (
                        f"mlp.shared_experts.{projection}_proj.weight",
                        shared_shape,
                    )
            if block >= c.num_hidden_layers:
                layer.update(
                    {
                        "nextn.eh_proj.weight": (
                            "eh_proj.weight",
                            (hidden, 2 * hidden),
                        ),
                        "nextn.enorm.weight": ("enorm.weight", (hidden,)),
                        "nextn.hnorm.weight": ("hnorm.weight", (hidden,)),
                    }
                )
                for suffix, dest, shape in (
                    ("embed_tokens.weight", "embed_tokens.weight", (vocab, hidden)),
                    (
                        "shared_head_head.weight",
                        "shared_head.head.weight",
                        (vocab, hidden),
                    ),
                    ("shared_head_norm.weight", "shared_head.norm.weight", (hidden,)),
                ):
                    optional[f"blk.{block}.nextn.{suffix}"] = (base + dest, shape)
            specs.update(
                {
                    f"blk.{block}.{raw}": (base + dest, shape)
                    for raw, (dest, shape) in layer.items()
                }
            )
        return specs, optional

    def _validated_names(self, tensors):
        required, optional = self.tensor_specs()
        missing = set(required) - tensors.keys()
        unknown = tensors.keys() - required.keys() - optional.keys()
        if missing or unknown:
            raise ValueError(
                f"GLM5-Next GGUF tensor inventory mismatch: missing={sorted(missing)}, "
                f"unmapped={sorted(unknown)}"
            )
        result = {}
        for raw, tensor in tensors.items():
            name, shape = required[raw] if raw in required else optional[raw]
            # The exporter uses [1, channels, 1, kernel]; older GGUF writers
            # trim the outer singleton dimension. Both have identical order.
            if ".ssm_conv1d_" in raw and len(tensor.shape) == 4:
                require_tensor_shape(tensor, (1, *shape), raw)
            else:
                require_tensor_shape(tensor, shape, raw)
            result[raw] = name
        return result

    def build_name_map(self, tensors):
        result = {}
        self.excluded_tensor_names = {}
        for raw, name in self._validated_names(tensors).items():
            match = re.match(r"blk\.(\d+)\.", raw)
            if match and int(match[1]) >= self.config.num_hidden_layers:
                self.excluded_tensor_names[raw] = (
                    "validated NextN tensor; target backbone excludes draft head"
                )
            else:
                result[raw] = name
        if self.excluded_tensor_names:
            logger.info(
                "GLM5-Next GGUF validated %d backbone and %d separate NextN tensors.",
                len(result),
                len(self.excluded_tensor_names),
            )
        return result

    @staticmethod
    def is_linear(name):
        return name.endswith(".weight") and (
            name.endswith("lm_head.weight")
            or name.endswith(".shared_head.head.weight")
            or (".mlp." in name and ".mlp.gate." not in name)
        )

    def needs_dense_fallback(self, name, tensor):
        if ".mlp.experts." in name:
            return False
        if not self.is_linear(name):
            return False  # The GLM modules explicitly own dense parameters.
        return super().needs_dense_fallback(name, tensor)

    @staticmethod
    def _dense_dtype(name, dtype):
        if any(
            part in name
            for part in (
                ".hc_attn_",
                ".hc_ffn_",
                ".q_conv1d.",
                ".k_conv1d.",
                ".v_conv1d.",
            )
        ) or name.endswith(
            (
                ".A_log",
                ".dt_bias",
                ".e_score_correction_bias",
                ".mlp.gate.weight",
                ".indexer.weights_proj.weight",
                ".index_kpool_compress_ape",
            )
        ):
            return torch.float32
        return dtype

    def validate_loaded_parameters(self, model, loaded):
        """Fail before postprocessing if a local text parameter was skipped.

        Bulk expert transport fills independent banks directly. Its four empty
        vLLM mapping parameters are checked through the actual per-projection
        payload ledger, never exempted merely because they are quantized.
        """
        from vllm.model_executor.layers.quantization.gguf_moe import GGUFNativeMoEMethod

        if not isinstance(loaded, set):
            raise ValueError("GLM5-Next GGUF requires a loaded-parameter set")
        expected = {
            name
            for name, _ in model.named_parameters()
            if not name.startswith("visual.")
        }
        for prefix, module in model.named_modules():
            if prefix == "visual" or prefix.startswith("visual."):
                continue
            method = getattr(module, "quant_method", None)
            if isinstance(method, GGUFNativeMoEMethod):
                expert_ids = set(range(method.num_experts))
                for shard in ("w1", "w3", "w2"):
                    if (
                        shard not in method.weight_types
                        or method.loaded_experts.get(shard) != expert_ids
                    ):
                        raise ValueError(
                            "GLM5-Next GGUF incomplete expert payloads: "
                            f"{prefix}.{shard}"
                        )
                for name in (
                    "w13_qweight",
                    "w2_qweight",
                    "w13_qweight_type",
                    "w2_qweight_type",
                ):
                    expected.discard(f"{prefix}.{name}" if prefix else name)
            if "_wp_fp32" in module._buffers:
                value = module._buffers["_wp_fp32"]
                if value is None or value.dtype != torch.float32:
                    raise ValueError(
                        f"GLM5-Next GGUF missing FP32 indexer gate: {prefix}"
                    )
        if missing := expected - loaded:
            raise ValueError(
                "GLM5-Next GGUF did not load text parameters: "
                + ", ".join(sorted(missing))
            )

    def weights(self, tensors, name_map, dtype):
        regular = {
            raw: name for raw, name in name_map.items() if ".mlp.experts." not in name
        }
        experts = {
            raw: name for raw, name in name_map.items() if ".mlp.experts." in name
        }
        for raw, name in regular.items():
            if self.is_linear(name):
                kind = tensors[raw].tensor_type
                if kind in (
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.BF16,
                ) or self.needs_dense_fallback(name, tensors[raw]):
                    kind = {
                        torch.float16: gguf.GGMLQuantizationType.F16,
                        torch.bfloat16: gguf.GGMLQuantizationType.BF16,
                        torch.float32: gguf.GGMLQuantizationType.F32,
                    }[dtype]
                yield (
                    name.removesuffix(".weight") + ".qweight_type",
                    torch.tensor(int(kind)),
                )
        for raw, name in regular.items():
            tensor = tensors[raw]
            if raw.endswith(".attn_v_b.weight"):
                partner = raw.replace(".attn_v_b.weight", ".attn_k_b.weight")
                if partner not in regular or regular[partner] != name:
                    raise ValueError(
                        f"GLM5-Next MLA KV-B pair was filtered incompletely: {raw}"
                    )
                continue
            if raw.endswith(".attn_k_b.weight"):
                partner = raw.replace(".attn_k_b.weight", ".attn_v_b.weight")
                if partner not in regular or regular[partner] != name:
                    raise ValueError(
                        f"GLM5-Next MLA KV-B pair was filtered incompletely: {raw}"
                    )
                key = decode_dense_tensor(tensor, dtype, raw)
                value = decode_dense_tensor(tensors[partner], dtype, partner)
                # Exact inverse of exporter.view(heads, qk+v, kv).split() and
                # k_b.transpose(1, 2), preserving each head's K-then-V rows.
                weight = torch.cat((key.transpose(1, 2), value), dim=1)
                yield name, weight.reshape(-1, self.config.kv_lora_rank)
                continue
            if name.endswith("embed_tokens.weight"):
                weight = _dequantize_embedding(tensor, dtype, raw)
                if any(
                    not torch.isfinite(chunk).all()
                    for chunk in weight.view(-1).split(1 << 20)
                ):
                    raise ValueError(f"GLM5-Next {raw}: non-finite embedding values")
                yield name, weight
                continue
            if self.is_linear(name):
                floating = tensor.tensor_type in (
                    gguf.GGMLQuantizationType.F16,
                    gguf.GGMLQuantizationType.F32,
                    gguf.GGMLQuantizationType.BF16,
                )
                weight = (
                    decode_dense_tensor(tensor, dtype, raw)
                    if floating or self.needs_dense_fallback(name, tensor)
                    else torch.from_numpy(tensor.data.copy())
                )
                yield name.removesuffix(".weight") + ".qweight", weight
                continue
            weight = decode_dense_tensor(tensor, self._dense_dtype(name, dtype), raw)
            if name.endswith(".A_log"):
                if not torch.all(weight < 0):
                    raise ValueError(
                        f"GLM5-Next {raw} must contain negative exp(A_log)"
                    )
                weight = torch.log(-weight).reshape(
                    1, 1, self.config.linear_num_heads, 1
                )
            elif ".ssm_conv1d_" in raw:
                weight = weight.reshape(
                    self.config.linear_num_heads * self.config.linear_head_dim,
                    1,
                    self.config.linear_conv_kernel_dim,
                )
            yield name, weight
        yield from self._expert_weights(tensors, experts, dtype)


class Glm5NextMTPAdapter(Glm5NextAdapter):
    """Load the NextN head from the same GGUF as its target, on the final PP rank.

    Validate the complete directory before selecting the draft subset. Module
    paths are final paths, including mtp_block, so expert arenas and quantized
    linear layouts see the same names as the weight loader.
    """

    def build_name_map(self, tensors):
        names = self._validated_names(tensors)
        if self.config.num_nextn_predict_layers != 1:
            raise ValueError("GLM5-Next GGUF MTP currently requires one NextN layer")
        block = self.config.num_hidden_layers
        raw_prefix = f"blk.{block}."
        prefix = f"model.layers.{block}."
        embed = raw_prefix + "nextn.embed_tokens.weight"
        head = raw_prefix + "nextn.shared_head_head.weight"
        norm = raw_prefix + "nextn.shared_head_norm.weight"
        if norm not in names:
            raise ValueError("GLM5-Next GGUF MTP requires its shared_head_norm tensor")
        self.has_own_embed_tokens = embed in names
        self.has_own_lm_head = head in names
        result = {}
        for raw, name in names.items():
            if not raw.startswith(raw_prefix):
                continue
            if raw == embed:
                name = "model.embed_tokens.weight"
            elif ".nextn." not in raw:
                name = name.replace(prefix, prefix + "mtp_block.", 1)
            result[raw] = name
        if not self.has_own_embed_tokens:
            result["token_embd.weight"] = "model.embed_tokens.weight"
        if not self.has_own_lm_head:
            result["output.weight"] = prefix + "shared_head.head.weight"
        self.excluded_tensor_names = {
            raw: "validated target tensor; not needed by the NextN draft"
            for raw in names.keys() - result.keys()
        }
        return result

    def configure_model(self, model):
        # Distinct optional draft embeddings/heads must survive the later
        # shared-weight binding pass. A quantized head is compared conservatively.
        model.has_own_embed_tokens = self.has_own_embed_tokens
        model.has_own_lm_head = self.has_own_lm_head
        if self.has_own_lm_head:
            model.lm_head = model.model.layers[
                str(self.config.num_hidden_layers)
            ].shared_head.head
