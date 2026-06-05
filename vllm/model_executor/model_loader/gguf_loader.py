# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import glob
import os
from collections.abc import Generator

import gguf
import numpy as np
import regex as re
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    download_gguf,
    get_gguf_extra_tensor_names,
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
)
from vllm.transformers_utils.gguf_utils import detect_gguf_multimodal
from vllm.utils.torch_utils import set_default_torch_dtype

logger = init_logger(__name__)


_GGUF_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def _resolve_gguf_shards(model_name_or_path: str) -> list[str]:
    """Return the list of GGUF shards for `model_name_or_path`.

    GGUF "split" files are named ``<base>-NNNNN-of-MMMMM.gguf``. vLLM is
    typically given the first shard; the remaining shards live next to it
    in the same directory. For non-split files, returns a single-element
    list with the original path.
    """
    base = os.path.basename(model_name_or_path)
    m = _GGUF_SPLIT_RE.match(base)
    if m is None:
        return [model_name_or_path]
    prefix, _, total = m.groups()
    directory = os.path.dirname(model_name_or_path)
    pattern = os.path.join(directory, f"{prefix}-*-of-{total}.gguf")
    shards = sorted(glob.glob(pattern))
    return shards if shards else [model_name_or_path]


def _mistral4_moe_iterator(
    gguf_files: list[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Synthesize per-expert ``gate_proj`` and ``up_proj`` qweights from the
    fused ``ffn_gate_up_exps`` GGUF tensor.

    Bartowski's Mistral4 GGUF ships the gate+up MoE projection FUSED into a
    single quantized tensor ``blk.X.ffn_gate_up_exps.weight`` of physical
    shape ``(num_experts, 2*intermediate, packed_hidden_bytes)``. vLLM's
    ``GGUFMoEMethod`` materializes ``w13_qweight`` from two separate w1/w3
    full loads (gate & up), so we split the fused bytes along axis=1 (rows
    of the matrix — quant blocks pack along the innermost ``hidden`` axis,
    so row-axis splits are byte-clean) and yield two qweight events per
    layer plus matching ``qweight_type`` events. Convention follows HF
    Mistral4: first half is gate, second half is up
    (``gate_up_proj[:, :intermediate, :]`` = gate).

    Yields ``qweight_type`` events first across all layers (per the
    ``gguf_quant_weights_iterator`` convention) before any qweight events.
    """
    fused: list[tuple[int, gguf.ReaderTensor]] = []
    for path in gguf_files:
        reader = gguf.GGUFReader(path)
        for t in reader.tensors:
            if not t.name.startswith("blk."):
                continue
            if not t.name.endswith(".ffn_gate_up_exps.weight"):
                continue
            try:
                layer_idx = int(t.name.split(".")[1])
            except (IndexError, ValueError):
                continue
            fused.append((layer_idx, t))

    fused.sort(key=lambda x: x[0])

    # Pass 1: yield qweight_type for gate then up (same value).
    for layer_idx, t in fused:
        wt = torch.tensor(t.tensor_type)
        for shard in ("gate_proj", "up_proj"):
            name = (
                f"model.layers.{layer_idx}.mlp.experts.0.{shard}.qweight_type"
            )
            yield name, wt

    # Pass 2: yield qweight bytes for gate (rows [0:half)) then up
    # (rows [half:end)) per layer.
    for layer_idx, t in fused:
        data = np.ascontiguousarray(t.data)  # (experts, 2*intermediate, bytes)
        if data.ndim != 3:
            raise RuntimeError(
                f"Mistral4 ffn_gate_up_exps for layer {layer_idx} expected "
                f"ndim=3, got {data.ndim} (shape={data.shape})"
            )
        mid = data.shape[1]
        if mid % 2 != 0:
            raise RuntimeError(
                f"Mistral4 ffn_gate_up_exps middle dim {mid} must be even "
                f"(layer {layer_idx})"
            )
        half = mid // 2
        # Mistral4 transformers HF code does `gate, up = chunk(2, dim=-1)`
        # after Linear, so first half = gate. llama.cpp's converter preserves
        # this layout for the fused ffn_gate_up_exps tensor.
        gate_bytes = data[:, :half, :]
        up_bytes = data[:, half:, :]
        # FusedMoE.weight_loader reads loaded_weight.shape so the input must
        # be a torch tensor with the 3D shape; the kernel only sees the
        # raw bytes.
        for shard, slice_ in (("gate_proj", gate_bytes), ("up_proj", up_bytes)):
            name = f"model.layers.{layer_idx}.mlp.experts.0.{shard}.qweight"
            yield name, torch.from_numpy(np.ascontiguousarray(slice_))


def _mistral4_kv_b_iterator(
    gguf_files: list[str],
    num_heads: int,
    kv_lora: int,
    qk_nope: int,
    v_head: int,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Synthesize fused ``kv_b_proj.weight`` from split ``attn_k_b`` + ``attn_v_b``.

    Bartowski's Mistral4 GGUFs ship the MLA K/V decompression already split
    into two tensors per layer (``attn_k_b`` Q8_0, ``attn_v_b`` Q6_K), but
    our model class wants a fused ``kv_b_proj`` matching the HF Linear
    convention ``[num_heads*(qk_nope+v_head), kv_lora]`` with K rows first
    per head, then V rows.

    GGUF logical layout (gguf-py, innermost first):
      attn_k_b: (qk_nope, kv_lora, heads) → memory: heads, kv_lora, qk_nope
      attn_v_b: (kv_lora, v_head, heads)  → memory: heads, v_head, kv_lora

    Merge per layer:
      k = dequant(attn_k_b).reshape(heads, kv_lora, qk_nope).transpose(1,2)
      v = dequant(attn_v_b).reshape(heads, v_head, kv_lora)
      fused = cat([k, v], dim=1).reshape(heads*(qk_nope+v_head), kv_lora)
    """
    out_per_head = qk_nope + v_head
    # Index by layer idx to pair k_b with v_b. Accumulate across ALL shards
    # before pairing — gguf-split partitions on cumulative byte size, so a
    # layer's attn_k_b and attn_v_b may land in different shards. Mirrors
    # the cross-shard accumulation pattern in _mistral4_moe_iterator above.
    k_b_tensors: dict[int, gguf.ReaderTensor] = {}
    v_b_tensors: dict[int, gguf.ReaderTensor] = {}
    for path in gguf_files:
        reader = gguf.GGUFReader(path)
        for t in reader.tensors:
            if not t.name.startswith("blk."):
                continue
            parts = t.name.split(".")
            try:
                layer_idx = int(parts[1])
            except (IndexError, ValueError):
                continue
            if t.name.endswith(".attn_k_b.weight"):
                k_b_tensors[layer_idx] = t
            elif t.name.endswith(".attn_v_b.weight"):
                v_b_tensors[layer_idx] = t
    for layer_idx in sorted(k_b_tensors):
        if layer_idx not in v_b_tensors:
            logger.warning(
                "Mistral4 GGUF layer %d has attn_k_b but no attn_v_b; "
                "skipping kv_b_proj synthesis (kv_b_proj will remain "
                "uninitialized).",
                layer_idx,
            )
            continue
        t_k = k_b_tensors[layer_idx]
        t_v = v_b_tensors[layer_idx]
        from vllm import _custom_ops as _ops

        k_data = torch.from_numpy(np.ascontiguousarray(t_k.data)).cuda()
        v_data = torch.from_numpy(np.ascontiguousarray(t_v.data)).cuda()
        k_dq = _ops.ggml_dequantize(
            k_data, int(t_k.tensor_type), num_heads * kv_lora, qk_nope,
            torch.float16,
        )
        v_dq = _ops.ggml_dequantize(
            v_data, int(t_v.tensor_type), num_heads * v_head, kv_lora,
            torch.float16,
        )
        k_b = k_dq.view(num_heads, kv_lora, qk_nope).transpose(1, 2)
        v_b = v_dq.view(num_heads, v_head, kv_lora)
        fused = torch.cat([k_b, v_b], dim=1).contiguous()
        fused = fused.view(num_heads * out_per_head, kv_lora).cpu()
        del k_data, v_data, k_dq, v_dq, k_b, v_b
        yield (
            f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight",
            fused,
        )


def _mimo2_attn_qkv_iterator(
    gguf_files: list[str],
    hf_config,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Split fused ``attn_qkv`` into per-shard q_proj / k_proj / v_proj.

    Bartowski's MiMo-V2.5 GGUF ships each layer's QKV projection as a
    single fused ``blk.X.attn_qkv.weight`` tensor (HF "Pro" layout). vLLM's
    QKVParallelLinear, when GGUF-quantized, keeps a 3-element
    ``qweight_type`` vector keyed by shard_id ∈ {"q","k","v"} — so the
    fused tensor cannot be loaded with ``default_weight_loader``. We slice
    along output dim 0 (each row = one output feature, so the cut is row-
    aligned in the quantized byte layout) and yield three sub-tensors
    that the model's stacked_params_mapping then fans back into
    ``qkv_proj``.

    Sizes are layer-dependent because hybrid_layer_pattern == 1 (SWA)
    uses different KV head counts than full-attention layers.
    """
    from vllm.distributed import get_tensor_model_parallel_world_size

    tp_size = get_tensor_model_parallel_world_size()
    head_dim_full = hf_config.head_dim
    v_head_dim_full = hf_config.v_head_dim
    num_heads_full = hf_config.num_attention_heads
    num_kv_heads_full = hf_config.num_key_value_heads
    head_dim_swa = hf_config.swa_head_dim
    v_head_dim_swa = getattr(hf_config, "swa_v_head_dim", head_dim_swa)
    num_heads_swa = hf_config.swa_num_attention_heads
    num_kv_heads_swa = hf_config.swa_num_key_value_heads
    pattern = list(getattr(hf_config, "hybrid_layer_pattern", []))

    def layer_dims(layer_idx: int):
        """Return (nh, nkv, head_dim, v_head_dim) for this layer."""
        kind = pattern[layer_idx] if layer_idx < len(pattern) else 0
        if kind == 1:  # SWA
            return (num_heads_swa, num_kv_heads_swa, head_dim_swa, v_head_dim_swa)
        return (
            num_heads_full,
            num_kv_heads_full,
            head_dim_full,
            v_head_dim_full,
        )

    # Index attn_qkv tensors by layer across all shards (gguf-split can
    # land different layers' tensors in different shards). Bartowski's
    # MiMo-V2.5 GGUF carries `nextn_predict_layers` (MTP) trailing blocks
    # past num_hidden_layers; we don't load those (our model class is
    # text-only), so cap on hf_config.num_hidden_layers.
    num_text_layers = hf_config.num_hidden_layers
    qkv_tensors: dict[int, gguf.ReaderTensor] = {}
    for path in gguf_files:
        reader = gguf.GGUFReader(path)
        for t in reader.tensors:
            if not t.name.endswith(".attn_qkv.weight"):
                continue
            parts = t.name.split(".")
            try:
                layer_idx = int(parts[1])
            except (IndexError, ValueError):
                continue
            if layer_idx >= num_text_layers:
                continue
            qkv_tensors[layer_idx] = t

    # Pass 1: yield all qweight_type scalars first (matches the contract of
    # gguf_quant_weights_iterator — packed-layer loaders need types before
    # data so each shard slot has its type recorded before kernel selection).
    for layer_idx in sorted(qkv_tensors):
        t = qkv_tensors[layer_idx]
        wtype = t.tensor_type
        if wtype.name in ("F32", "BF16", "F16"):
            continue
        base = f"model.layers.{layer_idx}.self_attn"
        wtype_tensor = torch.tensor(wtype)
        for sub in ("q_proj", "k_proj", "v_proj"):
            yield (f"{base}.{sub}.qweight_type", wtype_tensor)

    # Pass 2: split the data along dim 0 and yield qweight (quantized) or
    # weight (unquantized) sub-tensors. When num_kv_heads < tp_size,
    # vLLM's GGUF weight_loader for QKVParallelLinear divides naively by
    # tp_size (no per-head replication), so we have to pre-replicate K/V
    # rows along dim 0 — each KV head appears num_kv_head_replicas times
    # consecutively, matching the replica layout the non-GGUF path bakes
    # into QKVParallelLinear (rank → kv_head_idx via tp_rank // replicas).
    for layer_idx in sorted(qkv_tensors):
        t = qkv_tensors[layer_idx]
        wtype = t.tensor_type
        nh, nkv, hd, vhd = layer_dims(layer_idx)
        q_size, k_size, v_size = nh * hd, nkv * hd, nkv * vhd
        out_rows = int(t.data.shape[0])
        expected = q_size + k_size + v_size
        assert out_rows == expected, (
            f"layer {layer_idx}: attn_qkv has {out_rows} output rows but "
            f"hf_config implies q+k+v = {expected}"
        )
        tdata = torch.from_numpy(np.ascontiguousarray(t.data))
        suffix = (
            "weight" if wtype.name in ("F32", "BF16", "F16") else "qweight"
        )
        base = f"model.layers.{layer_idx}.self_attn"
        q_part = tdata[:q_size]
        k_part = tdata[q_size : q_size + k_size]
        v_part = tdata[q_size + k_size :]
        replicas = max(1, tp_size // nkv)
        if replicas > 1:
            packed_in = tdata.shape[-1]
            k_part = (
                k_part.view(nkv, hd, packed_in)
                .repeat_interleave(replicas, dim=0)
                .contiguous()
                .view(nkv * replicas * hd, packed_in)
            )
            v_part = (
                v_part.view(nkv, vhd, packed_in)
                .repeat_interleave(replicas, dim=0)
                .contiguous()
                .view(nkv * replicas * vhd, packed_in)
            )
        yield (f"{base}.q_proj.{suffix}", q_part)
        yield (f"{base}.k_proj.{suffix}", k_part)
        yield (f"{base}.v_proj.{suffix}", v_part)


class GGUFModelLoader(BaseModelLoader):
    """
    Model loader that can load GGUF files. This is useful for loading models
    that are quantized with GGUF and saved in the GGUF format. This loader
    supports loading both full models and sharded models.
    """

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        if load_config.model_loader_extra_config:
            raise ValueError(
                f"Model loader extra config is not supported for "
                f"load format {load_config.load_format}"
            )

    def _prepare_weights(self, model_config: ModelConfig):
        model_name_or_path = model_config.model
        if os.path.isfile(model_name_or_path):
            return model_name_or_path
        # for raw HTTPS link
        if model_name_or_path.startswith(
            ("http://", "https://")
        ) and model_name_or_path.endswith(".gguf"):
            return hf_hub_download(url=model_name_or_path)
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            repo_id, filename = model_name_or_path.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        # repo_id:quant_type
        elif "/" in model_name_or_path and ":" in model_name_or_path:
            repo_id, quant_type = model_name_or_path.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )

        raise ValueError(
            f"Unrecognised GGUF reference: {model_name_or_path} "
            "(expected local file, raw URL, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )

    def _get_gguf_weights_map(self, model_config: ModelConfig):
        """
        GGUF uses this naming convention for their tensors from HF checkpoint:
        `blk.N.BB.weight` and `blk.N.BB.bias`
        where N signifies the block number of a layer, and BB signifies the
        attention/mlp layer components.
        See "Standardized tensor names" in
        https://github.com/ggerganov/ggml/blob/master/docs/gguf.md for details.
        """
        config = model_config.hf_config
        # Get text config to handle both nested (multimodal) and flat
        # (text-only) config structures. For multimodal models like
        # Gemma3Config, this returns config.text_config. For text-only
        # models, this returns config itself.
        text_config = config.get_text_config()
        model_type = config.model_type
        is_multimodal = (
            hasattr(config, "vision_config") and config.vision_config is not None
        )
        gguf_to_hf_name_map = {}
        sideload_params: list[re.Pattern] = []
        # hack: ggufs have a different name than transformers
        if model_type == "cohere":
            model_type = "command-r"
        if model_type == "gemma3_text":
            # Gemma3 models use "gemma3_text" in HuggingFace but
            # "gemma3" in GGUF architecture naming
            model_type = "gemma3"
        if model_type in ("deepseek_v3", "deepseek_v2"):
            model_type = "deepseek2"
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen2_moe", "qwen3_moe"):
            model_type = model_type.replace("_", "")
            # GGUF layer map assumes that we will have a merged expert weights
            # so we need to map them manually
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type in ("qwen3_5_moe", "qwen3_5"):
            # Qwen3.5/3.6 ship as a multimodal *ForConditionalGeneration*
            # wrapper, but the bartowski text GGUF carries only the LM backbone
            # (gguf arch "qwen35moe"/"qwen35"). Load text-only: drop the vision
            # path and operate on the text config. gguf-py already maps the
            # hybrid Gated-DeltaNet / SSM tensors (in_proj_*, A_log, conv1d,
            # ssm_norm, out_proj); we only patch the gaps below.
            is_multimodal = False
            config = text_config
            if model_type == "qwen3_5_moe":
                for idx in range(config.num_hidden_layers):
                    # 256-expert MoE with shared experts, merged per-expert in
                    # the GGUF (like qwen3_moe). Map merged gate/up/down to
                    # expert 0 (FusedMoE.weight_loader full-loads ndim==3).
                    gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                        f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                    )
                    gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                        f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                    )
                    gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                        f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                    )
                    sideload_params.append(
                        re.compile(
                            f"model\\.layers\\.{idx}"
                            r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                        )
                    )
                    # Gated-DeltaNet dt_bias is a bare nn.Parameter (no
                    # .weight/.bias suffix), so find_hf_name_in_tensor_map
                    # can't route it through gguf-py. Map explicitly.
                    gguf_to_hf_name_map[f"blk.{idx}.ssm_dt.bias"] = (
                        f"model.layers.{idx}.linear_attn.dt_bias"
                    )
                model_type = "qwen35moe"
            else:
                model_type = "qwen35"
        if model_type == "mimo_v2":
            # gguf-py uses arch name "mimo2"; HF model_type is "mimo_v2".
            model_type = "mimo2"
            # Bartowski's MiMo-V2.5 GGUF ships attn_qkv FUSED but
            # QKVParallelLinear's GGUF buffers expect shard_id-aware loading
            # (qweight_type is a 3-element vector keyed by Q/K/V). We split
            # the fused tensor in `_mimo2_attn_qkv_iterator` and yield it as
            # q_proj / k_proj / v_proj so the model's stacked_params_mapping
            # fans them back into qkv_proj. The HF dummy model exposes a
            # single qkv_proj.weight (HF Pro layout); register a sideload
            # pattern so the unmapped-param check ignores it.
            sideload_params.append(
                re.compile(r"model\.layers\.[0-9]+\.self_attn\.qkv_proj\.weight")
            )
            # Layer 0 is a dense MLP (ffn_gate / ffn_up / ffn_down — the
            # default name map handles these). Layers 1..N-1 are MoE.
            # attention_sink_bias is an nn.Parameter (no .weight suffix in
            # the HF state_dict), which trips find_hf_name_in_tensor_map's
            # trailing-dot bug. Override explicitly for every layer that
            # carries a sink in the hybrid pattern (1 = SWA layer).
            for idx, kind in enumerate(
                getattr(text_config, "hybrid_layer_pattern", [])
            ):
                if kind == 1:
                    gguf_to_hf_name_map[f"blk.{idx}.attn_sinks.weight"] = (
                        f"model.layers.{idx}.self_attn.attention_sink_bias"
                    )
            for idx in range(1, config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.mlp.gate.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.gate_proj.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.up_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )
        if model_type == "mistral4":
            # Mistral4 ships a Mistral4Config + Mistral4ForCausalLM in
            # transformers, but Mistral4ForCausalLM is not in the
            # AutoModelForCausalLM registry (HF intends it to be wrapped by
            # Mistral3ForConditionalGeneration). For text-only GGUF (Bartowski
            # strips the wrapper, so model_type == "mistral4" at the top
            # level), register it explicitly so from_config() below resolves.
            try:
                from transformers import Mistral4Config, Mistral4ForCausalLM
                AutoModelForCausalLM.register(
                    Mistral4Config, Mistral4ForCausalLM, exist_ok=True
                )
            except (ImportError, ValueError):
                pass
            # MoE expert tensors. Bartowski's GGUF stores ffn_down_exps as a
            # single 3D tensor, and (unlike deepseek2) ships gate+up FUSED as
            # ffn_gate_up_exps. Map ffn_down_exps to expert 0's down_proj
            # (FusedMoE.weight_loader detects ndim==3 and full-loads all
            # experts at once). The fused gate_up tensor is split into
            # synthesized gate / up qweights by `_mistral4_moe_iterator`,
            # so it has no entry here.
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.mlp.experts.0.down_proj.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.mlp\.experts\.[0-9]+\.(gate|up|down)_proj\.weight"
                    )
                )

        arch = None
        for key, value in gguf.MODEL_ARCH_NAMES.items():
            if value == model_type:
                arch = key
                break
        if arch is None:
            raise RuntimeError(f"Unknown gguf model_type: {model_type}")
        text_num_layers = text_config.num_hidden_layers
        text_name_map = gguf.get_tensor_name_map(arch, text_num_layers)

        if is_multimodal:
            mm_proj_arch = gguf.MODEL_ARCH.MMPROJ
            vision_num_layers = config.vision_config.num_hidden_layers
            vision_name_map = gguf.get_tensor_name_map(mm_proj_arch, vision_num_layers)
        else:
            vision_name_map = None

        # Create dummy model to extract parameter names
        # For multimodal: use AutoModelForImageTextToText to get
        # language + vision + projector params
        # For text-only: use AutoModelForCausalLM to get language model params
        auto_cls = (
            AutoModelForImageTextToText if is_multimodal else AutoModelForCausalLM
        )
        with torch.device("meta"):
            dummy_model = auto_cls.from_config(
                config, trust_remote_code=model_config.trust_remote_code
            )

        state_dict = dummy_model.state_dict()
        if hf_checkpoint_map := getattr(
            dummy_model, "_checkpoint_conversion_mapping", None
        ):

            def revert_hf_rename(name: str) -> str:
                for original_name, hf_name in hf_checkpoint_map.items():
                    if hf_name in name:
                        name = name.replace(hf_name, original_name).lstrip("^")
                return name

            state_dict = {
                revert_hf_rename(name): tensor for name, tensor in state_dict.items()
            }

        def find_hf_name_in_tensor_map(hf_name: str) -> str | None:
            """
            Map HuggingFace parameter name to GGUF tensor name.

            This function handles the mismatch between HF parameter naming
            conventions and gguf-py's expected format:
            1. Strips 'model.' prefix (common in multimodal models)
            2. Converts '_weight' suffix to '.weight' (Gemma3 compatibility)
            3. Searches vision_name_map for multimodal parameters
            4. Falls back to text_name_map for language model parameters

            Args:
                hf_name: Full HuggingFace parameter name (e.g.,
                        'model.multi_modal_projector.mm_soft_emb_norm.weight')

            Returns:
                GGUF tensor name with suffix (e.g., 'mm.soft_emb_norm.weight')
                or None if no mapping found
            """
            # Strip 'language_model.' prefix for multimodal models - gguf-py
            # tensor mappings expect parameter names without this prefix.
            # Note: 'model.' prefix should be KEPT for text-only models as
            # gguf-py expects it.
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]  # Remove 'language_model.'

            # Parse parameter name and suffix
            if hf_name.endswith((".weight", ".bias")):
                base_name, suffix = hf_name.rsplit(".", 1)
            else:
                base_name, suffix = hf_name, ""
                # Handle '_weight' suffix (Gemma3 naming: parameter ends with
                # '_weight' instead of '.weight')
                if base_name.endswith("_weight"):
                    base_name = base_name[:-7]  # Remove '_weight'
                    suffix = "weight"

            gguf_name = None
            # Priority 1: Search vision/projector parameters for multimodal models
            if vision_name_map is not None:
                gguf_name = vision_name_map.get_name(base_name)

            # Priority 2: Search text backbone parameters
            if gguf_name is None:
                gguf_name = text_name_map.get_name(base_name)

            if gguf_name is None:
                return None

            return gguf_name + "." + suffix

        # Build mapping and track unmapped parameters
        unmapped_params = []
        for hf_name in state_dict:
            gguf_name_with_suffix = find_hf_name_in_tensor_map(hf_name)

            # Track mapping success
            if gguf_name_with_suffix is not None:
                gguf_to_hf_name_map[gguf_name_with_suffix] = hf_name
                logger.debug("Mapped GGUF %s → HF %s", gguf_name_with_suffix, hf_name)
            elif hf_name not in gguf_to_hf_name_map.values():
                # Parameter not in manual overrides either
                unmapped_params.append(hf_name)

        # All parameters (except those initialized by other means) must be mapped:
        # both vision/projector and backbone
        if unmapped_params:
            unmapped_params = list(
                filter(
                    lambda x: not any(re.fullmatch(p, x) for p in sideload_params),
                    unmapped_params,
                )
            )
        if unmapped_params:
            raise RuntimeError(
                f"Failed to map GGUF parameters "
                f"({len(unmapped_params)}): "
                f"{unmapped_params}"
            )
        return gguf_to_hf_name_map

    def _get_gguf_weight_type(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map = get_gguf_weight_type_map(
            model_name_or_path, gguf_to_hf_name_map
        )
        is_multimodal = (
            hasattr(model_config.hf_config, "vision_config")
            and model_config.hf_config.vision_config is not None
        )
        if is_multimodal:
            mmproj_file = detect_gguf_multimodal(model_name_or_path)
            assert mmproj_file is not None, (
                "Could not find mm_proj file for multimodal GGUF model"
            )
            logger.info("Loading extra mm_proj weights from %s...", mmproj_file)
            mm_proj_weight_type_map = get_gguf_weight_type_map(
                mmproj_file, gguf_to_hf_name_map
            )
            weight_type_map.update(mm_proj_weight_type_map)
        return weight_type_map

    def _get_weights_iterator(
        self,
        model_config: ModelConfig,
        model_name_or_path: str,
        gguf_to_hf_name_map: dict[str, str],
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """
        Iterate over GGUF model weights, loading from both main model file and
        mmproj.gguf for multimodal Gemma3 models.

        For Gemma3 multimodal GGUF models:
        - Main file (gemma-3-*.gguf): Language model weights (model.*)
        - mmproj file (mmproj*.gguf): Vision tower + projector weights (v.*, mm.*)

        For split GGUF files (``*-NNNNN-of-MMMMM.gguf``), iterates over all
        sibling shards in the same directory.

        For Mistral4 GGUFs (Bartowski layout), prepends a synthesizer that
        builds a fused ``kv_b_proj.weight`` from per-layer ``attn_k_b`` +
        ``attn_v_b`` pairs (the upstream gguf-py mapping points at
        ``attn_kv_b`` which Bartowski never writes).

        Yields:
            Tuples of (parameter_name, tensor) for all model weights
        """
        hf_config = model_config.hf_config
        is_multimodal = (
            hasattr(hf_config, "vision_config")
            and hf_config.vision_config is not None
        )

        if is_multimodal:
            # Load mm_proj (mm_encoder + projector) for multimodal weights
            mmproj_file = detect_gguf_multimodal(model_name_or_path)
            assert mmproj_file is not None, (
                "Could not find mm_proj file for multimodal GGUF model"
            )
            yield from gguf_quant_weights_iterator(mmproj_file, gguf_to_hf_name_map)

        shards = _resolve_gguf_shards(model_name_or_path)
        if len(shards) > 1:
            logger.info("Loading GGUF model from %d shards: %s",
                        len(shards), [os.path.basename(s) for s in shards])

        if hf_config.model_type == "mistral4":
            yield from _mistral4_kv_b_iterator(
                shards,
                num_heads=hf_config.num_attention_heads,
                kv_lora=hf_config.kv_lora_rank,
                qk_nope=hf_config.qk_nope_head_dim,
                v_head=hf_config.v_head_dim,
            )
            yield from _mistral4_moe_iterator(shards)

        if hf_config.model_type == "mimo_v2":
            yield from _mimo2_attn_qkv_iterator(shards, hf_config)

        for shard in shards:
            yield from gguf_quant_weights_iterator(shard, gguf_to_hf_name_map)

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        model.load_weights(
            self._get_weights_iterator(model_config, local_model_path, gguf_weights_map)
        )

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)

        # Resolve all shards (single file → [path]; split file → all siblings).
        shards = _resolve_gguf_shards(local_model_path)
        # tie_word_embeddings: lm_head must be missing across ALL shards, not
        # just the first one (would otherwise wrongly tie when lm_head lives
        # in a later shard).
        all_extra: set[str] = set(gguf_weights_map.values())
        for shard in shards:
            all_extra &= set(get_gguf_extra_tensor_names(shard, gguf_weights_map))
        if "lm_head.weight" in all_extra:
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map: dict[str, str] = {}
        for shard in shards:
            weight_type_map.update(
                self._get_gguf_weight_type(model_config, shard, gguf_weights_map)
            )
        # filter out unquantized modules to skip
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]
        logger.debug(
            "GGUF unquantized modules: %s",
            unquant_names,
        )
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
