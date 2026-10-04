# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, cast

import gguf
import regex as re
import torch
import torch.nn as nn

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
    gguf_quant_weights_iterator_multi,
)
from vllm.transformers_utils.gguf_utils import detect_gguf_multimodal
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.gguf import GGUFConfig

logger = init_logger(__name__)


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
        # repo id/filename.gguf
        if "/" in model_name_or_path and model_name_or_path.endswith(".gguf"):
            from vllm.transformers_utils.gguf_files import resolve_gguf_file

            return str(
                resolve_gguf_file(
                    model_name_or_path,
                    revision=model_config.revision,
                    cache_dir=self.load_config.download_dir,
                )
            )
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
            "(expected local file, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )

    @staticmethod
    def _get_all_gguf_files(model_path: str) -> list[str]:
        """Discover all GGUF shard files from a single shard path.

        Supports variable-width shard indices by dynamically detecting
        the padding from the original filename.
        E.g. ``*-00001-of-00005.gguf`` → all 5 shards,
             ``*-01-of-15.gguf`` → all 15 shards.
        """
        from vllm.transformers_utils.gguf_files import gguf_shard_paths

        return [str(path) for path in gguf_shard_paths(model_path)]

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
        text_config = config.get_text_config()
        if text_config.model_type == "qwen4_exp_text":
            from vllm.transformers_utils.gguf_config import load_gguf_config

            native_config = load_gguf_config(self._prepare_weights(model_config))
            # HF dimensions remain authoritative. Add only checkpoint-specific
            # hash/storage metadata absent from the supplied config.
            for key in (
                "gguf_architecture",
                "gguf_ple_constants",
                "gguf_ple_eos_token_id",
            ):
                if not hasattr(text_config, key):
                    setattr(text_config, key, getattr(native_config, key))
        from vllm.transformers_utils.gguf_files import (
            gguf_shard_paths,
            gguf_tensor_index,
        )

        from .gguf_adapters import get_gguf_adapter

        if (
            adapter := get_gguf_adapter(
                config, tp_size=getattr(self, "_gguf_tp_size", 1)
            )
        ) is not None:
            local = self._prepare_weights(model_config)
            self._native_adapter = adapter
            self._native_tensors = gguf_tensor_index(gguf_shard_paths(local))
            return adapter.build_name_map(self._native_tensors)
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
        if model_type == "minimax_m2":
            model_type = "minimax-m2"
            # GGUF layer map assumes merged expert weights
            # map them manually like deepseek2
            for idx in range(config.num_hidden_layers):
                gguf_to_hf_name_map[f"blk.{idx}.exp_probs_b.bias"] = (
                    f"model.layers.{idx}.block_sparse_moe.e_score_correction_bias"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_down_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w2.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_gate_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w1.weight"
                )
                gguf_to_hf_name_map[f"blk.{idx}.ffn_up_exps.weight"] = (
                    f"model.layers.{idx}.block_sparse_moe.experts.0.w3.weight"
                )
                sideload_params.append(
                    re.compile(
                        f"model\\.layers\\.{idx}"
                        r"\.block_sparse_moe\.experts\.(gate_up_proj|down_proj)"
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

        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

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

        if model_type == "minimax-m2" and not hf_checkpoint_map:
            # Reverse HF convention: mlp -> block_sparse_moe
            state_dict = {
                name.replace(".mlp.", ".block_sparse_moe."): tensor
                for name, tensor in state_dict.items()
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
            # In transformers v5, multimodal models (e.g. Gemma3) wrap
            # all sub-models under an outer 'model.' attribute, producing
            # state_dict keys like 'model.language_model.layers.0...' and
            # 'model.vision_tower.vision_model...'.  Strip this outer
            # prefix so the keys match what gguf-py expects.
            if is_multimodal and hf_name.startswith("model."):
                hf_name = hf_name[6:]  # Remove outer 'model.'

            # Strip 'language_model.' prefix for multimodal models - gguf-py
            # tensor mappings expect parameter names without this prefix.
            # Note: 'model.' prefix should be KEPT for text-only models as
            # gguf-py expects it.
            if hf_name.startswith("language_model."):
                hf_name = hf_name[15:]  # Remove 'language_model.'
                # Re-add 'model.' prefix because gguf-py text tensor maps
                # expect 'model.layers...' format.
                if is_multimodal:
                    hf_name = "model." + hf_name

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
        gguf_files = self._get_all_gguf_files(model_name_or_path)
        weight_type_map = {}
        for f in gguf_files:
            weight_type_map.update(get_gguf_weight_type_map(f, gguf_to_hf_name_map))
        is_multimodal = hasattr(model_config.hf_config, "vision_config")
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

        Yields:
            Tuples of (parameter_name, tensor) for all model weights
        """
        if (adapter := getattr(self, "_native_adapter", None)) is not None:
            yield from adapter.weights(
                self._native_tensors, gguf_to_hf_name_map, model_config.dtype
            )
            return
        hf_config = model_config.hf_config
        is_multimodal = hasattr(hf_config, "vision_config")

        if is_multimodal:
            # Load mm_proj (mm_encoder + projector) for multimodal weights
            mmproj_file = detect_gguf_multimodal(model_name_or_path)
            assert mmproj_file is not None, (
                "Could not find mm_proj file for multimodal GGUF model"
            )
            yield from gguf_quant_weights_iterator(mmproj_file, gguf_to_hf_name_map)

        gguf_files = self._get_all_gguf_files(model_name_or_path)
        if len(gguf_files) > 1:
            yield from gguf_quant_weights_iterator_multi(
                gguf_files, gguf_to_hf_name_map
            )
        else:
            yield from gguf_quant_weights_iterator(
                model_name_or_path, gguf_to_hf_name_map
            )

    def download_model(self, model_config: ModelConfig) -> None:
        self._prepare_weights(model_config)

    def get_all_weights(
        self,
        model_config: ModelConfig,
        model: nn.Module,
        *,
        skip_weight: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """Filter mapped names before the adapter touches tensor payloads."""
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        if skip_weight is not None:
            gguf_weights_map = {
                raw: name
                for raw, name in gguf_weights_map.items()
                if not skip_weight(name)
            }
        yield from self._get_weights_iterator(
            model_config, local_model_path, gguf_weights_map
        )

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        model.load_weights(self.get_all_weights(model_config, model))

    def load_model(
        self, vllm_config: VllmConfig, model_config: ModelConfig, prefix: str = ""
    ) -> nn.Module:
        device_config = vllm_config.device_config
        self._gguf_tp_size = vllm_config.parallel_config.tensor_parallel_size
        local_model_path = self._prepare_weights(model_config)
        gguf_weights_map = self._get_gguf_weights_map(model_config)
        # we can only know if tie word embeddings after mapping weights
        gguf_files = self._get_all_gguf_files(local_model_path)
        all_extra_names = []
        for f in gguf_files:
            all_extra_names.extend(get_gguf_extra_tensor_names(f, gguf_weights_map))
        if "lm_head.weight" in all_extra_names:
            model_config.hf_config.update({"tie_word_embeddings": True})

        weight_type_map = self._get_gguf_weight_type(
            model_config, local_model_path, gguf_weights_map
        )
        # filter out unquantized modules to skip
        adapter = getattr(self, "_native_adapter", None)
        unquant_names = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16")
            and name.endswith(".weight")
            and (adapter is None or not adapter.is_linear(name))
        ]
        if adapter is not None:
            quant_config = cast("GGUFConfig", vllm_config.quant_config)
            for raw, name in gguf_weights_map.items():
                adapter.needs_dense_fallback(name, self._native_tensors[raw])
            quant_config.fallback_reasons = adapter.fallback_reasons
            quant_config.linear_layouts = adapter.linear_layouts(gguf_weights_map)
            quant_config.native_expert_storage = getattr(
                adapter, "native_expert_storage", False
            )
            if "output.weight" not in self._native_tensors:
                model_config.hf_config.tie_word_embeddings = True
        logger.debug("GGUF unquantized modules: %s", unquant_names)
        if TYPE_CHECKING:
            vllm_config.quant_config = cast(GGUFConfig, vllm_config.quant_config)
        vllm_config.quant_config.unquantized_modules.extend(unquant_names)

        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(vllm_config=vllm_config, prefix=prefix)
            self.load_weights(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
        return model
