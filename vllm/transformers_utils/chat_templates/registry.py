# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypeAlias

from vllm.logger import init_logger

logger = init_logger(__file__)

CHAT_TEMPLATES_DIR = Path(__file__).parent

ChatTemplatePath: TypeAlias = Path | Callable[[str], Path | None]


def _get_qwen_chat_template_fallback(tokenizer_name_or_path: str) -> Path | None:
    if tokenizer_name_or_path.endswith("-Chat"):
        return CHAT_TEMPLATES_DIR / "template_chatml.jinja"

    return CHAT_TEMPLATES_DIR / "template_basic.jinja"


def _get_minicpmv_chat_template_fallback(tokenizer_name_or_path: str) -> Path | None:
    # MiniCPM-V-4.5 version uses a dedicated template
    if "4.5" in tokenizer_name_or_path or "4_5" in tokenizer_name_or_path:
        return CHAT_TEMPLATES_DIR / "template_minicpmv45.jinja"

    # Other versions use chatml template
    return CHAT_TEMPLATES_DIR / "template_chatml.jinja"


_MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {
    "blip-2": CHAT_TEMPLATES_DIR / "template_blip2.jinja",
    "chameleon": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "clip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "colpali": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "deepseek_ocr": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "deepseek_ocr2": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
    "deepseek_vl_v2": CHAT_TEMPLATES_DIR / "template_deepseek_vl2.jinja",
    "fuyu": CHAT_TEMPLATES_DIR / "template_fuyu.jinja",
    "minicpmv": _get_minicpmv_chat_template_fallback,
    "minicpmv4_6": _get_minicpmv_chat_template_fallback,
    "paligemma": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "qwen": _get_qwen_chat_template_fallback,
    "siglip": CHAT_TEMPLATES_DIR / "template_basic.jinja",
    "siglip2": CHAT_TEMPLATES_DIR / "template_basic.jinja",
}

# Architecture-keyed fallbacks take precedence over model_type ones: some
# checkpoints reuse another family's model_type while needing a different
# prompt contract. DeepSeek-OCR reports model_type "deepseek_vl_v2", whose
# fallback template inserts chat role markers ("<|User|>: ...") and would
# silently change the OCR prompt relative to the offline recipe; the OCR
# architecture must resolve the raw-concatenation OCR template instead.
# Keyed by architecture so genuine DeepSeek-VL2 chat checkpoints (same
# model_type, different architecture) keep their chat template.
_ARCH_TO_CHAT_TEMPLATE_FALLBACK: dict[str, ChatTemplatePath] = {
    "DeepseekOCRForCausalLM": CHAT_TEMPLATES_DIR / "template_deepseek_ocr.jinja",
}


def register_chat_template_fallback_path_for_arch(
    architecture: str,
    chat_template: ChatTemplatePath,
) -> None:
    """Register an architecture-keyed chat template fallback (plugins too)."""
    if architecture in _ARCH_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Architecture %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.",
            architecture,
            chat_template,
        )

    _ARCH_TO_CHAT_TEMPLATE_FALLBACK[architecture] = chat_template


def register_chat_template_fallback_path(
    model_type: str,
    chat_template: ChatTemplatePath,
) -> None:
    if model_type in _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK:
        logger.warning(
            "Model type %s already has a chat template registered. "
            "It will be overwritten by the new chat template %s.",
            model_type,
            chat_template,
        )

    _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK[model_type] = chat_template


def get_chat_template_fallback_path(
    model_type: str,
    tokenizer_name_or_path: str,
    architectures: Iterable[str] | None = None,
) -> Path | None:
    chat_template: ChatTemplatePath | None = None
    for arch in architectures or ():
        chat_template = _ARCH_TO_CHAT_TEMPLATE_FALLBACK.get(arch)
        if chat_template is not None:
            break
    if chat_template is None:
        chat_template = _MODEL_TYPE_TO_CHAT_TEMPLATE_FALLBACK.get(model_type)
    if callable(chat_template):
        chat_template = chat_template(tokenizer_name_or_path)

    if chat_template is None:
        return None

    return chat_template
