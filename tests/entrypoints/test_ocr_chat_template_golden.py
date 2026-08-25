# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden tests for the DeepSeek-OCR chat-template contract.

The OCR checkpoint reports model_type "deepseek_vl_v2"; that model_type's
fallback template inserts chat role markers and silently changes the OCR
prompt relative to the validated offline recipe. These goldens pin (a) the
architecture-keyed resolution to the raw-concatenation OCR template and
(b) the exact rendered prompt strings for both documented OCR modes.
"""

import jinja2
import pytest

from vllm.transformers_utils.chat_templates.registry import (
    get_chat_template_fallback_path,
)

pytestmark = pytest.mark.cpu_test


def _render(template_path, messages, add_generation_prompt=True):
    env = jinja2.Environment()  # noqa: S701 (prompt templates, not HTML)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(
        jinja2.TemplateError(msg)
    )
    template = env.from_string(template_path.read_text())
    return template.render(
        messages=messages,
        bos_token="",
        eos_token="",
        add_generation_prompt=add_generation_prompt,
    )


def test_ocr_architecture_resolves_ocr_template_despite_vl2_model_type():
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2",
        tokenizer_name_or_path="x",
        architectures=["DeepseekOCRForCausalLM"],
    )
    assert path is not None and path.name == "template_deepseek_ocr.jinja"


def test_genuine_vl2_chat_model_keeps_vl2_template():
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2",
        tokenizer_name_or_path="x",
        architectures=["DeepseekVLV2ForCausalLM"],
    )
    assert path is not None and path.name == "template_deepseek_vl2.jinja"
    # And with no architecture info at all (old call shape), behavior is
    # unchanged.
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2", tokenizer_name_or_path="x"
    )
    assert path is not None and path.name == "template_deepseek_vl2.jinja"


def test_golden_plain_ocr_prompt_renders_byte_exact():
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2",
        tokenizer_name_or_path="x",
        architectures=["DeepseekOCRForCausalLM"],
    )
    out = _render(path, [{"role": "user", "content": "<image>\nFree OCR. "}])
    assert out == "<image>\nFree OCR. "


def test_golden_grounding_prompt_renders_byte_exact():
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2",
        tokenizer_name_or_path="x",
        architectures=["DeepseekOCRForCausalLM"],
    )
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    out = _render(path, [{"role": "user", "content": prompt}])
    assert out == prompt


def test_negative_control_vl2_template_injects_role_markers():
    """Proves the goldens discriminate: rendering the same message through
    the model_type fallback (what serving would do without the arch key)
    changes the prompt."""
    path = get_chat_template_fallback_path(
        model_type="deepseek_vl_v2", tokenizer_name_or_path="x"
    )
    out = _render(path, [{"role": "user", "content": "<image>\nFree OCR. "}])
    assert out != "<image>\nFree OCR. "
    assert "<|User|>: " in out and "<|Assistant|>: " in out
