# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Frontend ports: tokenizer/encoder, chat template parity, reasoning + DSML tool parsers, 1Cat #597 regression."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.tokenizers import deepseek_v41_encoding as enc

CKPT = Path("/mnt/nvme2/models/DeepSeek-V4.1-Flash")
FIXTURES = CKPT / "encoding" / "tests"
TOKENIZER_DIR = Path("/mnt/nvme2/scratch/ds41/model-ref")     # tokenizer.json / tokenizer_config.json of 2cba9e42
TEMPLATE = CKPT / "chat_template.jinja"


def _fixture_ids() -> list[int]:
    return sorted(int(p.stem.rsplit("_", 1)[1]) for p in FIXTURES.glob("test_input_*.json"))


@pytest.fixture(scope="module")
def hf_tokenizer():
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip("V4.1 tokenizer files not available")
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast.from_pretrained(str(TOKENIZER_DIR))


@pytest.fixture(scope="module")
def v41_tokenizer(hf_tokenizer):
    from vllm.tokenizers.deepseek_v41 import get_deepseek_v41_tokenizer

    return get_deepseek_v41_tokenizer(hf_tokenizer)


def _jinja(hf_tokenizer, messages, **kwargs) -> str:
    """Render the shipped template. Jinja cannot JSON-decode, so OpenAI string ``arguments`` are given as dicts
    (what vLLM's chat preprocessing hands HF templates); the template would otherwise emit a single
    ``arguments`` parameter where encoding.py decodes the JSON."""
    if not TEMPLATE.is_file():
        pytest.skip("chat_template.jinja not available")
    import copy

    messages = copy.deepcopy(messages)
    for m in messages:
        for tc in m.get("tool_calls") or []:
            if isinstance(tc["function"].get("arguments"), str):
                tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    return hf_tokenizer.apply_chat_template(messages, chat_template=TEMPLATE.read_text(), tokenize=False, **kwargs)


# ------------------------------------------------------------------ encoder
def test_vendored_encoder_is_the_official_file() -> None:
    src = CKPT / "encoding" / "encoding.py"
    if not src.is_file():
        pytest.skip("checkpoint encoding.py not available")
    text = Path(enc.__file__).read_text()
    vendored = text.split("# ---- vendored file starts here ----\n", 1)[1]
    assert vendored == src.read_text()
    assert hashlib.sha256(src.read_bytes()).hexdigest() in text


@pytest.mark.parametrize("case_id", _fixture_ids() or [pytest.param(0, marks=pytest.mark.skip("no fixtures"))])
def test_official_fixture_byte_exact(case_id: int) -> None:
    case = enc.load_cases(str(FIXTURES / f"test_input_{case_id}.json"))[0]
    prompt, _ = enc.encode_case(case, thinking_mode="chat")
    assert prompt == (FIXTURES / f"test_output_{case_id}.txt").read_text()


def _case_request(case_id: int):
    raw = json.loads((FIXTURES / f"test_input_{case_id}.json").read_text())
    if isinstance(raw, list):
        raw = {"messages": raw}
    kwargs = {"thinking_mode": raw.get("thinking_mode") or "chat"}
    if raw.get("reasoning_effort") is not None:
        kwargs["reasoning_effort"] = raw["reasoning_effort"]
    return raw["messages"], raw.get("tools"), kwargs


TEXT_ONLY = [i for i in _fixture_ids() if i != 5]       # case 5 carries images (text-only port)


@pytest.mark.parametrize("case_id", TEXT_ONLY)
def test_wrapper_and_jinja_match_fixture(case_id: int, v41_tokenizer, hf_tokenizer) -> None:
    messages, tools, kwargs = _case_request(case_id)
    expected = (FIXTURES / f"test_output_{case_id}.txt").read_text()
    assert v41_tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, **kwargs) == expected
    assert _jinja(hf_tokenizer, messages, tools=tools, **kwargs) == expected
    ids = v41_tokenizer.apply_chat_template(messages, tools=tools, **kwargs)
    assert ids == hf_tokenizer.encode(expected, add_special_tokens=False)


def test_image_content_rejected(v41_tokenizer) -> None:
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]
    with pytest.raises(ValueError, match="text-only"):
        v41_tokenizer.apply_chat_template(messages, tokenize=False)


@pytest.mark.parametrize("effort, budget", [("low", 50), ("medium", 62), ("high", 75), ("max", 100), (88, 88),
                                            (None, 75), ("xhigh", 100), ("minimal", 50)])
def test_reasoning_effort_budgets(v41_tokenizer, hf_tokenizer, effort, budget) -> None:
    messages = [{"role": "user", "content": "hi"}]
    kwargs = {} if effort is None else {"reasoning_effort": effort}
    prompt = v41_tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    assert prompt.startswith(f"{enc.bos_token}<｜System｜>Reasoning Effort: {budget} (range 1-100")
    assert prompt.endswith("<｜Assistant｜><think>")
    if effort in ("low", "medium", "high", "max", 88, None):        # the values the jinja template defines
        assert _jinja(hf_tokenizer, messages, add_generation_prompt=True, **kwargs) == prompt


def test_chat_mode_selection(v41_tokenizer) -> None:
    messages = [{"role": "user", "content": "hi"}]
    chat = f"{enc.bos_token}<｜User｜>hi<｜Assistant｜></think>"
    for kwargs in ({"reasoning_effort": "none"}, {"enable_thinking": False}, {"thinking": False},
                   {"thinking_mode": "chat"}):
        assert v41_tokenizer.apply_chat_template(messages, tokenize=False, **kwargs) == chat, kwargs
    with pytest.raises(ValueError):
        v41_tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort=101)
    with pytest.raises(ValueError):
        v41_tokenizer.apply_chat_template(messages, tokenize=False, reasoning_effort="extreme")


def test_get_tokenizer_by_mode() -> None:
    if not (TOKENIZER_DIR / "tokenizer.json").is_file():
        pytest.skip("V4.1 tokenizer files not available")
    from vllm.tokenizers import get_tokenizer

    tok = get_tokenizer(str(TOKENIZER_DIR), tokenizer_mode="deepseek_v41")
    assert "DSV41" in type(tok).__name__
    prompt = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, reasoning_effort="medium")
    assert "Reasoning Effort: 62 " in prompt


def test_tokenizer_registered() -> None:
    from vllm.tokenizers.registry import TokenizerRegistry

    module, cls = TokenizerRegistry.tokenizers["deepseek_v41"]
    assert module.endswith("deepseek_v41") and cls == "DeepseekV41Tokenizer"


# ------------------------------------------------------------------ reasoning parser
def test_reasoning_parser_modes(hf_tokenizer) -> None:
    from vllm.reasoning import ReasoningParserManager
    from vllm.reasoning.deepseek_r1_reasoning_parser import DeepSeekR1ReasoningParser
    from vllm.reasoning.deepseek_v41_engine_reasoning_parser import DeepSeekV41EngineReasoningParser
    from vllm.reasoning.identity_reasoning_parser import IdentityReasoningParser

    assert ReasoningParserManager.get_reasoning_parser("deepseek_v41") is DeepSeekV41EngineReasoningParser
    default = DeepSeekV41EngineReasoningParser(hf_tokenizer, chat_template_kwargs={})
    assert default.thinking_mode == "thinking" and isinstance(default._parser, DeepSeekR1ReasoningParser)
    request = SimpleNamespace(include_reasoning=True)
    assert default.extract_reasoning("plan it</think>The answer.", request) == ("plan it", "The answer.")
    for kwargs in ({"reasoning_effort": "none", "enable_thinking": False}, {"thinking_mode": "chat"},
                   {"enable_thinking": False}):
        parser = DeepSeekV41EngineReasoningParser(hf_tokenizer, chat_template_kwargs=kwargs)
        assert isinstance(parser._parser, IdentityReasoningParser), kwargs


# ------------------------------------------------------------------ DSML tool parser + #597 regression
READ_FILE = {"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}
GOOD_597 = ("我来帮你读取 `/etc/hostname` 文件的内容。\n\n"
            "<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n"
            "<｜DSML｜ parameter name=\"path\" string=\"true\">/etc/hostname</｜DSML｜ parameter>\n"
            "</｜DSML｜ invoke>\n</｜DSML｜ calls>")
# 1Cat issue #597 (V4 on SM70): '<' dropped before the parameter tag and a stray U+FF09 in the parameter name,
# transposed to V4.1's spaced tags; and the name-only variant with well-formed tags
CORRUPT_597 = GOOD_597.replace("<｜DSML｜ parameter name=\"path\"", "｜DSML｜ parameter name=\"）path\"")
STRAY_NAME_597 = GOOD_597.replace("name=\"path\"", "name=\"）path\"")


def _tool_parser(hf_tokenizer, tools):
    from vllm.tool_parsers import ToolParserManager
    from vllm.tool_parsers.deepseekv41_engine_tool_parser import DeepSeekV41EngineToolParser

    assert ToolParserManager.get_tool_parser("deepseek_v41") is DeepSeekV41EngineToolParser
    return DeepSeekV41EngineToolParser(hf_tokenizer, tools)


def test_597_prompt_parity(v41_tokenizer, hf_tokenizer) -> None:
    messages = [{"role": "user", "content": "请读取 /etc/hostname 文件的内容"}]
    ours = v41_tokenizer.apply_chat_template(messages, tools=[READ_FILE], tokenize=False)
    official = enc.encode_messages([{"role": "system", "content": "", "tools": [READ_FILE]}] + messages,
                                   thinking_mode="thinking", reasoning_effort="high")
    assert ours == official == _jinja(hf_tokenizer, messages, tools=[READ_FILE], add_generation_prompt=True)
    assert "<｜DSML｜ calls>" in ours and ours.endswith("<｜Assistant｜><think>")


def test_dsml_special_token_roundtrip(hf_tokenizer) -> None:
    dsml_id = hf_tokenizer.convert_tokens_to_ids(enc.dsml_token)
    assert isinstance(dsml_id, int) and dsml_id != hf_tokenizer.unk_token_id
    ids = hf_tokenizer.encode(GOOD_597, add_special_tokens=False)
    assert ids.count(dsml_id) == 6                                   # every DSML tag uses the special token once
    assert hf_tokenizer.decode(ids, skip_special_tokens=False) == GOOD_597


def test_parse_good_597_and_official_example(hf_tokenizer) -> None:
    parser = _tool_parser(hf_tokenizer, [READ_FILE])
    out = parser.extract_tool_calls(GOOD_597, SimpleNamespace(tools=[READ_FILE]))
    assert out.tools_called and out.content == "我来帮你读取 `/etc/hostname` 文件的内容。"   # no "\n\n" lead
    (call,) = out.tool_calls
    assert call.function.name == "read_file" and json.loads(call.function.arguments) == {"path": "/etc/hostname"}
    example = ('summary\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name="lookup">\n'
               '<｜DSML｜ parameter name="query" string="true">value</｜DSML｜ parameter>\n'
               '<｜DSML｜ parameter name="limit" string="false">2</｜DSML｜ parameter>\n'
               '</｜DSML｜ invoke>\n</｜DSML｜ calls><｜end▁of▁sentence｜>')
    out = _tool_parser(hf_tokenizer, None).extract_tool_calls(example, SimpleNamespace(tools=None))
    assert out.tools_called and json.loads(out.tool_calls[0].function.arguments) == {"query": "value", "limit": 2}


@pytest.mark.parametrize("text", [CORRUPT_597, STRAY_NAME_597,
                                  GOOD_597.replace("｜DSML｜ ", "｜DSML｜").replace("｜DSML｜calls", "｜DSML｜tool_calls"),
                                  GOOD_597.replace("read_file", "write_file"),
                                  GOOD_597.replace('string="true">/etc/hostname', 'string="false">/etc/hostname'),
                                  GOOD_597[:-len("</｜DSML｜ calls>")]])
def test_malformed_dsml_is_never_a_tool_call(hf_tokenizer, text) -> None:
    out = _tool_parser(hf_tokenizer, [READ_FILE]).extract_tool_calls(text, SimpleNamespace(tools=[READ_FILE]))
    assert not out.tools_called and out.tool_calls == [] and out.content == text


EOS_ID = 1          # <｜end▁of▁sentence｜> in the V4.1 tokenizer (checked below)
PARITY_CASES = {
    "good": GOOD_597,
    "good_eos_text": GOOD_597 + "<｜end▁of▁sentence｜>",
    "good_ws_tail": GOOD_597 + "\n ",
    "corrupt_597": CORRUPT_597,
    "stray_name_597": STRAY_NAME_597,
    "v4_unspaced": GOOD_597.replace("｜DSML｜ ", "｜DSML｜").replace("｜DSML｜calls", "｜DSML｜tool_calls"),
    "wrong_tool": GOOD_597.replace("read_file", "write_file"),
    "bad_json": GOOD_597.replace('string="true">/etc/hostname', 'string="false">/etc/hostname'),
    "unterminated": GOOD_597[:-len("</｜DSML｜ calls>")],
    "unterminated_short": "Answer:\n\n<｜DSML｜ calls>\n<｜DSML｜ invoke name=\"read_file\">\n",
    "trailing_text": GOOD_597 + " trailing text",
    "two_blocks": GOOD_597 + "\n\n" + GOOD_597[GOOD_597.index("<｜DSML｜ calls>"):],
    "no_block": "Plain answer without tools.",
    "trailing_blank_lines": "Hello world\n\n",
    "partial_lead_at_end": "Hello\n\n<｜DSML｜",
    "block_only": GOOD_597[GOOD_597.index("<｜DSML｜ calls>"):],
}


def _stream(parser, text, step, final_via="eos"):
    deltas, prev = [], ""
    cuts = list(range(step, len(text), step)) + [len(text)]
    for k, i in enumerate(cuts):
        cur = text[:i]
        last = k == len(cuts) - 1
        ids = [EOS_ID] if last and final_via == "eos" else [7]
        d = parser.extract_tool_calls_streaming(prev, cur, cur[len(prev):], [], [], ids,
                                                SimpleNamespace(tools=[READ_FILE]))
        if d is not None:
            deltas.append(d)
        prev = cur
    if final_via == "finish":
        d = parser.finish_streaming(SimpleNamespace(tools=[READ_FILE]))
        if d is not None:
            deltas.append(d)
    return deltas


def test_eos_id(hf_tokenizer) -> None:
    assert hf_tokenizer.eos_token_id == EOS_ID


@pytest.mark.parametrize("final_via", ["eos", "finish"])
@pytest.mark.parametrize("step", [1, 2, 3, 5, 17, 10_000])
@pytest.mark.parametrize("case", sorted(PARITY_CASES))
def test_streaming_matches_non_streaming(hf_tokenizer, case, step, final_via) -> None:
    """MC-CORE F3: streaming reconstructs exactly what extract_tool_calls returns (content + calls)."""
    text = PARITY_CASES[case]
    ref = _tool_parser(hf_tokenizer, [READ_FILE]).extract_tool_calls(text, SimpleNamespace(tools=[READ_FILE]))
    parser = _tool_parser(hf_tokenizer, [READ_FILE])
    deltas = _stream(parser, text, step, final_via)
    content = "".join(d.content or "" for d in deltas)
    calls = [tc for d in deltas for tc in (d.tool_calls or [])]
    assert content == (ref.content or "")
    assert [(c.function.name, json.loads(c.function.arguments)) for c in calls] == \
        [(c.function.name, json.loads(c.function.arguments)) for c in ref.tool_calls]
    if ref.tools_called:
        assert [a["arguments"] for a in parser.prev_tool_call_arr] == parser.streamed_args_for_tool
    else:
        assert parser.prev_tool_call_arr == []


@pytest.mark.parametrize("case", ["good", "unterminated", "trailing_blank_lines", "corrupt_597"])
def test_delegating_parser_finish_hook(hf_tokenizer, case) -> None:
    """A stream ending WITHOUT an EOS token (length / stop string) is finalised through parse_delta(finished=True)."""
    from vllm.parser.abstract_parser import _WrappedParser
    from vllm.tool_parsers.deepseekv41_engine_tool_parser import DeepSeekV41EngineToolParser

    text = PARITY_CASES[case]
    ref = _tool_parser(hf_tokenizer, [READ_FILE]).extract_tool_calls(text, SimpleNamespace(tools=[READ_FILE]))

    class Wrapped(_WrappedParser):
        reasoning_parser_cls = None
        tool_parser_cls = DeepSeekV41EngineToolParser

    parser = Wrapped(hf_tokenizer, [READ_FILE])
    request = SimpleNamespace(tools=[READ_FILE], tool_choice="auto")
    deltas = []
    for i in range(0, len(text), 4):
        last = i + 4 >= len(text)
        d = parser.parse_delta(text[i: i + 4], [7], request, prompt_token_ids=[0], finished=last)
        if d is not None:
            deltas.append(d)
    assert "".join(d.content or "" for d in deltas) == (ref.content or "")
    assert len([tc for d in deltas for tc in (d.tool_calls or [])]) == len(ref.tool_calls)


def test_expected_verdicts(hf_tokenizer) -> None:
    called = {case for case, text in PARITY_CASES.items()
              if _tool_parser(hf_tokenizer, [READ_FILE]).extract_tool_calls(
                  text, SimpleNamespace(tools=[READ_FILE])).tools_called}
    assert called == {"good", "good_eos_text", "good_ws_tail", "block_only"}


def test_skip_special_tokens_disabled_for_tools(hf_tokenizer) -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "x"}], tools=[READ_FILE])
    assert _tool_parser(hf_tokenizer, None).adjust_request(request).skip_special_tokens is False


def test_renderer_registered_for_tokenizer_mode() -> None:
    """L-INTEG: tokenizer_mode deepseek_v41 (also the auto default for the arch) needs a renderer, else LLMEngine
    construction fails with "No renderer registered for renderer_mode='deepseek_v41'"."""
    from vllm.renderers.deepseek_v4 import DeepseekV4Renderer
    from vllm.renderers.registry import RENDERER_REGISTRY

    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v41") is DeepseekV4Renderer


@pytest.mark.parametrize("case_id", TEXT_ONLY)
def test_renderer_prompt_matches_official_fixture(case_id: int, v41_tokenizer, hf_tokenizer, monkeypatch) -> None:
    """L-CORE review of 17aae38bc: the chat path the server takes (DeepseekV4Renderer.render_messages with the V4.1
    tokenizer, kwargs from ChatParams) yields exactly the official encoder's tokens for every text-only fixture."""
    from vllm.renderers import deepseek_v4 as renderer_mod
    from vllm.renderers.params import ChatParams

    messages, tools, kwargs = _case_request(case_id)
    # parse_chat_messages needs a full ModelConfig; the V4.1 tokenizer encodes the original ``messages`` (not the
    # flattened conversation), so a pass-through keeps the path under test intact.
    monkeypatch.setattr(renderer_mod, "parse_chat_messages", lambda msgs, *a, **k: (list(msgs), None, None))
    renderer = renderer_mod.DeepseekV4Renderer.__new__(renderer_mod.DeepseekV4Renderer)
    renderer.model_config = None
    renderer.get_tokenizer = lambda: v41_tokenizer
    params = ChatParams(chat_template_kwargs={"tools": tools, **kwargs})
    _, prompt = renderer.render_messages(messages, params)
    expected = (FIXTURES / f"test_output_{case_id}.txt").read_text()
    assert prompt["prompt_token_ids"] == hf_tokenizer.encode(expected, add_special_tokens=False)
