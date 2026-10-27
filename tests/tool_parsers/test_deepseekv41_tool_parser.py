# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from unittest.mock import MagicMock

import pytest

from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers.deepseekv41_tool_parser import DeepSeekV41ToolParser

TEXT = (
    '说明<｜DSML｜ calls>\n<｜DSML｜ invoke name="get_weather">\n'
    '<｜DSML｜ parameter name="city" string="true">北京</｜DSML｜ parameter>\n'
    '<｜DSML｜ parameter name="days" string="false">2</｜DSML｜ parameter>\n'
    "</｜DSML｜ invoke>\n</｜DSML｜ calls>"
)


def _parser():
    tokenizer = MagicMock()
    tokenizer.get_vocab.return_value = {}
    return DeepSeekV41ToolParser(tokenizer)


def test_v41_registration_and_complete_typed_tool_call():
    assert ToolParserManager.get_tool_parser("deepseek_v41") is DeepSeekV41ToolParser
    parsed = _parser().extract_tool_calls(TEXT, MagicMock())
    assert parsed.tools_called and parsed.content == "说明"
    assert parsed.tool_calls[0].function.name == "get_weather"
    assert json.loads(parsed.tool_calls[0].function.arguments) == {
        "city": "北京",
        "days": 2,
    }


@pytest.mark.parametrize("chunk", [1, 2, 7, 23])
def test_spaced_tags_stream_across_arbitrary_character_boundaries(chunk):
    parser, previous, content, fragments, names = _parser(), "", [], [], []
    for first in range(0, len(TEXT), chunk):
        delta_text = TEXT[first : first + chunk]
        current = previous + delta_text
        delta = parser.extract_tool_calls_streaming(
            previous, current, delta_text, [], [], [1], MagicMock()
        )
        previous = current
        if delta is None:
            continue
        if delta.content:
            content.append(delta.content)
        for call in delta.tool_calls or []:
            if call.function.name:
                names.append(call.function.name)
            if call.function.arguments:
                fragments.append(call.function.arguments)
    assert "".join(content) == "说明"
    assert names == ["get_weather"]
    assert json.loads("".join(fragments)) == {"city": "北京", "days": 2}
