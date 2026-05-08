# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionToolsParam
from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
from vllm.tool_parsers.utils import get_json_schema_from_tools


def test_tool_schema_defs_extraction_does_not_mutate_request_tool() -> None:
    parameters = {
        "type": "object",
        "$defs": {
            "City": {
                "type": "string",
            },
        },
        "properties": {
            "city": {
                "$ref": "#/$defs/City",
            },
        },
        "required": ["city"],
    }
    tool = ChatCompletionToolsParam(
        function=FunctionDefinition(
            name="get_weather",
            parameters=deepcopy(parameters),
        ),
    )

    schema = get_json_schema_from_tools("required", [tool])

    assert tool.function.parameters == parameters
    assert isinstance(schema, dict)
    assert schema["$defs"] == parameters["$defs"]
    nested_parameters = schema["items"]["anyOf"][0]["properties"]["parameters"]
    assert "$defs" not in nested_parameters
