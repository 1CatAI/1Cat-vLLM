# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""V4.1 leading-space DSML tags, including partial streaming tag boundaries."""

import json

from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser


class DeepSeekV41ToolParser(DeepSeekV32ToolParser):
    tool_call_start_token = "<｜DSML｜ calls>"
    tool_call_end_token = "</｜DSML｜ calls>"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"
    # The existing JSON-based required/named constraint is not V4.1 DSML.
    supports_required_and_named = False

    def _convert_params_with_schema(self, function_name, param_dict):
        del function_name
        converted = {}
        for name, (value, string_attr) in param_dict.items():
            if string_attr == "true":
                converted[name] = value
            else:
                try:
                    converted[name] = json.loads(value)
                except json.JSONDecodeError:
                    converted[name] = value
        return converted
