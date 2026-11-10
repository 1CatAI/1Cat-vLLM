# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm import envs
from vllm.parser.abstract_parser import _WrappedParser
from vllm.tool_parsers import ToolParserManager
from vllm.tool_parsers import _hotreload as reload

HELPER = "vllm.tool_parsers._reload_test_helper"
PARSER = "vllm.tool_parsers._reload_test_parser"
PARSER_SOURCE = """from vllm.tool_parsers import ToolParser, ToolParserManager
from vllm.tool_parsers._reload_test_helper import VALUE
@ToolParserManager.register_module("audit_reload")
class Parser(ToolParser):
    def extract_tool_calls(self, *args, **kwargs):
        return VALUE
    def extract_tool_calls_streaming(self, *args, **kwargs):
        return VALUE
def value():
    return VALUE
"""


@pytest.fixture
def source(monkeypatch, tmp_path):
    monkeypatch.setattr(ToolParserManager, "tool_parsers", {})
    monkeypatch.setattr(ToolParserManager, "lazy_parsers", {})
    monkeypatch.setattr(reload, "_VERSION", 0)
    modules = {}
    paths = {}
    for name, content in ((HELPER, "VALUE = 1\n"), (PARSER, PARSER_SOURCE)):
        path = tmp_path / (name.rsplit(".", 1)[1] + ".py")
        path.write_text(content)
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
        paths[name] = path
    ToolParserManager.tool_parsers["audit_reload"] = modules[PARSER].Parser
    monkeypatch.setattr(reload, "_implementation_modules", lambda: modules)
    return SimpleNamespace(modules=modules, paths=paths, cls=modules[PARSER].Parser)


def test_failed_reload_preserves_globals_and_registry(source):
    old_request = source.modules[PARSER].value
    registry = ToolParserManager.tool_parsers
    lazy = dict(ToolParserManager.lazy_parsers)
    source.paths[HELPER].write_text("VALUE = 2\n")
    source.paths[PARSER].write_text(
        PARSER_SOURCE + '\nVALUE = 20001\nraise RuntimeError("bad edit")\n'
    )
    with pytest.raises(RuntimeError, match="bad edit"):
        reload.reload_now()
    assert old_request() == 1
    assert source.modules[HELPER].VALUE == 1
    assert ToolParserManager.tool_parsers is registry
    assert ToolParserManager.lazy_parsers == lazy
    assert reload.current_version() == 0


def test_success_uses_new_helper_and_preserves_old_requests(source):
    old_request = object.__new__(source.cls)
    source.paths[HELPER].write_text("VALUE = 2\n")
    result = reload.reload_now()
    new_cls = ToolParserManager.tool_parsers["audit_reload"]
    assert result["version"] == 1
    assert new_cls is not source.cls
    assert object.__new__(new_cls).extract_tool_calls() == 2
    assert old_request.extract_tool_calls() == 1
    assert sys.modules[PARSER] is source.modules[PARSER]


def test_failed_registration_does_not_leak(source):
    source.paths[PARSER].write_text(
        PARSER_SOURCE.replace('"audit_reload"', '"leak"')
        + '\nraise RuntimeError("bad edit")\n'
    )
    with pytest.raises(RuntimeError):
        reload.reload_now()
    assert "leak" not in ToolParserManager.lazy_parsers


def test_new_parser_dependency_requires_restart(source):
    source.paths[PARSER].write_text(
        PARSER_SOURCE + "\nimport vllm.tool_parsers._not_loaded_yet\n"
    )
    with pytest.raises(ValueError, match="requires a restart"):
        reload.reload_now()
    assert ToolParserManager.tool_parsers["audit_reload"] is source.cls


def test_concurrent_reload_serializes_generations(source):
    with ThreadPoolExecutor(max_workers=4) as pool:
        versions = list(pool.map(lambda _: reload.reload_now()["version"], range(4)))
    assert sorted(versions) == [1, 2, 3, 4]


def make_app(source, template_path):
    from vllm.entrypoints.serve.toolparser.api_router import attach_router

    app = FastAPI()
    app.state.args = SimpleNamespace(chat_template=template_path)
    wrapper = type(
        "Wrapper",
        (_WrappedParser,),
        {
            "tool_parser_cls": source.cls,
            "reasoning_parser_cls": None,
        },
    )
    app.state.openai_serving_chat = SimpleNamespace(
        tool_parser=source.cls, parser_cls=wrapper, chat_template="old template"
    )
    attach_router(app)
    return app


def test_endpoint_publishes_complete_generation_for_one_app(source, tmp_path):
    template = tmp_path / "chat.jinja"
    template.write_text("new {{ messages }}")
    app = make_app(source, str(template))
    other = make_app(source, "{{ untouched }}")
    holder = app.state.openai_serving_chat
    old_wrapper = holder.parser_cls
    source.paths[HELPER].write_text("VALUE = 2\n")
    result = TestClient(app).post("/v1/internal/reload")
    assert result.status_code == 200
    assert result.json()["chat_template_holders"] == 1
    assert holder.chat_template == "new {{ messages }}"
    assert holder.tool_parser is holder.parser_cls.tool_parser_cls
    assert object.__new__(holder.tool_parser).extract_tool_calls() == 2
    assert old_wrapper.tool_parser_cls is source.cls
    assert other.state.openai_serving_chat.chat_template == "old template"
    assert other.state.openai_serving_chat.tool_parser is source.cls


@pytest.mark.parametrize("bad_template", ["{% if %}", None])
def test_template_failure_rolls_back_parser_and_holders(source, tmp_path, bad_template):
    template = tmp_path / "chat.jinja"
    if bad_template is not None:
        template.write_text(bad_template)
    app = make_app(source, str(template))
    holder = app.state.openai_serving_chat
    before = dict(vars(holder))
    registry = ToolParserManager.tool_parsers
    source.paths[HELPER].write_text("VALUE = 2\n")
    result = TestClient(app).post("/v1/internal/reload")
    assert result.status_code == 409
    assert vars(holder) == before
    assert ToolParserManager.tool_parsers is registry
    assert reload.current_version() == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_reload_endpoint_requires_dev_mode(monkeypatch, enabled):
    from vllm.entrypoints.openai.api_server import build_app
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", str(int(enabled)))
    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    app = build_app(args, supported_tasks=("generate",))
    # Exercise routing rather than FastAPI's internal included-router objects.
    response = TestClient(app).get("/v1/internal/reload")
    assert response.status_code == (405 if enabled else 404)


def test_prepare_existing_parser_implementations():
    from vllm.tool_parsers import ToolParser

    baseline = {
        name: ToolParserManager.get_tool_parser(name)
        for name in ("qwen3_coder", "deepseek_v4")
    }
    registry = dict(ToolParserManager.tool_parsers)
    with reload.reload_lock:
        prepared = reload.prepare_reload()
    assert ToolParserManager.tool_parsers == registry
    for name, old in baseline.items():
        candidate = prepared.manager.tool_parsers[name]
        assert candidate is not old
        assert issubclass(candidate, ToolParser)
