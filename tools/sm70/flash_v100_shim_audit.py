# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in pytest plugin: reject ownerless patches and observe callable use.

Usage: pytest -p tools.sm70.flash_v100_shim_audit --shim-audit=report.json ...
The report separates callable patches reached from production from patches
that were never consumed. No replacement wrappers can make identity tests pass.
"""

import dis
import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from tools.sm70.flash_v100_trace import strict_shim


def pytest_addoption(parser):
    parser.addoption("--shim-audit", type=Path)
    parser.addoption("--require-shim-use", action="store_true")


def pytest_sessionfinish(session, exitstatus):
    if session.config.getoption("--require-shim-use"):
        report = getattr(session.config, "_flash_shim_report", {})
        missing = [
            name
            for name, uses in report.items()
            if not uses["called_from"]
            and (uses["requires_call"] or not uses["read_from"])
        ]
        if not report or missing:
            session.config.get_terminal_writer().line(
                f"Unconsumed shim patches: {missing}"
            )
            session.exitstatus = 1


def production(frame):
    return frame.f_globals.get("__name__", "").startswith("vllm.") and (
        frame.f_code.co_name not in ("__getattr__", "__setattr__", "__dir__")
    )


@pytest.fixture(autouse=True)
def audit_legacy_patches(request):
    report_path = request.config.getoption("--shim-audit")
    if report_path is None:
        yield
        return
    with strict_shim() as legacy:
        original_class = type(legacy)
        targets = {}
        c_targets = []
        globals_targets = {}
        last_read = {}
        instructions = {}
        report = getattr(request.config, "_flash_shim_report", {})
        request.config._flash_shim_report = report

        class ObservedModule(original_class):
            def __setattr__(self, name, value):
                original = getattr(self, name, None)
                super().__setattr__(name, value)
                if name.startswith("__"):
                    return
                record = report.setdefault(
                    name,
                    dict(patched_in=[], called_from=[], read_from=[], requires_call=[]),
                )
                record["patched_in"].append(request.node.nodeid)
                if inspect.isfunction(original) and original.__name__ == name:
                    record["requires_call"].append(request.node.nodeid)
                globals_targets[name] = value
                if callable(original):
                    code = getattr(value, "__code__", None)
                    if code is not None:
                        targets.setdefault(code, set()).add(name)
                    else:
                        c_targets.append((name, value))

        def observe(frame, event, arg):
            if event == "call":
                names = targets.get(frame.f_code, set())
                name = next(iter(names)) if len(names) == 1 else None
                if len(names) > 1:
                    # Same-code closures cannot be distinguished by code identity.
                    # Require the caller's observed lookup; never credit an alias.
                    name = last_read.get(id(frame.f_back))
                    if name not in names:
                        name = None
                if name is None and frame.f_code.co_name == "__call__":
                    name = next(
                        (n for n, t in c_targets if frame.f_locals.get("self") is t),
                        None,
                    )
                caller = frame.f_back
                if name is not None and production(caller):
                    report[name]["called_from"].append(
                        caller.f_globals["__name__"] + ":" + caller.f_code.co_name
                    )
            if event == "c_call" and production(frame):
                for name, target in c_targets:
                    if arg is target:
                        report[name]["called_from"].append(
                            frame.f_globals["__name__"] + ":" + frame.f_code.co_name
                        )

        def reads(frame, event, arg):
            module_name = frame.f_globals.get("__name__", "")
            if event == "return":
                last_read.pop(id(frame), None)
            if event == "call" and (
                not production(frame)
                or not any(
                    name in globals_targets
                    and frame.f_globals.get(name) is globals_targets[name]
                    for name in frame.f_code.co_names
                )
            ):
                return None
            frame.f_trace_opcodes = True
            if event == "opcode":
                if frame.f_code not in instructions:
                    instructions[frame.f_code] = {
                        i.offset: i for i in dis.get_instructions(frame.f_code)
                    }
                code = instructions[frame.f_code]
                instruction = code.get(frame.f_lasti)
                if instruction and instruction.opname == "LOAD_GLOBAL":
                    name = instruction.argval
                    if (
                        name in globals_targets
                        and frame.f_globals.get(name) is globals_targets[name]
                    ):
                        last_read[id(frame)] = name
                        report[name]["read_from"].append(
                            module_name + ":" + frame.f_code.co_name
                        )
            return reads

        class ObservedOwner(types.ModuleType):
            def __getattribute__(self, name):
                value = super().__getattribute__(name)
                if name in globals_targets and value is globals_targets[name]:
                    caller = sys._getframe(1)
                    module_name = caller.f_globals.get("__name__", "")
                    if production(caller):
                        last_read[id(caller)] = name
                        report[name]["read_from"].append(
                            module_name + ":" + caller.f_code.co_name
                        )
                return value

        owners = [(module, type(module)) for module in legacy._package.SUBMODULES]
        for module, _ in owners:
            module.__class__ = ObservedOwner

        legacy.__class__ = ObservedModule
        previous = sys.getprofile()
        previous_trace = sys.gettrace()
        frame = sys._getframe()
        previous_opcodes = frame.f_trace_opcodes
        frame.f_trace_opcodes = True  # CPython 3.12 requires this before settrace.
        sys.setprofile(observe)
        sys.settrace(reads)
        try:
            yield
        finally:
            sys.setprofile(previous)
            sys.settrace(previous_trace)
            frame.f_trace_opcodes = previous_opcodes
            legacy.__class__ = original_class
            for module, cls in owners:
                module.__class__ = cls
            report_path.write_text(
                json.dumps(
                    {
                        k: {
                            field: sorted(set(values))
                            for field, values in record.items()
                        }
                        for k, record in sorted(report.items())
                    },
                    indent=2,
                )
                + "\n"
            )
