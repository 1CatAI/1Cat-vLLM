# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend documentation follows capability predicates without importing CUDA."""

import ast
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "tools/pre_commit/generate_attention_backend_docs.py"
)
spec = importlib.util.spec_from_file_location("attention_backend_docs", SCRIPT)
assert spec is not None and spec.loader is not None
docs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(docs)


def parse(source, path=None):
    tree = ast.parse(source)
    node = docs.find_class_in_ast(tree, "Backend")
    return docs.parse_compute_capability(node, tree, path)


def test_imported_nullable_capability_predicate(tmp_path):
    helper = tmp_path / "capabilities.py"
    helper.write_text("""
raise RuntimeError("Documentation must never execute this module")
CAPS = frozenset({(8, 9), (7, 0), (8, 6), (7, 2)})
def supports(device):
    return device is not None and (device.major, device.minor) in CAPS
""")
    source = """
class Backend:
    @staticmethod
    def supports_compute_capability(capability):
        from .capabilities import supports as allowed
        return allowed(capability)
"""
    assert parse(source, tmp_path / "backend.py") == "7.0, 7.2, 8.6, 8.9"
    helper.write_text(helper.read_text().replace("(8, 9)", "(9, 0)"))
    assert parse(source, tmp_path / "backend.py") == "7.0, 7.2, 8.6, 9.0"


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("(capability.major, capability.minor) in {(8, 6), (7, 0)}", "7.0, 8.6"),
        ("capability >= DeviceCapability(8, 0)", "≥8.0"),
        ("capability.major == 7", "7.x"),
        ("True", "Any"),
    ],
)
def test_existing_and_literal_predicates(expression, expected):
    assert (
        parse(
            "class Backend:\n    def supports_compute_capability(capability):\n"
            f"        return {expression}\n"
        )
        == expected
    )


@pytest.mark.parametrize(
    "helper",
    [
        "def allowed(x):\n    return allowed(x)\n",
        "def allowed(x):\n    return (x.major, x.minor) in {(7, 0)} or x.major == 9\n",
        "def allowed(x):\n    return (x.major, x.minor) in build_capabilities()\n",
    ],
)
def test_unresolved_predicate_is_not_documented_as_exact(helper):
    source = (
        helper + "class Backend:\n    def supports_compute_capability(capability):\n"
        "        return allowed(capability)\n"
    )
    assert parse(source) == "Any"


def test_real_portable_backend_uses_exact_capabilities():
    path = docs.REPO_ROOT / "vllm/models/glm5next/sm70/sparse.py"
    tree = ast.parse(path.read_text())
    node = docs.find_class_in_ast(tree, "Glm5NextSM70SparseBackend")
    assert docs.parse_compute_capability(node, tree, path) == "7.0, 7.2, 8.6, 8.9"
    assert docs.is_relevant_file("vllm/models/deepseek_v4/common/portable_sparse.py")
