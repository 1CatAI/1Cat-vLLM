# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tools.config_inventory import python_references, typed_declarations
from tools.pre_commit.check_env_registration import native_reads


def test_records_aliases_helpers_and_dynamic_readers_without_evaluation():
    source = """
import vllm.envs as flags
from os import getenv as read
KEY = "VLLM_SM70_EXAMPLE"
class Layer:
    def forward(self):
        return (read(KEY), flags.VLLM_SM70_EXAMPLE,
                _config.registered("VLLM_SM70_EXAMPLE"),
                getattr(flags, variable_name), read("TM_GEMM_TUNE"))
"""
    rows = python_references(source)
    assert {row["kind"] for row in rows} == {"raw", "registered", "getter"}
    assert len(rows) == 5
    assert all(row["scope"] == "Layer.forward" for row in rows)
    assert sum(row["name"] is None for row in rows) == 1
    assert any(row["name"] == "TM_GEMM_TUNE" for row in rows)


def test_typed_sources_are_declarations_not_environment_reads():
    source = """
aliases = {"enabled": "VLLM_SM70_EXAMPLE"}
reverse_aliases = {"VLLM_SM70_SECOND": "second"}
NATIVE_FIELDS = (("tune", "TM_GEMM_TUNE", ("awq",), False),)
"""
    assert not python_references(source)
    assert set(typed_declarations(source)) == {
        "VLLM_SM70_EXAMPLE",
        "VLLM_SM70_SECOND",
        "TM_GEMM_TUNE",
    }


def test_non_vllm_native_settings_are_included():
    assert native_reads('std::getenv("TM_GEMM_CACHE_SUMMARY");') == [
        ("TM_GEMM_CACHE_SUMMARY", 1)
    ]


def test_parser_and_historical_alias_tuples_retain_one_owner():
    declarations = typed_declarations("""
bindings = {"pipeline": ("VLLM_SM70_PIPELINE", "first_ne0", True)}
legacy_aliases = {"scalar": ("VLLM_SM70_SCALAR", "VLLM_SM70_OLD_SCALAR")}
""")
    assert {name: rows[0]["field"] for name, rows in declarations.items()} == {
        "VLLM_SM70_PIPELINE": "pipeline",
        "VLLM_SM70_SCALAR": "scalar",
        "VLLM_SM70_OLD_SCALAR": "scalar",
    }


def test_direct_registered_imports_are_visible():
    assert (
        python_references("from vllm.envs import VLLM_SM70_EXAMPLE as flag")[0]["name"]
        == "VLLM_SM70_EXAMPLE"
    )
