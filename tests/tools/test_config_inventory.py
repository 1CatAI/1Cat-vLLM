# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tools.config_inventory import python_references, typed_declarations
from tools.pre_commit.check_env_registration import native_reads


def test_records_aliases_helpers_and_dynamic_readers_without_evaluation():
    source = """
import vllm.envs as flags
from os import getenv as read
from vllm.v1.attention.backends.flash_v100 import config as _config
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


def test_native_constant_array_records_each_compatible_consumer():
    source = (
        'const char* names[] = {"PREFIX_TORCH_EXACT_TAIL", "TM_TEST"};\n'
        "std::getenv(names[index]);"
    )
    assert native_reads(source) == [("PREFIX_TORCH_EXACT_TAIL", 2), ("TM_TEST", 2)]
    assert "PREFIX_TORCH_EXACT_TAIL" in typed_declarations(
        'bindings = {"exact": ("PREFIX_TORCH_EXACT_TAIL", "present", None)}'
    )


def test_dynamic_native_reads_do_not_disappear():
    source = "std::getenv(policy_name(field));\nstd::getenv(dynamic_name.c_str());"
    assert native_reads(source, include_unresolved=True) == [(None, 1), (None, 2)]
    assert native_reads(source) == []


def test_bound_native_policy_references_are_separate_from_raw_reads():
    from tools.config_inventory import native_policy_references

    source = """// PolicyField::enabled
const char* message = "PolicyField::enabled";
return policy_atoi(vllm::sm70::PolicyField::enabled, 1);
"""
    assert native_reads(source, include_unresolved=True) == []
    rows = native_policy_references(source, {"PolicyField::enabled": "VLLM_EXAMPLE"})
    assert rows == [
        dict(
            name="VLLM_EXAMPLE",
            line=3,
            kind="native_bound",
            scope="",
            binding="PolicyField::enabled",
        )
    ]


def test_native_policy_inventory_uses_shipped_declarations(tmp_path):
    from tools.config_inventory import native_policy_fields

    csrc = tmp_path / "csrc"
    csrc.mkdir()
    (csrc / "sm70_policy_fields.inc").write_text(
        'SM70_POLICY_FIELD(example, "VLLM_EXAMPLE", true)\n'
    )
    include = tmp_path / "flash-attention-v100" / "include"
    include.mkdir(parents=True)
    (include / "flash_v100_policy.h").write_text(
        'enum class Field { test, count };\nconst char* names[] = {"PREFIX_TEST"};\n'
    )
    assert native_policy_fields(tmp_path) == {
        "PolicyField::example": "VLLM_EXAMPLE",
        "flash_v100::policy::Field::test": "PREFIX_TEST",
    }


def test_forwarded_getters_and_assignments_cannot_hide_reads():
    source = """
import os as process
from vllm import envs as flags
read = process.getenv
second_read = read
KEY = "VLLM_SM70_TEST"
def enabled(name):
    return second_read(name) == "1"
def forward():
    return enabled(KEY), flags.VLLM_SM70_TEST
"""
    rows = python_references(source)
    consumers = [row for row in rows if row["scope"] == "forward"]
    assert len(consumers) == 2
    assert {row["name"] for row in consumers} == {"VLLM_SM70_TEST"}
    assert {row["kind"] for row in consumers} == {"registered", "getter"}
    assert any(row["name"] is None and row["scope"] == "enabled" for row in rows)


def test_frozen_policy_raw_method_is_not_an_environment_getter():
    source = """
def report(policy):
    return policy.raw("VLLM_SM70_TEST")
"""
    assert python_references(source) == []


def test_argument_shadowing_an_import_is_not_a_process_read():
    source = """
import os
from vllm import envs

def read_fixture(os, envs):
    return os.getenv("VLLM_SM70_TEST"), envs.VLLM_SM70_TEST
"""
    assert python_references(source) == []


def test_wrapped_reader_import_alias_uses_its_actual_identity():
    import ast

    from tools.pre_commit.environment_readers import forwarding_getters

    getters = forwarding_getters(
        ast.parse("""
import os

def query(name):
    return os.environ.get(name)
"""),
        "package.flags",
    )
    source = """
from package.flags import query as enabled
value = enabled("TM_GEMM_TUNE")
"""
    assert python_references(source, getters=getters)[0]["name"] == "TM_GEMM_TUNE"


def test_reader_assignment_is_scoped_and_shadowing_is_respected():
    source = """
import os
KEY = "VLLM_SM70_TEST"
def other():
    os = object()
    return os.getenv(KEY)
def forward():
    read = os.getenv
    return read(KEY)
def dynamic(KEY):
    return os.getenv(KEY)
"""
    rows = python_references(source)
    assert [(r["scope"], r["name"]) for r in rows] == [
        ("forward", "VLLM_SM70_TEST"),
        ("dynamic", None),
    ]


def test_nested_helper_does_not_make_outer_function_a_reader():
    import ast

    from tools.pre_commit.environment_readers import forwarding_getters

    getters = forwarding_getters(
        ast.parse("""
import os
def outer(name):
    def inner(name):
        return os.getenv(name)
    return name
"""),
        "mod",
    )
    assert getters == {"mod.outer.inner": 0}


def test_ownership_follows_types_and_reexports_not_alias_words():
    from tools.config_ownership import ConfigOwnership

    sources = {
        "vllm/config/vllm.py": """
from vllm.config.public import Kernel
class VllmConfig:
    kernel_config: Kernel
""",
        "vllm/config/public.py": "from vllm.config.kernel import Kernel",
        "vllm/config/kernel.py": """
class Kernel:
    weight_layout: bool = False
    aliases = {"weight_layout": "VLLM_SM70_MTP_WEIGHT_LAYOUT"}
""",
    }
    declaration = typed_declarations(sources["vllm/config/kernel.py"])
    owners = ConfigOwnership(sources).owners(
        "vllm/config/kernel.py",
        declaration["VLLM_SM70_MTP_WEIGHT_LAYOUT"][0],
        "VLLM_SM70_MTP_WEIGHT_LAYOUT",
    )
    assert [(owner["owner"], owner["field"]) for owner in owners] == [
        ("kernel_config", "weight_layout")
    ]


def test_direct_initialized_assignment_has_a_field_and_dynamic_key_does_not():
    source = """
import os
class Policy:
    def resolve(self):
        raw = os.getenv("VLLM_SM70_TEST", "0")
        self.enabled = raw == "1"
"""
    assert typed_declarations(source)["VLLM_SM70_TEST"][0]["field"] == "enabled"
    source = source.replace('"VLLM_SM70_TEST"', "name")
    assert not typed_declarations(source)


def test_real_native_declarations_do_not_attribute_fp8_to_awq_or_diagnostics():
    from pathlib import Path

    from tools.config_ownership import ConfigOwnership

    sources = {str(path): path.read_text() for path in Path("vllm/config").glob("*.py")}
    source = "vllm/config/sm70_native.py"
    alias = "VLLM_SM70_FP8_SAFE_FAST_SELECTOR"
    entry = typed_declarations(sources[source])[alias][0]
    owners = ConfigOwnership(sources).owners(source, entry, alias)
    assert {owner["owner"] for owner in owners} == {
        "kernel_config.sm70_fp8.native",
        "kernel_config.sm70_moe.fp8.native",
    }


def test_closure_rejects_new_forward_and_dynamic_reader_inside_config_directory():
    from tools.config_boundaries import consumer_lifecycle
    from tools.config_inventory import closure_errors

    site = dict(
        path="vllm/config/example.py", scope="Policy.forward", line=9, kind="raw"
    )
    site["lifecycle"] = consumer_lifecycle("VLLM_SM70_NEW", site, None)
    assert site["lifecycle"] == "unclassified"
    inventory = dict(
        parameters={"VLLM_SM70_NEW": dict(owners=[], boundary=None, consumers=[site])},
        unresolved_dynamic_readers=[dict(site, input_domain=None)],
    )
    errors = closure_errors(inventory)
    assert len(errors) == 3
    assert any("no typed owner" in error for error in errors)
    assert any("input domain" in error for error in errors)
    assert any("unclassified legacy consumer" in error for error in errors)


def test_native_copied_helpers_must_be_unreachable_from_registered_entry():
    from tools.config_inventory import native_retained_lifecycles, native_scopes

    source = """
namespace {
bool old_flag() { return std::getenv("VLLM_SM70_TEST") != nullptr; }
void entry() { kernel(); }
}
TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
 ops.impl("entry", &entry);
}
"""
    scopes = native_scopes(source)
    assert "old_flag" in native_retained_lifecycles(source, scopes)
    source = source.replace("kernel();", "old_flag(); kernel();")
    assert "old_flag" not in native_retained_lifecycles(source, native_scopes(source))


def test_native_extension_and_standalone_compile_branches_remain_distinct():
    from tools.config_inventory import native_compile_guards

    source = """#if defined(PREFIX_TORCH_EXTENSION)
explicit_policy();
#else
std::getenv("PREFIX_TEST");
#endif
"""
    guards = native_compile_guards(source)
    assert guards[2] == ("defined(PREFIX_TORCH_EXTENSION)",)
    assert guards[4] == ("!(defined(PREFIX_TORCH_EXTENSION))",)


def test_relative_import_inside_function_keeps_module_identity():
    rows = python_references(
        """
def invoke():
    from .flags import read
    return read("VLLM_SM70_TEST")
""",
        module="package.consumer",
        getters={"package.flags.read": 0},
    )
    assert len(rows) == 1 and rows[0]["name"] == "VLLM_SM70_TEST"
