# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import pytest

from tools.generate_env_reference import OUTPUT, render
from tools.pre_commit.check_env_metadata import ENVS_FILE, read_metadata
from vllm.envs_metadata import env_var


def source(*, getter='lambda: os.getenv("VLLM_EXAMPLE", "0")', **changes):
    fields = dict(
        description="Example of a documented legacy setting.",
        category="configuration",
        declared_default="'0'",
        effective_default="'0'",
        automatic_conditions=(),
        acceleration_paths=(),
        user_visible=True,
    )
    fields.update(changes)
    arguments = ", ".join(f"{key}={value!r}" for key, value in fields.items())
    return (
        'environment_variables: dict = {"VLLM_EXAMPLE": '
        f"env_var({getter}, {arguments})}}"
    )


def test_plain_new_registration_is_rejected():
    _, errors = read_metadata('environment_variables: dict = {"VLLM_NEW": lambda: 1}')
    assert errors and "complete metadata" in errors[0]


@pytest.mark.parametrize(
    "change",
    [
        {"description": ""},
        {"category": "unknown"},
        {"category": []},
        {"automatic_conditions": ("",)},
        {"acceleration_paths": ("route", "route")},
    ],
)
def test_incomplete_metadata_is_rejected(change):
    _, errors = read_metadata(source(**change))
    assert errors


def test_metadata_cannot_execute_expressions():
    text = source().replace("category='configuration'", "category=run_gpu_code()")
    _, errors = read_metadata(text)
    assert errors and "literals" in errors[0]


def test_inspection_and_render_do_not_execute_getters(tmp_path):
    target = tmp_path / "should_not_exist"
    text = source(getter=f"lambda: open({str(target)!r}, 'w').write('secret')")
    metadata, errors = read_metadata(text)
    assert not errors
    assert "VLLM_EXAMPLE" in render(metadata)
    assert not target.exists()


def test_stale_effective_default_is_rejected():
    _, errors = read_metadata(source(getter='lambda: os.getenv("VLLM_EXAMPLE", "1")'))
    assert errors and "getter default" in errors[0]
    _, errors = read_metadata(
        source(getter="lambda: 10", declared_default="10", effective_default="100")
    )
    assert errors and "getter default" in errors[0]


def test_unexplained_declared_default_mismatch_is_rejected():
    _, errors = read_metadata(source(declared_default="'1'"))
    assert errors and "without explanation" in errors[0]


def test_metadata_wrapper_only_delegates_when_read():
    calls = []
    sentinel = object()

    def getter():
        calls.append("read")
        return sentinel

    variable = env_var(
        getter,
        description="A lazy test getter.",
        category="configuration",
        declared_default="None",
        effective_default="None",
        automatic_conditions=(),
        acceleration_paths=(),
        user_visible=True,
    )
    assert not calls
    assert variable() is sentinel and calls == ["read"]


def test_all_current_registrations_and_generated_reference_are_complete():
    metadata, errors = read_metadata(ENVS_FILE.read_text())
    assert not errors
    assert OUTPUT.read_text() == render(metadata)
    assert metadata["VLLM_SM70_NVFP4_QPN2"]["category"] == "deprecated"
    assert "qualified DFlash" in metadata["VLLM_SM70_NVFP4_QPN2"]["effective_default"]
    assert metadata["VLLM_SM70_DUMP_GDN_CORE_DIR"]["category"] == "debug"
    assert metadata["VLLM_SM70_DUMP_GDN_CORE_DIR"]["automatic_conditions"]


def test_internal_metadata_is_kept_out_of_public_reference():
    metadata, errors = read_metadata(source(user_visible=False))
    assert not errors
    assert "VLLM_EXAMPLE" not in render(metadata)
    assert "VLLM_EXAMPLE" in render(metadata, include_internal=True)


def test_non_boolean_visibility_is_rejected():
    _, errors = read_metadata(source(user_visible="false"))
    assert errors and "literal boolean" in errors[0]
