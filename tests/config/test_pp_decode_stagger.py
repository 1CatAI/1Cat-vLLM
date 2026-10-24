# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import SchedulerConfig, VllmConfig

pytestmark = pytest.mark.cpu_test


def make_config(
    *,
    enabled=True,
    pp=7,
    depth=2,
    async_=True,
    v2=True,
    backend="mp",
    custom=None,
    draft=False,
):
    return SimpleNamespace(
        scheduler_config=SchedulerConfig.default_factory(
            pp_decode_stagger=enabled,
            async_scheduling=async_,
            scheduler_cls=custom,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp,
            distributed_executor_backend=backend,
        ),
        max_concurrent_batches=depth,
        use_v2_model_runner=v2,
        is_speculative_draft=draft,
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_supported_configuration(enabled):
    VllmConfig._validate_pp_decode_stagger(make_config(enabled=enabled))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pp": 1},
        {"depth": 3},
        {"async_": False},
        {"v2": False},
        {"backend": "ray"},
        {"custom": "custom.Scheduler"},
    ],
)
def test_explicit_unsupported_target_is_rejected_but_default_is_unchanged(kwargs):
    with pytest.raises(ValueError, match="pp_decode_stagger requires"):
        VllmConfig._validate_pp_decode_stagger(make_config(**kwargs))
    VllmConfig._validate_pp_decode_stagger(make_config(enabled=False, **kwargs))


def test_internal_draft_config_does_not_run_target_scheduler_validation():
    VllmConfig._validate_pp_decode_stagger(
        make_config(draft=True, v2=False, pp=1, depth=1)
    )


@pytest.mark.parametrize(
    "flags,expected",
    [
        ([], False),
        (["--pp-decode-stagger"], True),
        (["--pp-decode-stagger", "--no-pp-decode-stagger"], False),
    ],
)
def test_cli_and_engine_args_round_trip(tmp_path, flags, expected):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(["--model", str(tmp_path), *flags])
    )
    assert args.pp_decode_stagger is expected
