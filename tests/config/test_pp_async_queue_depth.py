# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.config import SchedulerConfig, VllmConfig

pytestmark = pytest.mark.cpu_test


def make_config(
    depth=None,
    *,
    pp_size=7,
    async_scheduling=True,
    backend="mp",
    v2=True,
    draft=False,
    scheduler_cls=None,
    no_pp_depth=None,
):
    # Exercise the production resolver and route validation without loading
    # model metadata, CUDA, or a distributed executor.
    return SimpleNamespace(
        scheduler_config=SchedulerConfig.default_factory(
            async_scheduling=async_scheduling,
            pp_async_queue_depth=depth,
            scheduler_cls=scheduler_cls,
            sm70_async_queue_depth=no_pp_depth,
        ),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=pp_size,
            distributed_executor_backend=backend,
        ),
        use_v2_model_runner=v2,
        is_speculative_draft=draft,
    )


def resolved_depth(config):
    return VllmConfig.max_concurrent_batches.fget(config)


@pytest.mark.parametrize("pp_size", [2, 4, 7])
@pytest.mark.parametrize("async_scheduling", [False, True])
def test_default_preserves_pipeline_depth(pp_size, async_scheduling):
    config = make_config(pp_size=pp_size, async_scheduling=async_scheduling)
    VllmConfig._validate_pp_async_queue_depth(config)
    assert resolved_depth(config) == pp_size


@pytest.mark.parametrize("depth", [2, 4, 7])
def test_explicit_depth_controls_executor_and_feedback_bound(depth):
    config = make_config(depth)
    VllmConfig._validate_pp_async_queue_depth(config)
    assert resolved_depth(config) == depth


@pytest.mark.parametrize("depth", [0, 1, -1])
def test_depth_one_and_nonpositive_are_rejected(depth):
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        make_config(depth)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pp_size": 1}, "between 2"),
        ({"pp_size": 2}, "between 2"),
        ({"async_scheduling": False}, "requires async scheduling"),
        ({"backend": "ray"}, "multiprocessing executor"),
        ({"backend": "external_launcher"}, "multiprocessing executor"),
        ({"scheduler_cls": "custom.Scheduler"}, "custom scheduler"),
        ({"v2": False}, "Model Runner V2"),
    ],
)
def test_unsupported_explicit_routes_fail_closed(kwargs, message):
    config = make_config(4, **kwargs)
    with pytest.raises(ValueError, match=message):
        VllmConfig._validate_pp_async_queue_depth(config)


def test_depth_cannot_exceed_physical_pipeline():
    with pytest.raises(ValueError, match="between 2"):
        VllmConfig._validate_pp_async_queue_depth(make_config(8))


def test_internal_mtp_config_inherits_validated_target_depth():
    # Qwen MTP builds a derived VllmConfig with is_speculative_draft=True;
    # its architecture is not the target's default V2 architecture.
    config = make_config(4, v2=False, draft=True)
    VllmConfig._validate_pp_async_queue_depth(config)
    assert resolved_depth(config) == 4


@pytest.mark.parametrize("override,expected", [(None, 2), (4, 4)])
def test_no_pp_async_depth_is_unchanged(override, expected):
    config = make_config(pp_size=1, no_pp_depth=override)
    VllmConfig._validate_pp_async_queue_depth(config)
    assert resolved_depth(config) == expected


def test_no_pp_synchronous_depth_is_unchanged():
    config = make_config(pp_size=1, async_scheduling=False)
    assert resolved_depth(config) == 1


def test_cli_round_trips_explicit_depth(tmp_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(
        parser.parse_args(["--model", str(tmp_path), "--pp-async-queue-depth", "4"])
    )
    assert args.pp_async_queue_depth == 4
