# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm import envs

ALIASES = {
    "VLLM_SM70_GREEDY_TOKEN_FASTPATH_TRACE": "trace",
    "VLLM_SM70_DFLASH2_QPN8_RERANK_SHADOW": "selector",
    "VLLM_SM70_PROFILE_TRACE": "trace",
    "VLLM_SM70_DECODE_TILE_PROFILE": "trace",
    "VLLM_SM70_MTP_PROFILE": "mtp",
    "VLLM_SM70_DECODE_EVENT_TRACE": "events",
    "VLLM_FLASH_V100_ROUTE_SUMMARY": "routing",
}


@pytest.fixture(autouse=True)
def clear_debug(monkeypatch):
    envs.disable_envs_cache()
    for name in (*ALIASES, "VLLM_SM70_DEBUG"):
        monkeypatch.delenv(name, raising=False)
    yield
    envs.disable_envs_cache()


@pytest.mark.parametrize("name", ALIASES)
@pytest.mark.parametrize("value", ("0", "1", "2"))
def test_legacy_debug_parser_is_preserved_and_warns(name, value, monkeypatch):
    monkeypatch.setenv(name, value)
    with pytest.warns(FutureWarning, match="compatibility release"):
        assert getattr(envs, name) == bool(int(value))


@pytest.mark.parametrize(
    "channels",
    (
        "",
        "trace",
        "mtp",
        "events",
        "routing",
        "selector",
        "trace,mtp",
        "trace,selector",
    ),
)
def test_channels_enable_only_requested_diagnostics(channels, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_DEBUG", channels)
    assert set(channels.split(",")) - {""} == envs.VLLM_SM70_DEBUG
    for name, channel in ALIASES.items():
        assert getattr(envs, name) == (channel in envs.VLLM_SM70_DEBUG)


def test_explicit_unified_off_wins_over_old_on(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_MTP_PROFILE", "1")
    monkeypatch.setenv("VLLM_SM70_DEBUG", "")
    with pytest.warns(FutureWarning):
        assert not envs.VLLM_SM70_MTP_PROFILE


def test_unknown_channel_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_SM70_DEBUG", "unknown")
    with pytest.raises(ValueError):
        _ = envs.VLLM_SM70_DEBUG
