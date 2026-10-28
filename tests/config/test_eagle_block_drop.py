# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage for disable_eagle_block_drop (vllm-project/vllm#53388 backport).

Binds methods on a lightweight stub so this does not need a target ModelConfig.
"""

from vllm.config.speculative import SpeculativeConfig


def _cfg(method: str, *, disable_eagle_block_drop: bool = False):
    cfg = SpeculativeConfig.__new__(SpeculativeConfig)
    cfg.method = method
    cfg.disable_eagle_block_drop = disable_eagle_block_drop
    return cfg


def test_mtp_keeps_drop_by_default():
    cfg = _cfg("mtp")
    assert cfg.use_eagle() is True
    assert cfg.use_eagle_kv_cache() is True


def test_mtp_opt_out_skips_drop():
    cfg = _cfg("mtp", disable_eagle_block_drop=True)
    assert cfg.use_eagle() is True
    assert cfg.use_eagle_kv_cache() is False


def test_dflash_still_skips_drop_without_the_flag():
    cfg = _cfg("dflash")
    assert cfg.use_eagle_kv_cache() is False
