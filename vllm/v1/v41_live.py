# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_LIVE_FILE=path (diagnostic, default off): "key value" lines that
override V4.1 PP scheduling knobs while the server runs, re-read when the
file changes (A/B without a restart). Unset or missing keys use the
defaults the callers pass."""

import os
import time

import vllm.envs as envs

_PATH = envs.VLLM_V41_LIVE_FILE
_STATE: dict = {"checked": 0.0, "mtime": None, "values": {}}


def get(key: str, default: int) -> int:
    if not _PATH:
        return default
    now = time.monotonic()
    if now - _STATE["checked"] > 0.5:
        _STATE["checked"] = now
        try:
            mtime = os.stat(_PATH).st_mtime_ns
        except OSError:
            mtime = 0
        if mtime != _STATE["mtime"]:
            values = {}
            if mtime:
                with open(_PATH) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
                            values[parts[0]] = int(parts[1])
            _STATE.update(mtime=mtime, values=values)
    return _STATE["values"].get(key, default)
