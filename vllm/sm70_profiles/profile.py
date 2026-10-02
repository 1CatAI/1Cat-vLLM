# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

PROFILE_NAME = "qwen38_27b_nvfp4_dflash2"
_PROFILE_PATH = Path(__file__).with_name(f"{PROFILE_NAME}.json")


def load_profile(name: str = PROFILE_NAME) -> dict:
    if name != PROFILE_NAME:
        raise ValueError(f"Unknown SM70 profile: {name}")
    return json.loads(_PROFILE_PATH.read_text())


def profile_argv(name: str = PROFILE_NAME, *, draft: str | None = None) -> list[str]:
    argv = []
    args = load_profile(name)["args"]
    if draft is not None:
        args["speculative_config"]["model"] = draft
        args["speculative_config"].pop("revision", None)
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            argv.append(flag if value else "--no-" + flag[2:])
        else:
            argv.extend(
                [
                    flag,
                    json.dumps(value, separators=(",", ":"))
                    if isinstance(value, (dict, list))
                    else str(value),
                ]
            )
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m vllm.sm70_profiles")
    parser.add_argument("action", choices=("show", "argv"))
    parser.add_argument("name", nargs="?", default=PROFILE_NAME)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--argv-lines", action="store_true")
    parser.add_argument("--draft", type=Path, help="Use a local draft checkpoint")
    ns = parser.parse_args()
    if ns.draft is not None and not ns.draft.is_dir():
        parser.error("--draft must name a local checkpoint directory")
    if ns.action == "show":
        print(
            json.dumps(load_profile(ns.name), indent=2)
            if ns.json
            else json.dumps(load_profile(ns.name))
        )
        return
    argv = profile_argv(ns.name, draft=str(ns.draft) if ns.draft is not None else None)
    if ns.argv_lines:
        print("\n".join(argv))
    else:
        print(json.dumps(argv) if ns.json else shlex.join(argv))
