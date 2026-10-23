# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Preserve teacher conditions when a comparison report becomes a reference."""

import hashlib
import json


def prefix_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest()


def teacher_conditions(reference, positions):
    previous = {
        row["key"]: row for row in reference.get("teacher_forcing", {}).get("rows", [])
    }
    for row in reference["rows"]:
        if len(row["output_token_ids"]) <= positions:
            raise RuntimeError("Teacher continuation is too short")
        for position in range(positions):
            key = f"{row['id']}-{position:03d}"
            prefix = row["prompt_token_ids"] + row["output_token_ids"][:position]
            forced = row["output_token_ids"][position]
            if (stored := previous.get(key)) is not None:
                prefix = stored.get("prefix_token_ids", prefix)
                forced = stored["forced"]
                if (
                    prefix_digest(prefix) != stored["prefix_sha256"]
                    or len(prefix) != stored["position"]
                ):
                    raise RuntimeError(
                        "Reference teacher conditioning is not recoverable; "
                        "use the original reference report"
                    )
            yield key, prefix, forced
