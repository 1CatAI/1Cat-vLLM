# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling 2 assistant output format, shared by the reasoning and tool parsers.

An assistant turn is a run of segments, each
`<|message_model|>{name}<|content_TYPE|>{body}<|end_message|>`, ended by
`<|content_model_end_sampling|>` (the EOS):

- `<|content_thinking|>`: reasoning;
- `<|content_text|>`: visible text;
- `<|content_invoke_tool_json|>` with the tool name as `{name}`: a tool call,
  body `{"name": ..., "args": {...}}`.

The markers are special tokens, so parsing needs the text decoded with
`skip_special_tokens=False` (and no spaces between special tokens).
"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass

MESSAGE_MODEL = "<|message_model|>"
THINKING = "<|content_thinking|>"
TEXT = "<|content_text|>"
INVOKE = "<|content_invoke_tool_json|>"
END = "<|end_message|>"
END_SAMPLING = "<|content_model_end_sampling|>"
CONTENT_TAGS = {THINKING: "thinking", TEXT: "text", INVOKE: "invoke"}

# Token ids (Inkling 2 tokenizer).
MESSAGE_MODEL_ID = 200001
THINKING_ID = 200008
# Tokens that start a non-model message or end a model turn.
TURN_BOUNDARY_IDS = frozenset({200000, 200002, 200003, 200006})


@dataclass
class Segment:
    kind: str  # "thinking" | "text" | "invoke" | "raw" | "header"
    name: str
    body: str
    closed: bool
    start: int  # index of the segment's first character in the text


def _strip_end(s: str) -> str:
    return s.replace(END_SAMPLING, "")


def parse_segments(text: str) -> list[Segment]:
    """Split (possibly partial) assistant output into segments. A trailing
    segment without `<|end_message|>` is `closed=False`; a `header` segment is
    a `<|message_model|>` whose content tag has not arrived yet."""
    segs: list[Segment] = []
    i, n = 0, len(text)
    while i < n:
        j = text.find(MESSAGE_MODEL, i)
        if j != i:
            chunk = _strip_end(text[i : n if j < 0 else j])
            if chunk:
                segs.append(Segment("raw", "", chunk, True, i))
            if j < 0:
                break
        head = j + len(MESSAGE_MODEL)
        next_mm = text.find(MESSAGE_MODEL, head)
        limit = n if next_mm < 0 else next_mm
        tag_pos, tag = -1, ""
        for candidate in CONTENT_TAGS:
            p = text.find(candidate, head, limit)
            if p >= 0 and (tag_pos < 0 or p < tag_pos):
                tag_pos, tag = p, candidate
        if tag_pos < 0:
            name = _strip_end(text[head:limit])
            if next_mm < 0:
                segs.append(Segment("header", name, "", False, j))
                break
            if name:
                segs.append(Segment("raw", "", name, True, j))
            i = next_mm
            continue
        name = text[head:tag_pos]
        start = tag_pos + len(tag)
        end = text.find(END, start)
        if next_mm >= 0 and (end < 0 or next_mm < end):
            # A new segment began before this one closed: close it there.
            segs.append(Segment(CONTENT_TAGS[tag], name, text[start:next_mm], True, j))
            i = next_mm
            continue
        if end < 0:
            segs.append(
                Segment(CONTENT_TAGS[tag], name, _strip_end(text[start:]), False, j)
            )
            break
        segs.append(Segment(CONTENT_TAGS[tag], name, text[start:end], True, j))
        i = end + len(END)
    return segs


def split_reasoning(text: str) -> tuple[str, int | None]:
    """Reasoning text of the leading thinking segments, and the index where
    the first non-thinking segment starts (None while undecided)."""
    parts: list[str] = []
    for seg in parse_segments(text):
        if seg.kind == "thinking":
            parts.append(seg.body)
            continue
        if seg.kind == "header" and (THINKING.startswith(seg.name) or not seg.name):
            break  # might still become a thinking segment
        return "".join(parts), seg.start
    return "".join(parts), None


def reasoning_ended(token_ids) -> bool:
    """Whether the model's current turn has left its reasoning: a
    `<|message_model|>` followed by a non-thinking token after the last turn
    boundary (prompt messages and earlier turns are ignored)."""
    ids = list(token_ids)
    start = 0
    for k in range(len(ids) - 1, -1, -1):
        if ids[k] in TURN_BOUNDARY_IDS:
            start = k + 1
            break
    for k in range(start, len(ids) - 1):
        if ids[k] == MESSAGE_MODEL_ID and ids[k + 1] != THINKING_ID:
            return True
    return False


def content_start_index(token_ids) -> int | None:
    ids = list(token_ids)
    for k in range(len(ids) - 1):
        if ids[k] == MESSAGE_MODEL_ID and ids[k + 1] != THINKING_ID:
            return k
    return None


def render_raw(seg: Segment) -> str:
    """A non-thinking segment as the model wrote it (for the tool parser)."""
    if seg.kind == "text":
        return MESSAGE_MODEL + seg.name + TEXT + seg.body + (END if seg.closed else "")
    if seg.kind == "invoke":
        return (
            MESSAGE_MODEL + seg.name + INVOKE + seg.body + (END if seg.closed else "")
        )
    if seg.kind == "header":
        return MESSAGE_MODEL + seg.name
    return seg.body


def visible_content(segs: list[Segment], closed_invokes_only: bool = False) -> str:
    """The content a client sees: text segments and stray text as plain text,
    with tool calls kept as model markup (for the tool parser, or as text when
    the request allows no tools). Thinking and unresolved headers are left out."""
    out: list[str] = []
    for seg in segs:
        if seg.kind in ("text", "raw"):
            out.append(seg.body)
        elif seg.kind == "invoke" and (seg.closed or not closed_invokes_only):
            out.append(render_raw(seg))
    return "".join(out)


def parse_tool_call(seg: Segment) -> tuple[str, str] | None:
    """`(name, arguments JSON)` of an invoke segment, or None if malformed."""
    try:
        obj = json.loads(seg.body.strip())
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or seg.name.strip()
    args = obj.get("args", obj.get("arguments", {}))
    if isinstance(args, str):
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            args = json.loads(args)
    if not name:
        return None
    return name, json.dumps(args if args is not None else {}, ensure_ascii=False)
