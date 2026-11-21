# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PREFIX_TOKEN_CACHE on DeepSeek-V4.1's real tokenizer and chat
template.

Every test compares COMPLETE id sequences with the tokenizer's own full encode
of the same text with the call's own arguments (the renderer's path without the
switch), never with the code under test. The text is public (Python's stdlib
source). The first group is the witness of the cut rule: a cut whose START, not
whole token, lay in the common prefix gave wrong ids; it must pass.

The tokenizer is read from V41_TOKENIZER_DIR (a directory with DeepSeek-V4.1's
tokenizer.json); without it the module is skipped as INCONCLUSIVE.
Run: V41_TOKENIZER_DIR=<dir> pytest -q tests/renderers/test_prefix_token_cache.py
"""

import importlib
import inspect
import json
import os
import random
import threading
import time

import pytest

BASE = os.environ.get("V41_TOKENIZER_DIR", "")
if not os.path.exists(os.path.join(BASE, "tokenizer.json")):
    pytest.skip(
        "INCONCLUSIVE: set V41_TOKENIZER_DIR to a directory with DeepSeek-V4.1's "
        "tokenizer.json",
        allow_module_level=True,
    )

MOD = (
    "argparse",
    "optparse",
    "textwrap",
    "difflib",
    "pydoc",
    "typing",
    "inspect",
    "tarfile",
    "zipfile",
    "subprocess",
    "shutil",
    "logging",
    "_pydecimal",
    "_pydatetime",
    "statistics",
    "random",
    "calendar",
    "enum",
    "dataclasses",
    "ast",
    "configparser",
    "smtplib",
    "ftplib",
    "imaplib",
    "mailbox",
    "pickle",
    "pickletools",
    "threading",
    "doctest",
    "pdb",
    "bdb",
    "ipaddress",
    "fractions",
    "locale",
    "functools",
    "collections",
    "heapq",
    "codecs",
    "gettext",
    "poplib",
)
CPT = 4.22
SEED = 20261117
_T = {}


def tok():
    if "t" not in _T:
        from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

        _T["t"] = DeepseekV41Tokenizer.from_pretrained(BASE)
        _T["src"] = "".join(inspect.getsource(importlib.import_module(m)) for m in MOD)
        _T["spec"] = json.loads(_T["t"].backend_tokenizer.to_str())
    return _T["t"]


def full(text, **kw):
    return list(tok().encode(text, **(kw or {"add_special_tokens": False})))


def render(msgs, mode="thinking"):
    from vllm.third_party.deepseek_v41.encoding import encode_messages

    return encode_messages(msgs, thinking_mode=mode, drop_thinking=True)


def chat(start, name, turns, first=7164, result=2700):
    tok()
    src, at = _T["src"], [start]

    def take(n):
        s = src[at[0] : at[0] + round(n * CPT)]
        at[0] += round(n * CPT)
        assert len(s) == round(n * CPT), "not enough public text"
        return s

    msgs = [
        {"role": "system", "content": f"conversation {name}\n" + take(1444)},
        {"role": "user", "content": take(first)},
    ]
    return msgs, [
        (
            {"role": "assistant", "content": take(40), "reasoning": take(150)},
            {"role": "user", "content": take(result)},
        )
        for _ in range(turns)
    ]


def new_cache():
    from vllm.renderers import prefix_token_cache as ptc

    t = tok()
    added = [a["content"] for a in _T["spec"]["added_tokens"]]
    return ptc.PrefixTokenCache(added, t.convert_tokens_to_ids(added))


def call(cache, text, full_calls=None, max_length=None, kwargs=None):
    kw = kwargs or {"add_special_tokens": False}

    def today():
        if full_calls is not None:
            full_calls.append(len(text))
        return full(text, **kw)

    ids, reused, _ = cache.tokenize(text, lambda s: full(s), today, max_length)
    return ids, reused


def check(cache, text):
    ids, reused = call(cache, text)
    assert ids == full(text), (
        f"WRONG IDS (reused {reused}; {len(ids)} against {len(full(text))})"
    )
    return reused


# ---------------------------------------------------- the cut rule's witness

LEAD = "Public example. " * 1800


@pytest.mark.parametrize("suffix", ["<think>test", "ing", "t", "<thirty", ""])
def test_witness_toy_cases(suffix):
    cache = new_cache()
    check(cache, LEAD + "test<think>one")
    check(cache, LEAD + "test" + suffix)


def test_witness_rendered_chat_edit_and_equal_length():
    cache = new_cache()
    old = render([{"role": "user", "content": LEAD + "test<think>one"}], mode="chat")
    check(cache, old)
    check(cache, render([{"role": "user", "content": LEAD + "testing"}], mode="chat"))
    # the equal-length witness: 'some' + '<think>one' then 'somesure' (same id
    # count, different ids on the old rule)
    cache = new_cache()
    check(
        cache,
        render([{"role": "user", "content": LEAD + "some<think>one"}], mode="chat"),
    )
    check(cache, render([{"role": "user", "content": LEAD + "somesure"}], mode="chat"))


# ------------------------------ the rule's premises, read from tokenizer.json


def test_premises_hold_for_this_tokenizer_and_each_flag_breaks_them():
    from vllm.renderers import prefix_token_cache as ptc

    tok()
    spec = _T["spec"]
    assert ptc.check_tokenizer_json(spec) == []
    flags = {
        k: sum(1 for a in spec["added_tokens"] if a.get(k))
        for k in ("lstrip", "rstrip", "single_word", "normalized")
    }
    print(
        f"\nadded tokens {len(spec['added_tokens'])}, flags set {flags}, "
        f"normalizer {spec['normalizer']}"
    )
    for flag in ("lstrip", "rstrip", "single_word"):
        bad = json.loads(json.dumps(spec))
        bad["added_tokens"][0][flag] = True
        assert any(flag in r for r in ptc.check_tokenizer_json(bad))
    bad = json.loads(json.dumps(spec))
    bad["added_tokens"].append(
        dict(bad["added_tokens"][0], content="<think>x", id=10**7)
    )
    assert ptc.check_tokenizer_json(bad)
    bad = json.loads(json.dumps(spec))
    bad["normalizer"] = {"type": "NFKC"}
    assert any("normalizer" in r for r in ptc.check_tokenizer_json(bad))
    # a normalized token whose start is a non-normalized token's end
    bad = json.loads(json.dumps(spec))
    plain = next(a for a in bad["added_tokens"] if not a.get("normalized"))
    bad["added_tokens"].append(
        dict(plain, content=plain["content"][-2:] + "zq", normalized=True, id=10**7)
    )
    assert any("overlaps" in r for r in ptc.check_tokenizer_json(bad))


def test_suffix_alone_equals_the_tail_of_the_full_encode():
    # START side: the suffix encode begins at an added token; it must equal
    # the full encode's tail from that token
    msgs, turns = chat(0, "S", 6)
    for a, u in turns:
        msgs = msgs + [a, u]
    text = render(msgs)
    ids = full(text)
    cache = new_cache()
    cuts = cache._cuts(text, ids)
    assert cuts and len(cuts) > 20
    for c, _, t in cuts:
        assert full(text[c:]) == ids[t:], c


# ------------------------------------- adversarial edits on the real template


def _primed(turns=8, start=0, name="E"):
    msgs, tr = chat(start, name, turns)
    for a, u in tr:
        msgs = msgs + [a, u]
    text = render(msgs)
    cache = new_cache()
    check(cache, text)
    return cache, text, msgs


def test_edits_before_inside_and_after_every_kind_of_delimiter():
    cache, text, _ = _primed()
    seen, n = set(), 0
    for c, e, t in cache._entries[max(cache._entries)].cuts:
        kind = text[c:e]
        if kind in seen or t < 1100:
            continue
        seen.add(kind)
        for pos in (c - 1, c, c + 1, (c + e) // 2, e - 1, e, e + 1):
            for new in (
                text[:pos] + "x" + text[pos + 1 :],
                text[:pos] + text[pos + 1 :],
                text[:pos] + "y" + text[pos:],
            ):
                fresh = new_cache()
                check(fresh, text)
                check(fresh, new)
                n += 1
    print(f"\n{len(seen)} delimiter kinds, {n} edits around them: all identical")
    assert len(seen) >= 4


@pytest.mark.parametrize(
    "content",
    [
        "<think>",
        "</think>",
        "<｜Assistant｜>",
        "<｜User｜>",
        "<｜tool▁calls▁begin｜>",
        "<｜tool▁call▁begin｜>x<｜tool▁sep｜>{}",
        "<｜end▁of▁sentence｜>",
        "<thi",
        "nk>",
    ],
)
def test_added_token_strings_typed_inside_user_text(content):
    msgs, tr = chat(0, "U", 3)
    for a, u in tr:
        msgs = msgs + [a, u]
    cache = new_cache()
    check(cache, render(msgs))
    check(
        cache,
        render(
            msgs
            + [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "see " + content + " here"},
            ]
        ),
    )
    check(
        cache,
        render(
            msgs[:-1] + [{"role": "user", "content": msgs[-1]["content"] + content}]
        ),
    )


@pytest.mark.parametrize("u", ["é", "中文", "😀", "​", "\r\n", " ", "123"])
def test_unicode_and_whitespace_at_the_cut(u):
    cache, text, msgs = _primed(turns=4, name="X")
    c = text.rfind("<｜User｜>")
    for new in (
        text[:c] + u + text[c:],
        text[: c + len("<｜User｜>")] + u + text[c + len("<｜User｜>") :],
    ):
        check(cache, new)


def test_a_branch_from_an_earlier_turn_and_two_chats_sharing_an_opening():
    cache, text, msgs = _primed(turns=10, name="B")
    branch = render(
        msgs[: 2 + 2 * 5]
        + [
            {"role": "assistant", "content": "other"},
            {"role": "user", "content": "a new turn"},
        ]
    )
    assert check(cache, branch) > 1024
    opening = msgs[:2]
    other = render(
        [
            dict(opening[0]),
            dict(opening[1]),
            {"role": "assistant", "content": "x"},
            {"role": "user", "content": "y"},
        ]
    )
    assert (
        check(cache, other) > 1024
    )  # the shared opening is reused, the rest tokenized: identical


def test_truncation_left_and_right_and_max_length():
    from vllm.renderers import prefix_token_cache as ptc

    cache, text, msgs = _primed(turns=3, name="T")
    text2 = render(
        msgs
        + [
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b " * 400},
        ]
    )
    # truncation_side set: get_encode_kwargs asks truncation=False and the
    # renderer truncates the full ids after
    assert (
        ptc.served_max_length({"add_special_tokens": False, "truncation": False})
        is None
    )
    ids, _ = call(cache, text2)
    assert ids[:1000] == full(text2)[:1000] and ids[-1000:] == full(text2)[-1000:]
    # max_length below the joined length: today's call with the call's own arguments
    m = len(full(text)) + 10
    kw = {"add_special_tokens": False, "truncation": True, "max_length": m}
    calls = []
    ids, reused = call(new_cache(), text2, calls, max_length=m, kwargs=kw)
    assert ids == full(text2, **kw) and reused == 0 and calls


def test_a_truncated_miss_is_never_a_parent():
    """A generate model's tokenizer truncates on the left
    (vllm/tokenizers/registry.py), dropping the head of the text. Were such a
    call's ids stored, a later prompt sharing that head and cutting past it
    would join a headless prefix to its tail and, when short enough, return
    it (1,395 ids where the full call gives 2,074, before the fix)."""
    t = tok()
    src = _T["src"]
    head = src[: round(2074 * CPT)]
    cache = new_cache()
    assert not cache._pattern.search(head)  # plain text: no added token in it
    m = len(full(head))
    kw = {"add_special_tokens": False, "truncation": True, "max_length": m}
    old = head + "<think>" + src[-round(680 * CPT) :]
    new = head + "<think>short"
    assert len(full(old)) > m and len(full(new)) > m  # both calls truncate
    side = t.truncation_side
    t.truncation_side = "left"
    try:
        assert full(old, **kw) == full(old)[-m:]  # the head is what goes
        ids, reused = call(cache, old, max_length=m, kwargs=kw)
        assert ids == full(old, **kw) and reused == 0 and len(ids) == m
        ids, reused = call(cache, new, max_length=m, kwargs=kw)
        assert ids == full(new, **kw) and reused == 0
        assert cache.stats["misses"] == 2 and not cache._entries
    finally:
        t.truncation_side = side


def test_fuzz_2000_random_edits():
    rnd = random.Random(SEED)
    msgs, tr = chat(300_000, "F", 4, first=6000, result=2900)
    for a, u in tr:
        msgs = msgs + [a, u]
    text = render(msgs)
    n_tok = len(full(text))
    specials = [a["content"] for a in _T["spec"]["added_tokens"]][:60] + [
        "<think>",
        "</think>",
        "<｜User｜>",
    ]
    cache = new_cache()
    check(cache, text)
    hits = 0
    for i in range(2000):
        pos = rnd.randrange(len(text))
        if rnd.random() < 0.5:
            j = rnd.randrange(len(text) - 30)
            rep = text[j : j + rnd.randrange(0, 25)]
        else:
            rep = rnd.choice(specials)
        cut = rnd.randrange(0, 4)
        new = text[:pos] + rep + text[pos + cut :]
        ids, reused = call(cache, new)
        assert ids == full(new), f"edit {i}: pos {pos} rep {rep!r} cut {cut}"
        hits += reused > 0
        if rnd.random() < 0.2:
            check(cache, text)  # keep the original among the parents
    print(
        f"\nfuzz seed {SEED}: 2000 edits of a {n_tok}-token chat, 0 mismatches, "
        f"{hits} served from the cache"
    )


# ---------------------------------- behaviour: chains, misses, eviction, verify


def test_growing_chat_identical_hit_rate_before_and_after_the_fix():
    """Reused ids per continued request under the old rule (cut START within
    the common prefix) and the new one (whole added token within it), both
    computed from the same entries; only the new rule's ids are returned."""
    from vllm.renderers import prefix_token_cache as ptc

    cache = new_cache()
    msgs, turns = chat(0, "A", 40)
    old_r, new_r, t_full, t_cache = [], [], 0.0, 0.0
    for k, (a, u) in enumerate(turns):
        msgs = msgs + [a, u]
        text = render(msgs)
        if cache._entries:
            e = cache._entries[max(cache._entries)]
            n = ptc.common_prefix_len(text, e.text)
            old_r.append(max(t for c, _, t in e.cuts if c <= n))
        t0 = time.perf_counter()
        ref = full(text)
        t1 = time.perf_counter()
        ids, reused = call(cache, text)
        t2 = time.perf_counter()
        assert ids == ref, k
        if k:
            new_r.append(reused)
            t_full, t_cache = t_full + t1 - t0, t_cache + t2 - t1
    lost = [o - n for o, n in zip(old_r, new_r)]
    print(
        f"\n{len(new_r)} continued requests up to {len(ref)} tokens: reused per "
        f"request old rule median {sorted(old_r)[len(old_r) // 2]}, new rule "
        f"{sorted(new_r)[len(new_r) // 2]}; ids lost per request "
        f"{min(lost)}-{max(lost)}; full {1000 * t_full / len(new_r):.1f} ms, "
        f"cached {1000 * t_cache / len(new_r):.1f} ms"
    )
    assert (
        cache.stats["hits"] == len(turns) - 1
        and len(cache._entries) == 1
        and max(lost) <= 2
    )


def test_two_conversations_interleaved_both_hit():
    cache = new_cache()
    (ma, ta), (mb, tb) = chat(0, "A", 8), chat(1_200_000, "B", 8)
    for k in range(8):
        ma, mb = ma + list(ta[k]), mb + list(tb[k])
        ra, rb = check(cache, render(ma)), check(cache, render(mb))
        if k:
            assert ra > 1024 and rb > 1024
    assert len(cache._entries) == 2


def test_eviction_and_a_miss_is_todays_call_and_its_cost():
    cache = new_cache()
    chats = [chat(150_000 * i, f"C{i}", 1, first=2000, result=500) for i in range(10)]
    calls = []
    for m, tr in chats:
        call(cache, render(m + list(tr[0])), calls)
    assert cache.stats["evictions"] == 2 and len(calls) == 10
    for i in (9, 2, 0):
        m, tr = chats[i]
        text = render(
            m
            + list(tr[0])
            + [
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "more"},
            ]
        )
        before = len(calls)
        ids, reused = call(cache, text, calls)
        assert ids == full(text)
        assert (len(calls) - before, reused > 0) == (
            (1, False) if i == 0 else (0, True)
        ), i
    msgs, tr = chat(0, "M", 40)
    for a, u in tr:
        msgs = msgs + [a, u]
    text = render(msgs)
    t0 = time.perf_counter()
    ref = full(text)
    t1 = time.perf_counter()
    ids, reused = call(new_cache(), text)
    t2 = time.perf_counter()
    assert ids == ref and reused == 0
    print(
        f"\nmiss on {len(ref)} tokens: today's call {1000 * (t1 - t0):.1f} ms, "
        f"the miss path {1000 * (t2 - t1):.1f} ms"
    )


def test_a_hit_beside_a_concurrent_miss_does_not_wait_for_it():
    cache, text, msgs = _primed(turns=30, name="L")
    big = render(
        [
            {
                "role": "user",
                "content": _T["src"][2_000_000 : 2_000_000 + round(120_000 * CPT)],
            }
        ]
    )
    nxt = render(
        msgs + [{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"}]
    )
    ref = full(nxt)
    t0 = time.perf_counter()
    assert call(cache, nxt)[0] == ref
    alone = time.perf_counter() - t0
    cache, text, msgs = _primed(turns=30, name="L")
    out = {}

    def miss():
        t = time.perf_counter()
        call(cache, big)
        out["miss"] = time.perf_counter() - t

    th = threading.Thread(target=miss)
    th.start()
    time.sleep(0.02)
    t0 = time.perf_counter()
    assert call(cache, nxt)[0] == ref
    beside = time.perf_counter() - t0
    th.join()
    print(
        f"\nhit alone {1000 * alone:.1f} ms; beside a {1000 * out['miss']:.0f} ms "
        f"miss {1000 * beside:.1f} ms"
    )
    # the lock no longer covers the tokenizer (the GIL is shared, so some wait
    # remains)
    assert beside < out["miss"] / 2


def test_verify_lines_hits_and_a_different_result_is_dropped(monkeypatch):
    from vllm.renderers import base
    from vllm.renderers import prefix_token_cache as ptc

    monkeypatch.setattr(ptc, "MODE", "verify")
    monkeypatch.setattr(ptc, "_CACHES", {})
    lines = []
    monkeypatch.setattr(base.logger, "info", lambda fmt, *a: lines.append(fmt % a))
    t = tok()
    msgs, turns = chat(0, "V", 6)
    kwargs = {"truncation": True, "max_length": 524289, "add_special_tokens": False}
    for a, u in turns:
        msgs = msgs + [a, u]
        text = render(msgs)
        assert base._prefix_cached(t, text, kwargs) == full(text, **kwargs)
    ver = [x for x in lines if x.startswith("VLLM_V41_PREFIX_TOKEN_CACHE=verify:")]
    assert len(ver) == 6 and all(x.endswith(": identical") for x in ver)
    assert (
        sum(" (miss): " in x for x in ver) == 1
        and sum(" (hit): " in x for x in ver) == 5
    )
    # a wrong suffix encode (injected) must be reported DIFFERENT, the full ids
    # returned, and the entry dropped
    cache = ptc.cache_for(t)
    real = t.encode
    monkeypatch.setattr(
        t,
        "encode",
        lambda s, **kw: [1] + real(s, **kw) if len(s) < 50_000 else real(s, **kw),
    )
    msgs = msgs + [
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "y"},
    ]
    text = render(msgs)
    assert base._prefix_cached(t, text, kwargs) == list(real(text, **kwargs))
    assert "DIFFERENT" in lines[-1] or any("DIFFERENT" in x for x in lines[-2:])
    assert (
        all(e.text != text for e in cache._entries.values())
        and cache.stats["different"] == 1
    )
    assert "summary: hits" in cache.summary()


def test_switch_values_and_kwargs_served():
    from vllm.renderers import prefix_token_cache as ptc

    assert ptc.mode_from_env(None) == "0" and ptc.mode_from_env("verify") == "verify"
    with pytest.raises(ValueError):
        ptc.mode_from_env("yes")
    assert ptc.served_max_length({"add_special_tokens": True}) is False
    assert (
        ptc.served_max_length({"add_special_tokens": False, "padding": True}) is False
    )
    assert (
        ptc.served_max_length({"add_special_tokens": False, "truncation": True})
        is False
    )
    assert (
        ptc.served_max_length(
            {"add_special_tokens": False, "truncation": True, "max_length": 9}
        )
        == 9
    )


def test_memory_of_eight_420000_token_prompts():
    cache = new_cache()
    src = _T["src"] if _T else (tok(), _T["src"])[1]
    n = round(420_000 * CPT)
    for i in range(8):
        off = (i * 150_000) % (len(src) - n)
        call(
            cache,
            render(
                [
                    {"role": "system", "content": f"s{i}"},
                    {"role": "user", "content": src[off : off + n]},
                ]
            ),
        )
    sizes = [len(e.ids) for e in cache._entries.values()]
    held = cache.held_bytes()
    print(
        f"\n8 entries of {min(sizes)}-{max(sizes)} tokens: "
        f"ids + texts held {held / 1e6:.1f} MB"
    )
    assert len(sizes) == 8 and min(sizes) > 400_000 and held < 40e6
