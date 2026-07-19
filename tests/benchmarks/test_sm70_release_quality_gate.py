# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib

from benchmarks.run_sm70_release_matrix import (
    _fixed_quality_contract_failures,
    _make_cases,
    _quality_metrics,
)

COMPLETE_HTML = """```html
<!DOCTYPE html>
<html><head><style>body { margin: 0; }</style></head>
<body><div>desktop</div><script>console.log('ok')</script></body></html>
```"""


def test_fixed_quality_contract_accepts_naturally_stopped_complete_html():
    assert not _fixed_quality_contract_failures("macos_6k_code", COMPLETE_HTML, "stop")


def test_fixed_quality_contract_rejects_length_truncation():
    assert "did_not_naturally_stop" in _fixed_quality_contract_failures(
        "macos_6k_code", COMPLETE_HTML, "length"
    )


def test_fixed_quality_contract_rejects_incomplete_html():
    assert "incomplete_html_document" in _fixed_quality_contract_failures(
        "macos_6k_code", "```html\n<html><body>", "stop"
    )


def test_fixed_quality_contract_rejects_missing_tag_open_bracket():
    malformed = COMPLETE_HTML.replace("<body><div>", "<body>\ndiv>")
    assert "malformed_html_tag_line" in _fixed_quality_contract_failures(
        "macos_6k_code", malformed, "stop"
    )


def test_fixed_quality_contract_allows_javascript_input_identifier():
    text = COMPLETE_HTML.replace(
        "console.log('ok')",
        "input.addEventListener('keydown', (event) => console.log(event))",
    )

    assert not _fixed_quality_contract_failures("macos_6k_code", text, "stop")


def test_quality_metrics_allow_repeated_markup_in_large_application():
    row_hashes = [
        hashlib.sha256(f"row-{idx}".encode()).hexdigest() for idx in range(90)
    ]
    settings = "\n".join(
        f'<div class="settings-row {digest[:16]}">{digest[16:]}</div>'
        for digest in row_hashes
    )
    unique_code = "".join(
        hashlib.sha256(str(idx).encode()).hexdigest() for idx in range(1000)
    )

    metrics = _quality_metrics(settings + unique_code, list(range(64)))

    assert metrics["repeat20"] > 80
    assert metrics["repeat20"] <= metrics["repeat20_limit"]
    assert metrics["passed"]


def test_quality_metrics_still_reject_degenerate_repetition():
    text = "broken generated output " * 2000

    metrics = _quality_metrics(text, [1] * 2000)

    assert metrics["repeat20"] > metrics["repeat20_limit"]
    assert "repeat20" in metrics["failures"]
    assert "same_token_run" in metrics["failures"]


def test_release_matrix_includes_fp8_weight_fp8_kv_mtp_by_default():
    cases = _make_cases(
        backends=("turbomind",),
        tps=(4,),
        kv_cache_dtypes=("fp8_e5m2",),
    )

    assert any(
        case.model.key == "qwen36-27b-fp8"
        and case.kv_cache_dtype == "fp8_e5m2"
        and case.mode == "mtp4"
        for case in cases
    )
