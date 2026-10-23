# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

from markdown import Markdown  # type: ignore[import-untyped]

from docs.mkdocs.hooks import url_schemes


def test_source_links_use_site_repository_and_preserve_upstream_links():
    extension = url_schemes.UrlSchemesExtension()
    extension.repo_url = "https://github.com/1CatAI/1Cat-vLLM"
    extension.page = SimpleNamespace(
        file=SimpleNamespace(
            abs_src_path=str(url_schemes.DOC_DIR / "design/architecture/README.md")
        )
    )
    text = (
        "[codec](../../../vllm/v1/attention/kv_codecs.py#L1)\n\n"
        "[directory](../../../vllm/v1/attention/)\n\n"
        "[upstream](https://github.com/vllm-project/vllm/blob/main/README.md)"
    )
    html = Markdown(extensions=[extension]).convert(text)
    assert "1CatAI/1Cat-vLLM/blob/main/vllm/v1/attention/kv_codecs.py#L1" in html
    assert "1CatAI/1Cat-vLLM/tree/main/vllm/v1/attention" in html
    assert "vllm-project/vllm/blob/main/README.md" in html


def test_hook_uses_each_build_config_without_reusing_previous_repo():
    for repo in (
        "https://github.com/1CatAI/1Cat-vLLM/",
        "https://github.com/vllm-project/vllm",
    ):
        config: dict[str, Any] = {"repo_url": repo, "markdown_extensions": []}
        url_schemes.on_config(config)
        assert config["markdown_extensions"][-1].repo_url == repo.rstrip("/")
