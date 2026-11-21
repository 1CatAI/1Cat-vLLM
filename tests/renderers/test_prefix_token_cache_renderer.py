# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_V41_PREFIX_TOKEN_CACHE through the chat route's render step.

OpenAIServingRender.render_chat_request is the step /v1/chat/completions
(and /v1/messages, which converts to a chat request) runs before the engine.
Here it runs with a real DeepseekV41Renderer (a real ModelConfig from the model
directory and the real DeepSeek-V4.1 tokenizer) on a chat that grows turn by
turn. The expected ids come from outside the cache: the tokenizer's own full
encode of the prompt text its chat template renders, and the same route with
the switch off (main's path).

The model directory needs only the config and tokenizer files; set
V41_TOKENIZER_DIR to it. Without it the module is skipped (INCONCLUSIVE:
nothing was checked).
"""

import asyncio
import os

import pytest

from vllm.renderers import prefix_token_cache

MODEL_DIR = os.environ.get("V41_TOKENIZER_DIR")
pytestmark = pytest.mark.skipif(
    not MODEL_DIR,
    reason="INCONCLUSIVE: V41_TOKENIZER_DIR (a DeepSeek-V4.1 model directory "
    "with config.json and tokenizer.json) is not set; nothing was checked",
)

MODEL_NAME = "deepseek-v41"


def _document(n: int) -> str:
    return " ".join(
        f"Clause {i}: the parties agree that item {i * 7 % 13} is held for "
        f"account {i * 31 % 97} until the {i % 28 + 1}th day."
        for i in range(n)
    )


def _turns() -> list[list[dict]]:
    """Three requests of one chat; each is the previous one plus a reply and
    a new user message."""
    first = [
        {"role": "system", "content": "You answer questions about documents."},
        {"role": "user", "content": _document(300) + "\n\nSummarise clause 4."},
    ]
    second = first + [
        {"role": "assistant", "content": "Clause 4 holds item 2 for account 27."},
        {"role": "user", "content": "And clause 250?"},
    ]
    third = second + [
        {"role": "assistant", "content": "Clause 250 holds item 8 for account 74."},
        {"role": "user", "content": "List the clauses that name account 3."},
    ]
    return [first, second, third]


@pytest.fixture(scope="module")
def loop():
    # One event loop for the module, as in a server: the renderer's async
    # tokenizer binds to the loop of its first call.
    loop = asyncio.new_event_loop()
    yield loop
    # the async tokenizer's batching task waits on its queue forever
    pending = asyncio.all_tasks(loop)
    for task in pending:
        task.cancel()
    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    loop.close()


@pytest.fixture(scope="module")
def serving():
    from vllm.config import ModelConfig, VllmConfig
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    from vllm.renderers.deepseek_v41 import DeepseekV41Renderer
    from vllm.tokenizers.deepseek_v41 import DeepseekV41Tokenizer

    model_config = ModelConfig(
        model=MODEL_DIR, tokenizer=MODEL_DIR, skip_tokenizer_init=True
    )
    tokenizer = DeepseekV41Tokenizer.from_pretrained(MODEL_DIR)
    renderer = DeepseekV41Renderer(VllmConfig(model_config=model_config), tokenizer)
    return OpenAIServingRender(
        model_config=model_config,
        renderer=renderer,
        model_registry=OpenAIModelRegistry(
            model_config, [BaseModelPath(name=MODEL_NAME, model_path=MODEL_DIR)]
        ),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )


def _route(serving, loop, messages) -> list[int]:
    """The ids the chat route sends to the engine for these messages."""
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest(model=MODEL_NAME, messages=messages)
    result = loop.run_until_complete(serving.render_chat_request(request))
    assert hasattr(result, "token_ids"), result
    return list(result.token_ids)


def _full_encode(serving, messages) -> list[int]:
    """The tokenizer's own full encode of the text the template renders."""
    tokenizer = serving.renderer.get_tokenizer()
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return list(tokenizer.encode(text, add_special_tokens=False))


@pytest.fixture
def fresh_caches():
    prefix_token_cache._CACHES.clear()
    yield
    prefix_token_cache._CACHES.clear()


def test_main_cache_does_not_serve_this_tokenizer(serving):
    # The renderer's own _PrefixTokenizeCache needs an atomic <|im_start|>;
    # DeepSeek-V4.1's tokenizer has none, so on main every turn is tokenized
    # in full. This is the case the switch serves.
    assert serving.renderer._prefix_tok_cache.enabled is False


def test_off_is_main_path(serving, loop, monkeypatch, fresh_caches):
    monkeypatch.setattr(prefix_token_cache, "MODE", "0")
    for messages in _turns():
        assert _route(serving, loop, messages) == _full_encode(serving, messages)
    assert prefix_token_cache._CACHES == {}  # off: no cache is ever made


def test_growing_chat_ids_equal_full_tokenization(
    serving, loop, monkeypatch, fresh_caches
):
    turns = _turns()
    expected = [_full_encode(serving, m) for m in turns]
    monkeypatch.setattr(prefix_token_cache, "MODE", "1")
    got = [_route(serving, loop, m) for m in turns]
    assert got == expected

    (cache,) = prefix_token_cache._CACHES.values()
    # the first request is a miss; each later turn continues the one before
    assert (cache.stats["misses"], cache.stats["hits"]) == (1, 2)
    # each later turn reuses at least the earlier turn's whole document
    assert cache.stats["tokens_reused"] >= len(expected[0]) + len(expected[1]) - 64
    assert (
        cache.stats["tokens_encoded"]
        == (len(expected[0]) + len(expected[1]) + len(expected[2]))
        - cache.stats["tokens_reused"]
    )


def test_edit_inside_earlier_message(serving, loop, monkeypatch, fresh_caches):
    """A request that changes an earlier assistant reply reuses only up to the
    last added token before the change; its ids still equal its full
    tokenization, and the unedited chat still continues correctly."""
    _, second, third = _turns()
    edited = [dict(m) for m in third]
    edited[2]["content"] = "Clause 4 holds item 2 for account 27, until day 5."
    monkeypatch.setattr(prefix_token_cache, "MODE", "1")
    _route(serving, loop, second)
    assert _route(serving, loop, edited) == _full_encode(serving, edited)
    assert _route(serving, loop, third) == _full_encode(serving, third)
    (cache,) = prefix_token_cache._CACHES.values()
    # the edit and the unedited third turn both continue the second turn's
    # document, which is far longer than MIN_REUSE
    assert (cache.stats["misses"], cache.stats["hits"]) == (1, 2)


def test_verify_mode(serving, loop, monkeypatch, fresh_caches):
    turns = _turns()
    expected = [_full_encode(serving, m) for m in turns]
    monkeypatch.setattr(prefix_token_cache, "MODE", "verify")
    assert [_route(serving, loop, m) for m in turns] == expected
    (cache,) = prefix_token_cache._CACHES.values()
    assert (cache.stats["misses"], cache.stats["hits"]) == (1, 2)
    assert cache.stats["different"] == 0


CHATML_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def test_renderer_cache_stays_in_charge_where_enabled(loop, monkeypatch, fresh_caches):
    """Where the renderer's own _PrefixTokenizeCache is enabled (a tokenizer
    with an atomic <|im_start|>), the switch leaves every call to it."""
    from vllm.config import ModelConfig, VllmConfig
    from vllm.entrypoints.openai.models.protocol import BaseModelPath
    from vllm.entrypoints.openai.models.serving import OpenAIModelRegistry
    from vllm.entrypoints.serve.render.serving import OpenAIServingRender
    from vllm.renderers.registry import renderer_from_config

    model_config = ModelConfig(model=CHATML_MODEL)
    renderer = renderer_from_config(VllmConfig(model_config=model_config))
    assert renderer._prefix_tok_cache.enabled is True
    chatml = OpenAIServingRender(
        model_config=model_config,
        renderer=renderer,
        model_registry=OpenAIModelRegistry(
            model_config,
            [BaseModelPath(name=MODEL_NAME, model_path=CHATML_MODEL)],
        ),
        request_logger=None,
        chat_template=None,
        chat_template_content_format="auto",
    )
    monkeypatch.setattr(prefix_token_cache, "MODE", "1")
    tokenizer = renderer.get_tokenizer()
    for messages in _turns():
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        expected = list(tokenizer.encode(text, add_special_tokens=False))
        assert _route(chatml, loop, messages) == expected
    assert prefix_token_cache._CACHES == {}
