# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref

import pytest
import torch.nn as nn

from vllm.model_executor.models.interfaces import _language_model_by_module

pytestmark = pytest.mark.cpu_test


class _LanguageModel(nn.Module):
    def embed_input_ids(self, input_ids):  # pragma: no cover
        raise NotImplementedError


class _MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()


def test_language_model_cache_is_weak_keyed():
    assert isinstance(_language_model_by_module, weakref.WeakKeyDictionary)


def test_language_model_cache_does_not_pin_model():
    model = _MultiModalModel()
    _language_model_by_module[model] = model.language_model
    model_ref = weakref.ref(model)

    del model
    gc.collect()

    assert model_ref() is None
