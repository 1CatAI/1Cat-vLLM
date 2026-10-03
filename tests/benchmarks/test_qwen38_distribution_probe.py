# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest

from benchmarks.qwen38_distribution_probe import distribution_metrics


def test_distribution_offset_invariance_and_raw_error():
    result = distribution_metrics([0, 1, 2], [4, 5, 6])
    assert result["kl"] < 1e-14
    assert result["max_logit_error"] == 4
    assert result["centered_max_logit_error"] == 0
    assert result["top1_agreement"]


def test_distribution_analytic_kl_and_changed_top1():
    result = distribution_metrics(
        [math.log(0.8), math.log(0.2)], [math.log(0.3), math.log(0.7)]
    )
    expected = 0.8 * math.log(0.8 / 0.3) + 0.2 * math.log(0.2 / 0.7)
    assert result["kl"] == pytest.approx(expected, abs=1e-14)
    assert not result["top1_agreement"]
    with pytest.raises(ValueError):
        distribution_metrics([0, 1], [0, float("nan")])


def test_worker_probe_tracks_request_mapping_and_rejects_wrong_input(tmp_path):
    from types import SimpleNamespace

    import numpy as np
    import torch

    from benchmarks.qwen38_distribution_probe import DistributionProbeWorkerExtension

    sampler = SimpleNamespace(sampling_states=object(), sample=lambda *args: args[0])
    runner = SimpleNamespace(
        sampler=sampler,
        req_states=SimpleNamespace(index_to_req_id={7: "main", 3: "filler"}),
    )
    extension = DistributionProbeWorkerExtension()
    extension.model_runner = runner
    extension.rank = 0
    extension.configure_distribution_probe(
        {
            "main": {
                "prompt_token_ids": [8, 9],
                "continuation": [2, 1],
                "vocabulary": 3,
                "capture": True,
                "output": str(tmp_path),
            },
            "filler": {
                "prompt_token_ids": [6],
                "continuation": [0, 2],
                "vocabulary": 3,
                "capture": False,
                "output": str(tmp_path),
            },
        }
    )
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    sampler.sample(
        logits, None, np.array([3, 7]), torch.tensor([0, 1]), torch.tensor([6, 9]), None
    )
    assert np.array_equal(np.load(tmp_path / "0000.npy"), [4, 5, 6])
    assert logits.argmax(dim=1).tolist() == [0, 2]
    with pytest.raises(RuntimeError, match="input token"):
        sampler.sample(
            torch.zeros(1, 3),
            None,
            np.array([7]),
            torch.tensor([2]),
            torch.tensor([0]),
            None,
        )
