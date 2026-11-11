# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP2 stage-1 commits reproduce the single-stage commit sequence (CPU only).

On one stage the runner commits each forward after sampling, with
``_count_contiguous_spec_tokens`` of the clipped sampled matrix (prefill
chunks: ``pending_length``). A first PP stage never samples; ``StageCommit``
commits prefill chunks after the forward and decode steps either from the
sampled counts (async scheduling with drafts) or from the next step's
``num_computed_tokens`` delta (sync). On seeded synthetic token streams (prompt
lengths, chunking, draft acceptance, EOS and max_tokens stops, prefix
restores) these tests check that both PP paths commit the same (start, count)
pairs as the single stage, in the order a device protocol model accepts. That
model mirrors ``_hash_request`` / ``_commit_history``
(layers/deepseek_v41_engram.py). The runner's two helpers are taken from
gpu_model_runner.py by source, so the test follows their current code without
importing the runner (which needs the compiled extension).

The ``test_model_*`` tests take DeepseekV41Model's ``commit`` / ``_commit`` /
``pp_after_forward`` / ``pp_before_forward`` / ``_ratio2_states`` from model.py
the same way and drive them as the runner does. Under async scheduling the
final prefill chunk is committed after the forward and the sampled-count
receive that follows must not commit it again (the armed-flag guard); without
the guard the device protocol traps the double commit.

    python tests/models/test_deepseek_v41_pp_commit.py   # or pytest
"""

import ast
import random
from pathlib import Path
from types import SimpleNamespace

import torch

from vllm.models.deepseek_v41.pp_commit import StageCommit

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "vllm" / "v1" / "worker" / "gpu_model_runner.py"
K = 5  # drafts per decode step (serve-pp2.yaml)
EOS = 1


def _runner_helpers():
    tree = ast.parse(RUNNER.read_text())
    names = {"_count_contiguous_spec_tokens", "_clip_stateful_sample_limits"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    namespace = {"torch": torch}
    module = ast.Module(body=list[ast.stmt](nodes), type_ignores=[])
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return (
        namespace["_count_contiguous_spec_tokens"],
        namespace["_clip_stateful_sample_limits"],
    )


COUNT, CLIP = _runner_helpers()


class DeviceProtocol:
    """The hash state's armed-flag protocol, as its Triton kernels enforce it."""

    def __init__(self):
        self.position, self.pending = 0, False
        self.step: tuple[int, int, bool] | None = None
        self.commits = []

    def forward(self, start, n, prefill):
        if start != 0 and (start != self.position or self.pending):
            raise AssertionError(
                f"device trap: forward at {start}, committed {self.position}, "
                f"pending {self.pending}"
            )
        self.step, self.pending = (start, n, prefill), True

    def commit(self, accepted):
        # DeepseekV41Model._commit: where(pending_prefill, pending_length, ..)
        assert self.step is not None, "device trap: commit before any forward"
        start, n, prefill = self.step
        selected = n if prefill else accepted
        if not self.pending or not 0 <= selected <= n:
            raise AssertionError(f"device trap: commit {selected} of {self.step}")
        self.position, self.pending = start + selected, False
        self.commits.append((start, selected))

    def restore(self, cached):
        self.position, self.pending = cached, False


def _request(rng, rid):
    prompt = rng.randint(1, 300)
    cached = 0
    if rng.random() < 0.3 and prompt > 8:
        cached = rng.randrange(2, prompt, 2)  # session prefix hit, even position
    return SimpleNamespace(
        rid=rid,
        prompt=prompt,
        cached=cached,
        max_tokens=rng.randint(1, 60),
        chunk=rng.choice([16, 64, 128, 8192]),
        eos_rate=rng.choice([0.0, 0.02, 0.1]),
    )


def _steps(rng, req):
    """The single-stage run: (start, n, prefill, count) per forward.

    count is what the single stage commits: pending_length for a prefill
    chunk, else _count_contiguous_spec_tokens of the clipped sampled matrix.
    The last step ends the request (EOS or max_tokens).
    """
    steps, position, outputs = [], req.cached, 0
    while position < req.prompt:
        n = min(req.chunk, req.prompt - position)
        steps.append((position, n, True, n))
        position += n
    # The final prefill chunk sampled the first output token.
    outputs, last_eos = 1, rng.random() < req.eos_rate
    params = SimpleNamespace(
        max_tokens=req.max_tokens,
        stop_token_ids=None,
        ignore_eos=False,
        eos_token_id=EOS,
    )
    while outputs < req.max_tokens and not last_eos:
        n = 1 + K
        accepted = rng.randint(0, K)
        row = [
            rng.choice([EOS] if rng.random() < req.eos_rate else [7, 8, 9])
            for _ in range(accepted + 1)
        ]
        sampled = torch.tensor([row + [-1] * (K - accepted)], dtype=torch.int32)
        # output_budget (model.py): outputs that exist before this step
        before = max(0, position + 1 - req.prompt)
        budget = torch.tensor([max(0, req.max_tokens - before)])
        request = SimpleNamespace(
            sampling_params=params, output_token_ids=[0] * outputs
        )
        CLIP(sampled, [req.rid], {req.rid: request}, budget)
        count = int(COUNT(sampled)[0])
        assert count >= 1
        steps.append((position, n, False, count))
        kept = sampled[0, :count].tolist()
        outputs += count
        last_eos = EOS in kept
        position += count
    return steps


def _single_stage(reqs, streams):
    """The single stage's commits, one list per request."""
    device, per_request = DeviceProtocol(), []
    for req, steps in zip(reqs, streams):
        if req.cached:
            device.restore(req.cached)
        first = len(device.commits)
        for start, n, prefill, count in steps:
            device.forward(start, n, prefill)
            device.commit(count)
        per_request.append(device.commits[first:])
    return per_request


def _pp_stage1(reqs, streams, mode):
    """mode 'sync': deferred delta commits; 'async': sampled-count commits."""
    device, commit = DeviceProtocol(), StageCommit()

    def model_commit(selected):  # DeepseekV41Model._commit
        device.commit(selected)

    for req, steps in zip(reqs, streams):
        if req.cached:
            commit.reset()  # DeepseekV41Model.restore_prefix
            device.restore(req.cached)
        for index, (start, n, prefill, count) in enumerate(steps):
            # runner: _update_states, then _pp_stage_before_forward
            deferred = commit.before_forward([req.rid], [start])
            if deferred is not None:
                model_commit(deferred)
            device.forward(start, n, prefill)
            # runner: _pp_stage_after_forward
            if commit.after_forward([req.rid], [start], [req.prompt]):
                model_commit(n)  # selected = pending_length
            final_chunk = prefill and start + n == req.prompt
            # _pp_receive_spec_decode_state -> commit_speculative_state;
            # the last stage skips the broadcast on non-final chunks.
            if (
                mode == "async"
                and (not prefill or final_chunk)
                and commit.take_sampled()
            ):
                model_commit(count)
            assert prefill == (start < req.prompt)
    return device.commits, commit.armed


def _case(seed, requests=4):
    rng = random.Random(seed)
    reqs = [_request(rng, f"r{seed}-{i}") for i in range(requests)]
    streams = [_steps(rng, req) for req in reqs]
    return reqs, streams


def test_sync_matches_single_stage_except_the_final_decode():
    for seed in range(300):
        reqs, streams = _case(seed)
        reference = _single_stage(reqs, streams)
        got, armed = _pp_stage1(reqs, streams, "sync")
        # A request's last forward, when it is a decode step, is never
        # committed on stage 1 (there is no next step); the next request
        # starts at 0 or after restore_prefix, which reset the protocol.
        expected = []
        for commits, steps in zip(reference, streams):
            expected += commits[:-1] if not steps[-1][2] else commits
        assert got == expected, seed
        assert armed == (not streams[-1][-1][2])


def test_async_matches_single_stage_exactly():
    for seed in range(300):
        reqs, streams = _case(seed)
        got, armed = _pp_stage1(reqs, streams, "async")
        assert got == sum(_single_stage(reqs, streams), []), seed
        assert not armed


def test_counts_are_the_runner_counts():
    # The deferred delta equals _count_contiguous_spec_tokens of the clipped
    # matrix whenever the request continues (the scheduler's rollback rule).
    for seed in range(100):
        _, streams = _case(seed)
        for steps in streams:
            for (start, n, prefill, count), following in zip(steps, steps[1:]):
                assert following[0] - start == count
                assert 1 <= count <= n


def test_device_protocol_catches_double_and_miscounted_commits():
    device = DeviceProtocol()
    device.forward(0, 6, True)
    device.commit(1)  # a prefill commits pending_length whatever the count
    assert device.position == 6
    device.forward(6, 6, False)
    try:
        device.forward(6, 6, False)  # the decode step was never committed
    except AssertionError:
        pass
    else:
        raise AssertionError("missing commit not caught")
    device.commit(3)
    try:
        device.commit(3)  # double commit
    except AssertionError:
        pass
    else:
        raise AssertionError("double commit not caught")
    try:
        device.forward(10, 6, False)  # a count that disagrees with the scheduler
    except AssertionError:
        pass
    else:
        raise AssertionError("miscount not caught")


def test_stage_commit_rules():
    commit = StageCommit()
    assert commit.after_forward(["a"], [0], [10]) is True  # prefill: now
    assert not commit.armed and commit.before_forward(["a"], [10]) is None
    assert commit.after_forward(["a"], [10], [10]) is False  # decode: armed
    try:
        commit.after_forward(["a"], [10], [10])
    except RuntimeError:
        pass
    else:
        raise AssertionError("second forward while armed")
    assert commit.before_forward(["a"], [14]) == 4
    assert commit.after_forward(["a"], [14], [10]) is False
    assert commit.take_sampled() is True and commit.take_sampled() is False
    assert commit.after_forward(["a"], [17], [10]) is False
    assert commit.before_forward(["b"], [0]) is None  # replaced: dropped
    assert commit.after_forward(["b"], [20], [10]) is False
    try:
        commit.before_forward(["b"], [20])  # no progress on a decode step
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-positive delta")
    try:
        commit.after_forward(["a", "b"], [0, 0], [1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("two requests")


MODEL = ROOT / "vllm" / "models" / "deepseek_v41" / "model.py"
LAYERS = 40
COMMIT_METHODS = (
    "commit",
    "_commit",
    "pp_after_forward",
    "pp_before_forward",
    "_ratio2_states",
)


def _model_methods(drop_guard=False):
    """DeepseekV41Model's commit methods from model.py, optionally without the
    armed-flag guard in ``commit`` (the mutation the guard exists for)."""
    tree = ast.parse(MODEL.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "DeepseekV41Model"
    )
    nodes = [
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name in COMMIT_METHODS
    ]
    assert {n.name for n in nodes} == set(COMMIT_METHODS)
    if drop_guard:
        commit = next(n for n in nodes if n.name == "commit")
        kept = [s for s in commit.body if "take_sampled" not in ast.unparse(s)]
        assert len(kept) == len(commit.body) - 1
        commit.body = kept
    # _BATCH: VLLM_V41_BATCH off (one request, the path these tests cover).
    namespace = {"torch": torch, "_BATCH": False}
    module = ast.Module(body=list[ast.stmt](nodes), type_ignores=[])
    exec(compile(module, str(MODEL), "exec"), namespace)
    return {n.name: namespace[n.name] for n in nodes}


class _State:
    """A committed-state object (hash state or ratio-2 compressor state) with
    the device protocol: one commit per forward, next forward at the commit."""

    def __init__(self):
        self.position, self.start, self.pending = 0, 0, False
        self.commits = []

    def forward(self, start):
        if start != 0 and (start != self.position or self.pending):
            raise AssertionError(f"device trap: forward at {start}")
        self.start, self.pending = start, True

    def commit(self, selected):
        if not self.pending:
            raise AssertionError("device trap: commit without a forward")
        count = int(selected.item())
        self.position, self.pending = self.start + count, False
        self.commits.append((self.start, count))


def _stage_model(start, end, drop_guard=False, ratio2=True):
    def layer(i):
        # ratio-2 compressors on even layers (none when ratio2 is False)
        state = _State() if ratio2 and i % 2 == 0 else None
        compressor = None if state is None else SimpleNamespace(ratio=2, state=state)
        return SimpleNamespace(attn=SimpleNamespace(compressor=compressor))

    model = SimpleNamespace(
        start_layer=start,
        end_layer=end,
        layers=[layer(i) if start <= i < end else None for i in range(LAYERS)],
        hash_state=_State() if start == 0 else None,
        # model.py: Engram (stage 0 here) or ratio-2 compressor state.
        owns_state=start == 0 or ratio2,
        stage_commit=StageCommit() if start == 0 and end < LAYERS else None,
        stage_commit_count=torch.zeros(1, dtype=torch.int64),
        pending_prefill=torch.zeros(1, dtype=torch.bool),
        pending_length=torch.zeros(1, dtype=torch.int64),
    )
    for name, fn in _model_methods(drop_guard).items():
        setattr(model, name, fn.__get__(model))
    model.states = [model.hash_state] if start == 0 else []
    model.states += [s for s in model._ratio2_states()]
    return model


def _forward(model, start, n, prefill):
    """What a forward does to the committed state: pending_* and hashing."""
    model.pending_prefill.fill_(prefill)
    model.pending_length.fill_(n)
    for state in model.states:
        state.forward(start)


def _async_request(model, prompt=10, chunk=8, accepted=(3, 2)):
    """One request through stage 1 under async scheduling, as the runner
    calls the model: before-forward hook, forward, after-forward hook, then
    (final prefill chunk and decode steps) the sampled-count receive."""
    position = 0
    while position < prompt:
        n = min(chunk, prompt - position)
        model.pp_before_forward(["a"], [position])
        _forward(model, position, n, True)
        model.pp_after_forward(["a"], [position], [prompt])
        if position + n == prompt:  # the final chunk samples the first token
            model.commit(torch.tensor([1]))
        position += n
    for count in accepted:
        model.pp_before_forward(["a"], [position])
        _forward(model, position, 1 + K, False)
        model.pp_after_forward(["a"], [position], [prompt])
        model.commit(torch.tensor([count]))
        position += count


def test_model_final_prefill_chunk_commits_once():
    # model.py's commit(): the receive after the final prefill chunk must
    # not commit again (pp_after_forward already did, with pending_length).
    model = _stage_model(0, 20)
    _async_request(model)
    for state in model.states:
        assert state.commits == [(0, 8), (8, 2), (10, 3), (13, 2)]
    assert not model.stage_commit.armed
    # Without the armed-flag guard the same calls double-commit and trap.
    unguarded = _stage_model(0, 20, drop_guard=True)
    try:
        _async_request(unguarded)
    except AssertionError as exc:
        assert "commit without a forward" in str(exc)
    else:
        raise AssertionError("unguarded commit() did not double-commit")


def test_model_sync_deferred_commit():
    model = _stage_model(0, 20)
    model.pp_before_forward(["a"], [0])
    _forward(model, 0, 10, True)
    model.pp_after_forward(["a"], [0], [10])
    model.pp_before_forward(["a"], [10])
    _forward(model, 10, 1 + K, False)
    model.pp_after_forward(["a"], [10], [10])
    model.pp_before_forward(["a"], [14])  # accepted 3 drafts + bonus
    for state in model.states:
        assert state.commits == [(0, 10), (10, 4)]


def test_model_single_and_last_stage():
    # TP4 (one stage): no StageCommit; the runner's commit always commits.
    single = _stage_model(0, LAYERS)
    assert single.stage_commit is None
    _forward(single, 0, 10, True)
    single.pp_after_forward(["a"], [0], [10])  # no-op without StageCommit
    single.commit(torch.tensor([1]))
    _forward(single, 10, 1 + K, False)
    single.commit(torch.tensor([4]))
    assert len(single.states) == 1 + LAYERS // 2
    for state in single.states:
        assert state.commits == [(0, 10), (10, 4)]
    # A stage without Engram or ratio-2 compressors holds no committed state;
    # its hooks and commit do nothing.
    last = _stage_model(20, LAYERS, ratio2=False)
    assert last.hash_state is None and last.stage_commit is None
    last.pp_before_forward(["a"], [0])
    last.pp_after_forward(["a"], [0], [10])
    last.commit(torch.tensor([1]))
    assert all(state.commits == [] for state in last.states)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print("PASS", test.__name__)
    print(f"{len(tests)} passed")
