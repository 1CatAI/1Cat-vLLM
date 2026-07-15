# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import DraftTokenIds
from vllm.v1.spec_decode.ddtree_payload import DDTreeDraftPayload


class _Request:
    def __init__(self, *, is_prefill_chunk: bool = False) -> None:
        self.request_id = "r0"
        self.is_prefill_chunk = is_prefill_chunk
        self.spec_token_ids: list[int] = []
        self.structured_output_request = None
        self.max_tokens = 8
        self._output_token_ids: list[int] = []
        self.num_output_placeholders = 0

    def is_finished(self) -> bool:
        return False

    @property
    def num_output_tokens(self) -> int:
        return len(self._output_token_ids)


class _StructuredOutputManager:
    def __init__(self, *, advance: bool = False) -> None:
        self.advance = advance

    def should_advance(self, request: _Request) -> bool:
        return self.advance


class _Grammar:
    def validate_tokens(self, token_ids: list[int]) -> list[int]:
        return token_ids[:-1]


def _payload(flat_token_ids: tuple[int, ...] = (11, 21, 31)) -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=flat_token_ids,
        parent_indices=tuple(
            -1 if idx == 0 else idx - 1 for idx in range(len(flat_token_ids))
        ),
        node_depths=tuple(range(1, len(flat_token_ids) + 1)),
        node_scores=tuple(0.0 for _ in flat_token_ids),
        top1_chain_token_ids=flat_token_ids,
        flat_draft_token_ids=flat_token_ids,
        budget=len(flat_token_ids),
        top_k=1,
        chain_seed=True,
    )


def _branched_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 12),
        parent_indices=(-1, 0, -1),
        node_depths=(1, 2, 1),
        node_scores=(0.0, 0.0, -0.1),
        top1_chain_token_ids=(11, 21),
        flat_draft_token_ids=(11, 21),
        budget=3,
        top_k=2,
        chain_seed=True,
    )


def _token_matching_branched_payload() -> DDTreeDraftPayload:
    return DDTreeDraftPayload(
        tree_token_ids=(11, 21, 31),
        parent_indices=(-1, -1, 1),
        node_depths=(1, 1, 2),
        node_scores=(0.0, -0.1, -0.2),
        top1_chain_token_ids=(11, 21, 31),
        flat_draft_token_ids=(11, 21, 31),
        budget=3,
        top_k=2,
        chain_seed=True,
    )


def _scheduler(
    request: _Request,
    *,
    advance: bool = False,
    allow_branched_tree: bool = False,
) -> Scheduler:
    scheduler = object.__new__(Scheduler)
    scheduler.requests = {"r0": request}
    scheduler.ddtree_payloads_by_req_id = {}
    scheduler.dflash_ddtree_tree_verify = True
    scheduler.dflash_ddtree_allow_branched_tree = allow_branched_tree
    scheduler.structured_output_manager = _StructuredOutputManager(advance=advance)
    return scheduler


def _scheduler_output(
    *,
    scheduled_tokens: list[int],
    scheduled_payloads: dict[str, DDTreeDraftPayload] | None = None,
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={"r0": len(scheduled_tokens)},
        total_num_scheduled_tokens=len(scheduled_tokens),
        scheduled_spec_decode_tokens={"r0": scheduled_tokens},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        scheduled_ddtree_payloads=scheduled_payloads,
    )


def test_update_draft_token_ids_caches_matching_ddtree_payload() -> None:
    request = _Request()
    scheduler = _scheduler(request)
    payload = _payload()

    scheduler.update_draft_token_ids(
        DraftTokenIds(["r0"], [[11, 21, 31]], [payload])
    )

    assert request.spec_token_ids == [11, 21, 31]
    assert scheduler.ddtree_payloads_by_req_id == {"r0": payload}


def test_update_draft_token_ids_drops_payload_after_grammar_trim() -> None:
    request = _Request()
    request.structured_output_request = SimpleNamespace(grammar=_Grammar())
    scheduler = _scheduler(request, advance=True)
    payload = _payload()

    scheduler.update_draft_token_ids(
        DraftTokenIds(["r0"], [[11, 21, 31]], [payload])
    )

    assert request.spec_token_ids == [11, 21]
    assert scheduler.ddtree_payloads_by_req_id == {}


def test_update_draft_token_ids_drops_payload_for_prefill_chunk() -> None:
    request = _Request(is_prefill_chunk=True)
    request.spec_token_ids = [1, 2, 3]
    scheduler = _scheduler(request)
    payload = _payload()
    scheduler.ddtree_payloads_by_req_id = {"r0": payload}

    scheduler.update_draft_token_ids(
        DraftTokenIds(["r0"], [[11, 21, 31]], [payload])
    )

    assert request.spec_token_ids == []
    assert scheduler.ddtree_payloads_by_req_id == {}


def test_update_draft_token_ids_in_output_refreshes_matching_payload() -> None:
    request = _Request()
    scheduler = _scheduler(request)
    payload = _payload()
    scheduler_output = _scheduler_output(
        scheduled_tokens=[-1, -1, -1],
        scheduled_payloads={},
    )

    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds(["r0"], [[11, 21, 31]], [payload]),
        scheduler_output,
    )

    assert scheduler_output.scheduled_spec_decode_tokens["r0"] == [11, 21, 31]
    assert scheduler_output.scheduled_ddtree_payloads == {"r0": payload}


def test_update_draft_token_ids_in_output_drops_mismatched_payload() -> None:
    request = _Request()
    scheduler = _scheduler(request)
    payload = _payload()
    scheduler_output = _scheduler_output(
        scheduled_tokens=[-1, -1],
        scheduled_payloads={"r0": payload},
    )

    scheduler.update_draft_token_ids_in_output(
        DraftTokenIds(["r0"], [[11, 21, 31]], [payload]),
        scheduler_output,
    )

    assert scheduler_output.scheduled_spec_decode_tokens["r0"] == [11, 21]
    assert scheduler_output.scheduled_ddtree_payloads == {}


def test_ddtree_payload_for_tree_schedule_accepts_flat_chain() -> None:
    request = _Request()
    request.spec_token_ids = [11, 21, 31]
    scheduler = _scheduler(request)
    payload = _payload()
    scheduler.ddtree_payloads_by_req_id = {"r0": payload}

    assert scheduler._ddtree_payload_for_tree_schedule(request) is payload


def test_ddtree_payload_for_tree_schedule_rejects_branched_without_compaction() -> None:
    request = _Request()
    request.spec_token_ids = [11, 21]
    scheduler = _scheduler(request)
    payload = _branched_payload()
    scheduler.ddtree_payloads_by_req_id = {"r0": payload}

    assert scheduler._ddtree_payload_for_tree_schedule(request) is None


def test_ddtree_payload_for_tree_schedule_rejects_token_matching_branch() -> None:
    request = _Request()
    request.spec_token_ids = [11, 21, 31]
    scheduler = _scheduler(request)
    payload = _token_matching_branched_payload()
    scheduler.ddtree_payloads_by_req_id = {"r0": payload}

    assert scheduler._ddtree_payload_for_tree_schedule(request) is None


def test_ddtree_payload_for_tree_schedule_can_allow_branched_payload() -> None:
    request = _Request()
    request.spec_token_ids = [11, 21]
    scheduler = _scheduler(request, allow_branched_tree=True)
    payload = _branched_payload()
    scheduler.ddtree_payloads_by_req_id = {"r0": payload}

    assert scheduler._ddtree_payload_for_tree_schedule(request) is payload


def test_ddtree_tree_max_output_uses_max_depth_plus_bonus() -> None:
    payload = _token_matching_branched_payload()

    assert Scheduler._ddtree_payload_max_output_tokens(payload) == 3


def test_ddtree_remaining_output_subtracts_placeholders() -> None:
    request = _Request()
    request.max_tokens = 8
    request._output_token_ids = [1, 2, 3, 4, 5]
    request.num_output_placeholders = 2

    assert Scheduler._remaining_output_tokens(request) == 1
