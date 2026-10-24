# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence

from vllm.v1.request import Request, RequestStatus


class EarlyPrefillQuota:
    """Briefly repay admission skew without changing request order or ledgers.

    Admission requires zero-start successful scheduling within one PP feedback
    window. Normal feedback is allowed: computed positions count *successful
    scheduling*, not completed GPU execution. Failed allocations and clipped
    chunks therefore never earn fictitious service in this planner.
    """

    def __init__(self) -> None:
        self.members: tuple[str, ...] = ()
        self.priority = 0
        self.expires_step = 0
        self.retired: frozenset[str] = frozenset()
        # request -> (first successful scheduling step, actual scheduled work).
        # Entries expire after one feedback window unless a group was frozen.
        self._service: dict[str, tuple[int, int]] = {}

    @staticmethod
    def _partial(request: Request) -> bool:
        return (
            request.status == RequestStatus.RUNNING
            and request.is_prefill_chunk
            and 0 < request.num_computed_tokens < request.num_prompt_tokens
            and request.num_output_tokens == 0
            and request.num_output_placeholders == 0
            and request.num_preemptions == 0
            and request.last_prefix_cache_reset_step == 0
            and not request.spec_token_ids
        )

    def cancel(self) -> None:
        if self.members:
            self.retired = frozenset(self.members)
            self.members = ()
        self._service.clear()

    def observe(
        self,
        running: Sequence[Request],
        live_requests: Mapping[str, Request],
        scheduled: Mapping[str, int],
        *,
        admitted: frozenset[str],
        waiting: bool,
        step: int,
        budget: int,
        feedback_depth: int,
        enabled: bool,
    ) -> None:
        # Observe only the output that survived allocation/preemption, after
        # AsyncScheduler's actual successful scheduling updates. No quota is
        # charged in advance, and no scheduler/request ledger is written here.
        if self.retired and not any(rid in live_requests for rid in self.retired):
            self.retired = frozenset()
        if not enabled or waiting or feedback_depth <= 1:
            self.cancel()
            return
        if self.retired:
            return
        service: dict[str, tuple[int, int]] = {}
        for request in running:
            rid = request.request_id
            count = scheduled.get(rid, 0)
            prior = self._service.get(rid)
            if not self._partial(request):
                continue
            if prior is not None:
                first, actual = prior
                actual += count
            elif rid in admitted and count > 0:
                first, actual = step, count
            else:
                continue
            # A cache hit/nonzero initial position or later rollback cannot
            # masquerade as initially uncached service. Feedback alone does
            # not alter computed positions during a pure partial prefill.
            if actual != request.num_computed_tokens:
                continue
            if not self.members and step - first > feedback_depth:
                continue
            service[rid] = (first, actual)
        self._service = service
        if self.members:
            if any(rid not in service for rid in self.members):
                self.cancel()
            return
        if not admitted:
            return
        if len(running) < 2 or not all(
            self._partial(request)
            and request.request_id in service
            and scheduled.get(request.request_id, 0) > 0
            and request.priority == running[0].priority
            for request in running
        ):
            return
        positions = [request.num_computed_tokens for request in running]
        skew = max(positions) - min(positions)
        if not 0 < skew <= budget * feedback_depth:
            return
        self.members = tuple(request.request_id for request in running)
        self.priority = running[0].priority
        # This is a scheduling-step deadline, not a progress-based extension.
        # Zero-work and clipped batches consume the same finite window.
        self.expires_step = step + 2 * len(running) * feedback_depth

    def plan(
        self,
        running: Sequence[Request],
        *,
        waiting: bool,
        step: int,
        nominal: int,
        budget: int,
    ) -> dict[str, int]:
        if not self.members:
            return {}
        if (
            waiting
            or step > self.expires_step
            or tuple(request.request_id for request in running) != self.members
            or not all(
                self._partial(request)
                and request.priority == self.priority
                and self._service.get(request.request_id, (-1, -1))[1]
                == request.num_computed_tokens
                for request in running
            )
            or nominal <= 1
            or nominal * len(running) > budget
        ):
            self.cancel()
            return {}

        positions = [request.num_computed_tokens for request in running]
        remaining = [
            request.num_prompt_tokens - request.num_computed_tokens
            for request in running
        ]
        # Every member retains half its ordinary share (subject to all normal
        # downstream guards); no request receives more than two ordinary shares.
        floor = [min(left, max(1, nominal // 2)) for left in remaining]
        upper = [min(left, 2 * nominal) for left in remaining]
        total = min(sum(min(left, nominal) for left in remaining), sum(upper))
        extra = total - sum(floor)

        def at_level(level: int) -> list[int]:
            return [
                max(low, min(high, level - position))
                for position, low, high in zip(positions, floor, upper)
            ]

        low = min(position + base for position, base in zip(positions, floor))
        high = max(position + cap for position, cap in zip(positions, upper)) + 1
        while low + 1 < high:
            mid = (low + high) // 2
            if sum(at_level(mid)) - sum(floor) <= extra:
                low = mid
            else:
                high = mid
        quotas = at_level(low)
        spare = total - sum(quotas)
        # Integer remainder follows original FCFS/priority order among equal
        # water levels. This does not reorder scheduler.running.
        for index in sorted(
            range(len(running)), key=lambda i: (positions[i] + quotas[i], i)
        ):
            if spare and quotas[index] < upper[index]:
                quotas[index] += 1
                spare -= 1
        assert spare == 0
        return dict(zip(self.members, quotas))
