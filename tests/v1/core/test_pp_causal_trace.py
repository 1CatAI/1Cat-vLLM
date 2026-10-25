# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only evidence integrity checks; imports no vLLM or CUDA modules."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from vllm.v1.core.sched.pp_causal_trace import PPCausalTrace as Trace


def request(name="chatcmpl-causal-test"):
    return NS(
        request_id=name,
        num_prompt_tokens=24,
        num_computed_tokens=24,
        num_output_tokens=1,
        num_tokens_with_spec=29,
        num_output_placeholders=0,
        num_in_flight_tokens=0,
        next_decode_eligible_step=2,
        status="RUNNING",
    )


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.trace = Trace(Path(self.temp.name), ("causal-",), 100)
        self.req = request()
        self.scheduler = NS(
            current_step=2,
            pp_feedback_depth=2,
            requests={self.req.request_id: self.req},
            running=[self.req],
            waiting=[],
        )

    def tearDown(self):
        self.trace.events.clear()
        self.trace.dropped = 0
        self.temp.cleanup()

    def output(self, tokens=5):
        return NS(
            scheduled_step=2,
            total_num_scheduled_tokens=tokens,
            num_scheduled_tokens={self.req.request_id: tokens} if tokens else {},
            scheduled_spec_decode_tokens={self.req.request_id: [-1] * 4}
            if tokens
            else {},
        )

    def test_disabled_has_no_trace_instance(self):
        self.assertIsNone(Trace.from_config(None))
        with self.assertRaisesRegex(ValueError, "dictionary"):
            Trace.from_config("not-a-config")
        with self.assertRaisesRegex(ValueError, "tag"):
            Trace.from_config({"directory": self.temp.name, "tags": []})

    def test_unmatched_warmup_is_not_collected(self):
        self.req.request_id = "warmup-0"
        self.scheduler.requests = {self.req.request_id: self.req}
        self.trace.schedule_begin(self.scheduler)
        self.trace.schedule_end(self.scheduler, self.output())
        self.assertEqual(self.trace.events, [])
        self.assertEqual(self.trace.pending, set())

    def test_snapshot_and_feedback_are_copied_before_mutation(self):
        output = self.output()
        self.trace.schedule_begin(self.scheduler)
        self.req.next_decode_eligible_step = 4
        self.trace.schedule_end(self.scheduler, output)
        tokens = [10, 11, 12]
        result = NS(
            sampled_token_ids=[tokens], req_id_to_index={self.req.request_id: 0}
        )
        self.trace.feedback_begin(self.scheduler, output, result)
        tokens.pop()
        self.assertEqual(
            self.trace.events[0]["requests"][self.req.request_id][
                "next_decode_eligible_step"
            ],
            2,
        )
        rows = self.trace.events[-1]["results"][self.req.request_id]
        self.assertEqual(rows["sampled_token_ids_before_stop_trim"], [10, 11, 12])
        self.assertEqual(rows["accepted_draft_tokens_before_stop_trim"], 2)
        self.assertEqual(rows["rejected_draft_tokens_before_stop_trim"], 2)
        self.assertEqual(tokens, [10, 11])
        self.scheduler.requests = {}
        self.trace.feedback_end(self.scheduler, output)
        files = list(Path(self.temp.name).glob("*.jsonl"))
        self.assertEqual(len(files), 1)
        saved = [json.loads(line) for line in files[0].read_text().splitlines()]
        self.assertEqual(saved[-1]["dropped_events"], 0)
        self.assertEqual(saved[-1]["pending_steps"], [])
        self.assertEqual(saved[-1]["events"], 4)

    def test_zero_token_control_is_explicit_and_paired(self):
        output = self.output(0)
        self.trace.schedule_begin(self.scheduler)
        self.trace.schedule_end(self.scheduler, output)
        self.assertFalse(self.trace.events[-1]["is_model_forward"])
        self.trace.feedback_begin(
            self.scheduler, output, NS(sampled_token_ids=[], req_id_to_index={})
        )
        self.trace.feedback_end(self.scheduler, output)
        self.assertFalse(self.trace.pending)

    def test_event_limit_is_reported_without_unbounded_buffer(self):
        self.trace.limit = 1
        self.trace.schedule_begin(self.scheduler)
        self.trace.schedule_end(self.scheduler, self.output())
        self.assertEqual(len(self.trace.events), 1)
        self.assertEqual(self.trace.dropped, 1)
        self.trace.flush()
        saved = [
            json.loads(line)
            for line in next(Path(self.temp.name).glob("*.jsonl"))
            .read_text()
            .splitlines()
        ]
        self.assertEqual(saved[-1]["dropped_events"], 1)
        self.assertEqual(saved[-1]["pending_steps"], [2])

    def test_unreturned_control_does_not_block_model_completion_flush(self):
        self.trace.schedule_begin(self.scheduler)
        self.trace.schedule_end(self.scheduler, self.output(0))
        self.scheduler.current_step = 3
        output = self.output()
        output.scheduled_step = 3
        self.trace.schedule_begin(self.scheduler)
        self.trace.schedule_end(self.scheduler, output)
        self.scheduler.requests = {}
        self.trace.feedback_end(self.scheduler, output)
        saved = [
            json.loads(line)
            for line in next(Path(self.temp.name).glob("*.jsonl"))
            .read_text()
            .splitlines()
        ]
        self.assertEqual(saved[-1]["pending_steps"], [])
        self.assertEqual(saved[-1]["unreturned_control_steps"], [2])


if __name__ == "__main__":
    unittest.main()
