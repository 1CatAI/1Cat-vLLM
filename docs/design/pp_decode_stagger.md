# Optional staggering for a two-request PP decode tail

With Model Runner V2 and asynchronous pipeline parallelism, each scheduled
decode must wait for the configured feedback depth before its next execution.
At depth two, two requests that become ready together can remain in one batch:
both execute, the next scheduler step has no model work, and both become ready
together again. This can leave only one model batch outstanding even though the
executor permits two. CPU scheduling records establish this queue pattern;
they do not establish physical overlap between GPU kernels.

`--pp-decode-stagger` is an opt-in policy for this specific tail. It requires
pipeline parallelism, asynchronous scheduling, Model Runner V2, the default
scheduler, the multiprocessing executor, and a batch/feedback depth of two.
An unsupported target configuration raises an error at startup. The default
is disabled. For an otherwise complete serving command, add:

```bash
--pp-async-queue-depth 2 --pp-decode-stagger
```

The policy applies only when exactly two running requests are ready to decode,
both have completed prefill and produced output, and neither waiting queue
contains a request. After the first request successfully obtains KV slots and
is scheduled, the second is deferred for this one scheduler step. Its feedback
eligibility, speculative tokens, placeholders, and KV allocation are untouched.
On the following step the first request is not yet eligible, so the deferred
request can execute under the normal eligibility rule. Requests already in
opposite phases are unchanged. No request executes before its existing feedback
deadline.

The decision is recalculated on every step, including a transition from three
running requests to two. There are no persistent request-to-lane assignments to
clean up after EOS, abort, or preemption. A new prefill, a waiting request,
preemption during allocation, or a different resident count retains the
existing scheduler policy. Waiting work deliberately disables the optimization,
even when current capacity would temporarily prevent admitting it.

This policy trades batch width for more outstanding model work. Its effect on
throughput depends on the model, hardware, and pipeline balance. Enable it only
after comparing the same deployment and requests with the flag off and on.
The CPU state tests verify eligibility, feedback accounting, draft rejection,
completion, and queue behavior; they do not predict GPU speed. The feature
does not change the feedback depth or the executor's queue algorithm.

CPU tests use local metadata and require no model download:

```bash
CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 .venv/bin/python -m pytest --noconftest \
  tests/v1/core/test_pp_decode_stagger.py tests/config/test_pp_decode_stagger.py
```
