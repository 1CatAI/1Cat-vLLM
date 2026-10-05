# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexed WY verification against a dense FP64 recurrence.

Tests use the shipped operator. Research runs may explicitly load an extension
built from this same source before invoking pytest; no runtime depends on it.
"""

import runpy
from pathlib import Path

import pytest
import torch

T, H, HV, K, V = 8, 4, 12, 128, 128


def dense(q, k, v, a, b, A_log, dt_bias, state, scale):
    q, k, v = q.double(), k.double(), v.double()
    q *= torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) * scale
    k *= torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q, k = q.repeat_interleave(3, 1), k.repeat_interleave(3, 1)
    g = -A_log.double().exp() * torch.nn.functional.softplus(
        a.double() + dt_bias.double()
    )
    beta = b.double().sigmoid()
    state = state.double().clone()
    outputs, states = [], []
    for i in range(q.size(0)):
        state *= g[i].exp()[:, None, None]
        residual = beta[i, :, None] * (v[i] - torch.einsum("hvk,hk->hv", state, k[i]))
        state += residual[:, :, None] * k[i, :, None, :]
        outputs.append(torch.einsum("hvk,hk->hv", state, q[i]))
        states.append(state.clone())
    return torch.stack(outputs).half(), states


def require_op():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires SM70")
    if not hasattr(torch.ops._C, "gdn_wy_verify_sm70_out"):
        pytest.skip("requires source-built WY operators")


@pytest.mark.parametrize("nseq", [1, 4, 8])
@pytest.mark.parametrize("variant", range(5))
def test_chained_reorder_prefix_commit(nseq, variant):
    require_op()
    torch.manual_seed(37)
    slots = 2 * nseq + 3
    # Match padded Mamba pages rather than assuming tightly packed states.
    state_storage = torch.randn(slots, HV * V * K + 64, device="cuda") * 0.03
    state = state_storage[:, : HV * V * K].view(slots, HV, V, K)
    initial = state.clone()
    u = torch.zeros(slots, 2, T, HV, V, device="cuda")
    kc = torch.zeros(slots, 2, T, H, K, device="cuda")
    G = torch.zeros(slots, 2, T, HV, device="cuda")
    A_log = torch.randn(HV, device="cuda") * 0.2 - 2
    dt_bias = torch.randn(HV, device="cuda") * 0.2
    selected = [1 + 2 * i for i in range(nseq)]
    active = {slot: initial[slot].double() for slot in selected}
    previous: dict[int, list[torch.Tensor]] = {}
    bank = 0
    for step in range(4):
        ordered = selected if step % 2 == 0 else selected[::-1]
        lengths = [1 + ((j * 3 + step) % T) for j in range(nseq)]
        # Include an empty padded request whose invalid slot must never be read.
        ids = torch.full((nseq + 1, T), -1, device="cuda", dtype=torch.int32)
        ids[:nseq, 0] = torch.tensor(ordered, device="cuda", dtype=torch.int32)
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        offsets.append(offsets[-1])
        cu = torch.tensor(offsets, device="cuda", dtype=torch.int32)
        acc = [
            0 if step == 0 else (j + step) % (len(previous[slot]) + 1)
            for j, slot in enumerate(ordered)
        ]
        accepted = torch.tensor(acc + [0], device="cuda", dtype=torch.int32)
        nt = offsets[-1]
        mixed = (
            torch.randn(nt, 2 * H * K + HV * V, device="cuda", dtype=torch.float16)
            * 0.3
        )
        q, k, v = mixed.split([H * K, H * K, HV * V], dim=1)
        a = torch.randn(nt, HV, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        out = torch.empty(nt, HV * V, device="cuda", dtype=torch.float16)
        expected = []
        next_previous = {}
        for j, slot in enumerate(ordered):
            if step and acc[j]:
                active[slot] = previous[slot][acc[j] - 1]
            lo, hi = offsets[j : j + 2]
            ref, states = dense(
                q[lo:hi].view(-1, H, K),
                k[lo:hi].view(-1, H, K),
                v[lo:hi].view(-1, HV, V),
                a[lo:hi],
                b[lo:hi],
                A_log,
                dt_bias,
                active[slot],
                K**-0.5,
            )
            expected.append(ref.reshape(-1, HV * V))
            next_previous[slot] = states
        torch.ops._C.gdn_wy_verify_sm70_out(
            out,
            state,
            u[:, 1 - bank],
            kc[:, 1 - bank],
            G[:, 1 - bank],
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ids,
            ids,
            None,
            cu,
            accepted,
            u[:, bank],
            kc[:, bank],
            G[:, bank],
            K**-0.5,
            variant,
        )
        torch.testing.assert_close(out, torch.cat(expected), atol=1e-4, rtol=2e-3)
        for slot in ordered:
            torch.testing.assert_close(
                state[slot].double(), active[slot], atol=2e-6, rtol=2e-5
            )
        previous = next_previous
        bank = 1 - bank
        # Save an interior prefix to independent physical blocks. Running state
        # and factors remain intact, so the next verify can accept another prefix.
        dest = torch.tensor(
            [slot + 1 for slot in ordered], device="cuda", dtype=torch.int32
        )
        counts = [max(1, length // 2) for length in lengths]
        prefix = torch.tensor(counts, device="cuda", dtype=torch.int32)
        torch.ops._C.gdn_wy_commit_sm70(
            state,
            u[:, bank],
            kc[:, bank],
            G[:, bank],
            ids[:nseq],
            dest,
            prefix,
            ids[:nseq],
        )
        for j, slot in enumerate(ordered):
            torch.testing.assert_close(
                state[slot + 1].double(),
                previous[slot][counts[j] - 1],
                atol=2e-6,
                rtol=2e-5,
            )
    # Exit speculation: materialize the final accepted state in place.
    src = torch.tensor(ordered, device="cuda", dtype=torch.int32)
    counts = torch.tensor(lengths, device="cuda", dtype=torch.int32)
    torch.ops._C.gdn_wy_commit_sm70(
        state, u[:, bank], kc[:, bank], G[:, bank], src, src, counts, src
    )
    for slot in ordered:
        torch.testing.assert_close(
            state[slot].double(), previous[slot][-1], atol=2e-6, rtol=2e-5
        )
    torch.testing.assert_close(state[0], initial[0], rtol=0, atol=0)


def test_commit_zero_copy_and_graph():
    require_op()
    state = torch.randn(3, HV, V, K, device="cuda")
    original = state.clone()
    u = torch.zeros(3, T, HV, V, device="cuda")
    k = torch.zeros(3, T, H, K, device="cuda")
    G = torch.zeros(3, T, HV, device="cuda")
    src = torch.tensor([0, -1], device="cuda", dtype=torch.int32)
    dst = torch.tensor([1, -1], device="cuda", dtype=torch.int32)
    acc = torch.zeros(2, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.ops._C.gdn_wy_commit_sm70(state, u, k, G, src, dst, acc, src)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gdn_wy_commit_sm70(state, u, k, G, src, dst, acc, src)
    graph.replay()
    torch.testing.assert_close(state[1], original[0], atol=0, rtol=0)
    torch.testing.assert_close(state[2], original[2], atol=0, rtol=0)


@pytest.mark.parametrize("runner", ["v1", "v2", "v2_triton"])
@pytest.mark.parametrize("same_block", [False, True])
def test_batched_commit_saves_boundary_and_running_state(runner, same_block):
    require_op()
    torch.manual_seed(91)
    nreq, nlayers, slots = 8, 3, 19
    states, factors, pending, descriptors, conv, conv_desc = [], [], [], [], [], []
    expected, conv_expected = [], []
    source = torch.tensor(
        [1 + 2 * i for i in range(nreq)], device="cuda", dtype=torch.int32
    )
    dest = source + 1
    tables = torch.stack([dest, source], dim=1).contiguous()
    batch_accepted = torch.tensor(
        [8, 7, 6, 5, 4, 3, 2, 3], device="cuda", dtype=torch.int32
    )
    base_computed = 504 if same_block else 249
    computed = torch.full((nreq,), base_computed, device="cuda", dtype=torch.int32)
    scheduled = torch.full_like(computed, 8)
    draft = torch.full_like(computed, 7)
    mapping = torch.tensor([6, 1, 5, 0, 4, 3, 2, 7], device="cuda", dtype=torch.int32)
    accepted = torch.empty_like(batch_accepted)
    updated_computed = torch.empty_like(computed)
    accepted[mapping.long()] = batch_accepted
    updated_computed[mapping.long()] = computed + batch_accepted
    snapshot = accepted.clone()
    next_counts = (
        accepted.clone() if runner.startswith("v2") else batch_accepted.clone()
    )
    next_selectors = next_counts.clone()
    for layer in range(nlayers):
        state = torch.randn(slots, HV, V, K, device="cuda") * 0.03
        u = torch.randn(nreq, T, HV, V, device="cuda") * 0.05
        k = torch.randn(nreq, T, H, K, device="cuda") * 0.04
        G = (-torch.rand(nreq, T, HV, device="cuda") * 0.1).cumsum(1)
        live = source.clone()
        live[-1] = -1  # plain decode/prefill has no deferred WY update
        states.append(state)
        factors.append((u, k, G))
        pending.append(live)
        descriptors.append(
            [
                state.data_ptr(),
                state.stride(0),
                u.data_ptr(),
                k.data_ptr(),
                G.data_ptr(),
                live.data_ptr(),
                0,
                slots,
                nreq,
            ]
        )
        history = torch.randn(slots, 10, 2560, device="cuda", dtype=torch.float16)
        conv.append(history)
        conv_desc.append([history.data_ptr(), history.stride(0), history.size(2)])
        conv_expected.append(
            [
                history[
                    source[i], int(batch_accepted[i]) - 1 : int(batch_accepted[i]) + 2
                ].clone()
                if i < nreq - 1
                else history[source[i], :3].clone()
                for i in range(nreq)
            ]
        )
        refs = []
        for req in range(nreq):

            def materialize(count, G=G, req=req, state=state, u=u, k=k):
                g = G[req, count - 1].double()
                result = state[source[req]].double() * g.exp()[:, None, None]
                for i in range(count):
                    coef = (g - G[req, i].double()).exp()[:, None] * u[req, i].double()
                    kk = k[req, i].double().repeat_interleave(3, 0)
                    result += coef[:, :, None] * kk[:, None, :]
                return result

            refs.append(
                (
                    materialize(int(batch_accepted[req])),
                    materialize(7),
                    state[source[req]].clone(),
                    state[dest[req]].clone(),
                )
            )
        expected.append(refs)
    desc = torch.tensor(descriptors, device="cuda", dtype=torch.int64)
    cd = torch.tensor(conv_desc, device="cuda", dtype=torch.int64)
    table_ptrs = torch.tensor([tables.data_ptr()], device="cuda", dtype=torch.int64)
    if runner.startswith("v2"):
        torch.ops._C.gdn_wy_commit_group_v2_sm70(
            states,
            pending,
            desc,
            table_ptrs,
            tables.stride(0),
            accepted,
            updated_computed,
            mapping,
            256,
        )
    else:
        torch.ops._C.gdn_wy_commit_group_sm70(
            states,
            pending,
            desc,
            table_ptrs,
            tables.stride(0),
            batch_accepted,
            scheduled,
            computed,
            draft,
            256,
            False,
        )
    # Existing postprocess consumes convolution prefixes before finish shifts
    # the running window. Exercise both same-block and distinct destinations.
    if runner == "v2_triton":
        # Exercise the production copy kernel, including overlapping SD-layout
        # convolution copies and the collapsed temporal-state copy guard.
        module = runpy.run_path(
            str(
                Path(__file__).resolve().parents[3]
                / "vllm/v1/worker/gpu/mamba_align.py"
            )
        )
        tensors = [tensor for pair in zip(conv, states) for tensor in pair]

        def metadata(values, dtype):
            return torch.tensor(values, device="cuda", dtype=dtype)

        kernel = module["_postprocess_mamba_align_kernel"]
        kernel[(nreq, len(tensors), 16)](
            snapshot,
            next_counts,
            torch.ones(nreq, device="cuda", dtype=torch.int32),
            updated_computed,
            table_ptrs,
            tables.stride(0),
            metadata([t.data_ptr() for t in tensors], torch.int64),
            metadata([t.stride(0) * t.element_size() for t in tensors], torch.int64),
            metadata([t.element_size() for t in tensors], torch.int32),
            metadata([2560, HV * V * K] * nlayers, torch.int64),
            metadata([10, 0] * nlayers, torch.int32),
            metadata([0] * len(tensors), torch.int32),
            metadata([False, True] * nlayers, torch.bool),
            mapping,
            nreq,
            MAMBA_BLOCK_SIZE=256,
            COPY_BLOCK_SIZE=1024,
            TEMPORAL_TILES=16,
        )
    else:
        for history in conv:
            if same_block:
                history[source[0], :3].copy_(history[source[0], 7:10].clone())
            else:
                for req in (0, 1):
                    history[dest[req], :4].copy_(history[source[req], 6:10])
    if runner.startswith("v2"):
        torch.ops._C.gdn_wy_finish_group_v2_sm70(
            pending,
            conv,
            desc,
            cd,
            table_ptrs,
            tables.stride(0),
            snapshot,
            updated_computed,
            mapping,
            256,
            next_counts,
        )
    else:
        torch.ops._C.gdn_wy_finish_group_sm70(
            pending,
            conv,
            desc,
            cd,
            table_ptrs,
            tables.stride(0),
            batch_accepted,
            scheduled,
            computed,
            draft,
            256,
            next_counts,
            next_selectors,
        )
    for layer in range(nlayers):
        for req in range(nreq):
            full, prefix, oldsource, olddest = expected[layer][req]
            if req == nreq - 1:
                torch.testing.assert_close(
                    states[layer][source[req]], oldsource, rtol=0, atol=0
                )
            else:
                torch.testing.assert_close(
                    states[layer][source[req]].double(), full, rtol=2e-5, atol=2e-6
                )
            if not same_block and req <= 1:
                torch.testing.assert_close(
                    states[layer][dest[req]].double(), prefix, rtol=2e-5, atol=2e-6
                )
            else:
                torch.testing.assert_close(
                    states[layer][dest[req]], olddest, rtol=0, atol=0
                )
            torch.testing.assert_close(
                conv[layer][source[req], :3], conv_expected[layer][req], rtol=0, atol=0
            )
        assert torch.equal(pending[layer], torch.full_like(pending[layer], -1))
    expected_counts = torch.ones_like(next_counts)
    expected_counts[int(mapping[-1]) if runner.startswith("v2") else nreq - 1] = 3
    torch.testing.assert_close(next_counts, expected_counts, rtol=0, atol=0)


def test_single_bank_scattered_request_mask_and_padded_rows():
    require_op()
    torch.manual_seed(13)
    state = torch.randn(20, HV, V, K, device="cuda") * 0.03
    active = state.double().clone()
    u = torch.empty(8, T, HV, V, device="cuda")
    kcache = torch.empty(8, T, H, K, device="cuda")
    G = torch.empty(8, T, HV, device="cuda")
    pending = torch.full((8,), -1, device="cuda", dtype=torch.int32)
    masks = torch.tensor(
        [False, True, False, True, True, False, True, False], device="cuda"
    )
    physical = [17, 3, 8, 12]
    factors = torch.tensor([1, 3, 4, 6], device="cuda", dtype=torch.int32)
    ids = torch.tensor(physical + [-1, -1, -1, -1], device="cuda", dtype=torch.int32)
    cu = torch.tensor(
        [0, 8, 16, 24, 32, 32, 32, 32, 32], device="cuda", dtype=torch.int32
    )
    zero = torch.zeros(8, device="cuda", dtype=torch.int32)
    A_log = torch.randn(HV, device="cuda") * 0.1 - 2
    dt_bias = torch.randn(HV, device="cuda", dtype=torch.float16) * 0.1
    for step in range(3):
        mixed = torch.randn(32, 2560, device="cuda", dtype=torch.float16) * 0.25
        q, k, v = mixed.split([512, 512, 1536], dim=1)
        a = torch.randn(32, HV, device="cuda", dtype=torch.float16)
        b = torch.randn_like(a)
        out = torch.empty(32, 1536, device="cuda", dtype=torch.float16)
        refs = []
        accepted = [1 + ((step + j * 3) % 8) for j in range(4)]
        for j, slot in enumerate(physical):
            region = slice(j * 8, (j + 1) * 8)
            ref, states = dense(
                q[region].view(8, H, K),
                k[region].view(8, H, K),
                v[region].view(8, HV, V),
                a[region],
                b[region],
                A_log,
                dt_bias,
                active[slot],
                K**-0.5,
            )
            refs.append(ref.reshape(8, 1536))
            active[slot] = states[accepted[j] - 1]
        # Same factor tensors are legal here: previous acceptance is zero,
        # and state was published by the preceding iteration's commit.
        torch.ops._C.gdn_wy_verify_sm70_out(
            out,
            state,
            u,
            kcache,
            G,
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            ids,
            masks,
            pending,
            cu,
            zero,
            u,
            kcache,
            G,
            K**-0.5,
            3,
        )
        torch.testing.assert_close(out, torch.cat(refs), atol=1e-4, rtol=2e-3)
        torch.testing.assert_close(pending[factors], ids[:4], atol=0, rtol=0)
        counts = torch.tensor(accepted, device="cuda", dtype=torch.int32)
        torch.ops._C.gdn_wy_commit_sm70(
            state, u, kcache, G, ids[:4], ids[:4], counts, factors
        )
        for slot in physical:
            torch.testing.assert_close(
                state[slot].double(), active[slot], atol=2e-6, rtol=2e-5
            )
        pending.fill_(-1)
