# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.backends.flex_attention import ddtree_logical_mask


def test_ddtree_logical_mask_hides_siblings() -> None:
    # Request starts tree verification at logical position 5.
    # Compact tree slots:
    # 0=root, 1=token 11, 2=token 21 child of 1, 3=token 12 sibling of 1.
    parent_ids = torch.tensor([[-1, -1, 1, -1]], dtype=torch.long)
    num_tree_tokens = torch.tensor([3], dtype=torch.long)
    decode_offset = torch.tensor([5], dtype=torch.long)
    q_req = torch.zeros(7, dtype=torch.long)
    logical_q_idx = torch.full((7,), 7, dtype=torch.long)  # node slot 2
    logical_kv_idx = torch.tensor([0, 4, 5, 6, 7, 8, 9], dtype=torch.long)
    base_mask = logical_q_idx >= logical_kv_idx

    mask = ddtree_logical_mask(
        base_mask=base_mask,
        q_req=q_req,
        logical_q_idx=logical_q_idx,
        logical_kv_idx=logical_kv_idx,
        decode_offset=decode_offset,
        parent_ids=parent_ids,
        num_tree_tokens=num_tree_tokens,
    )

    assert mask.tolist() == [
        True,   # history
        True,   # history
        True,   # root
        True,   # parent node 1
        True,   # self node 2
        False,  # sibling node 3
        False,  # beyond tree
    ]


def test_ddtree_logical_mask_uses_base_mask_for_non_tree_queries() -> None:
    parent_ids = torch.tensor([[-1, -1, 1]], dtype=torch.long)
    num_tree_tokens = torch.tensor([2], dtype=torch.long)
    decode_offset = torch.tensor([5], dtype=torch.long)
    q_req = torch.zeros(4, dtype=torch.long)
    logical_q_idx = torch.full((4,), 5, dtype=torch.long)  # root row
    logical_kv_idx = torch.tensor([4, 5, 6, 7], dtype=torch.long)
    base_mask = logical_q_idx >= logical_kv_idx

    mask = ddtree_logical_mask(
        base_mask=base_mask,
        q_req=q_req,
        logical_q_idx=logical_q_idx,
        logical_kv_idx=logical_kv_idx,
        decode_offset=decode_offset,
        parent_ids=parent_ids,
        num_tree_tokens=num_tree_tokens,
    )

    assert mask.equal(base_mask)
