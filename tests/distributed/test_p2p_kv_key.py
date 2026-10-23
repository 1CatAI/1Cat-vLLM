"""Unit tests for P2pNccl KV tensor-key normalization."""

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_engine import (
    _kv_key,
)


def test_kv_key_strips_per_request_suffix():
    """Two instances append different random suffixes to the same base id."""
    prefill_side = "req1___decode_addr_10.0.0.1:14590___-80f66240"
    decode_side = "req1___decode_addr_10.0.0.1:14590___-abdaa5a1"
    assert _kv_key(prefill_side) == _kv_key(decode_side)
    assert _kv_key(prefill_side) == "req1___decode_addr_10.0.0.1:14590___"


def test_kv_key_keeps_plain_ids():
    """Ids without an 8-hex tail pass through untouched.

    Note: an id that legitimately ends in "-<8 hex chars>" is
    indistinguishable from the random suffix and will be stripped; the
    router is expected to use unambiguous base ids (e.g. uuid4().hex).
    """
    for rid in ("req-1234", "cmpl-abc", "deadbeef", "r-12345678-abc"):
        assert _kv_key(rid) == rid
