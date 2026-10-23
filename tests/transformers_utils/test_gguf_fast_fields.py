# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fast metadata parser must reproduce gguf-py's fields exactly."""

import gguf
import numpy as np

from vllm.transformers_utils.gguf_config import read_gguf_metadata
from vllm.transformers_utils.gguf_tensor_reader import GGUFReader


class _UpstreamReader(gguf.GGUFReader):
    pass


def write_model(path):
    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_uint32("test.scalar", 7)
    writer.add_string("test.string", "héllo")
    writer.add_array("tokenizer.ggml.tokens", ["a", "", "ü€", "token"] * 50)
    writer.add_array("tokenizer.ggml.token_type", [1, 2, 3, 4] * 50)
    writer.add_array("test.floats", [0.5, -1.25, 3.0])
    writer.add_array("test.bools", [True, False, True])
    writer.add_array("test.nested", [[1, 2], [3]])
    writer.add_tensor("w", np.arange(64, dtype=np.float32).reshape(2, 32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_fast_fields_match_upstream_structure(tmp_path):
    path = tmp_path / "m.gguf"
    write_model(path)
    fast, upstream = GGUFReader(path), _UpstreamReader(path)
    assert list(fast.fields) == list(upstream.fields)
    for name, expected in upstream.fields.items():
        actual = fast.fields[name]
        assert actual.offset == expected.offset, name
        assert actual.types == expected.types, name
        assert actual.data == expected.data, name
        assert len(actual.parts) == len(expected.parts), name
        for a, b in zip(actual.parts, expected.parts):
            assert a.dtype == b.dtype and a.shape == b.shape, name
            np.testing.assert_array_equal(a, b)
        assert actual.contents() == expected.contents(), name
    assert fast.data_offset == upstream.data_offset
    assert [t.name for t in fast.tensors] == [t.name for t in upstream.tensors]
    np.testing.assert_array_equal(fast.tensors[0].data, upstream.tensors[0].data)


def test_metadata_reader_uses_fast_fields(tmp_path):
    path = tmp_path / "m.gguf"
    write_model(path)
    metadata = read_gguf_metadata(path)
    assert metadata["tokenizer.ggml.tokens"][:4] == ["a", "", "ü€", "token"]
    assert metadata["tokenizer.ggml.token_type"][:4] == [1, 2, 3, 4]
