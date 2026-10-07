# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Remove dynamic RankData indexing from the norm and publication screens.

Retained SASS copies all eight pointers into a 64-byte thread-local array
before indexing the rank. Pass the local pointer separately; constant peer
indices remain in kernel parameters. Arithmetic and peer protocol stay intact.
"""

from build_sm70_qpn2_warp_publish import generate as publication_source
from sm70_tp4_norm_partial_packets import generate as packet_source


def rewrite_kernel(text, name):
    begin = text.index("void " + name + "(")
    body = text.index("{", begin)
    cursor, depth = body + 1, 1
    while depth:
        depth += (text[cursor] == "{") - (text[cursor] == "}")
        cursor += 1
    kernel = text[begin:cursor]
    if name == "warp_publish_down":
        old = "vllm::RankData buffers, int rank) {"
        new = "vllm::RankData buffers, int rank, const void* local_buffer) {"
    else:
        old = "float epsilon) {"
        new = "const void* local_buffer, float epsilon) {"
    assert kernel.count(old) == 1
    assert "buffers.ptrs[rank]" in kernel
    kernel = kernel.replace(old, new).replace("buffers.ptrs[rank]", "local_buffer")
    return text[:begin] + kernel + text[cursor:]


def generate(root, parts=5, publication=False):
    text = publication_source(root) if publication else packet_source(root, parts)
    names = ["partial_packet_norm"]
    if publication:
        names += ["consume_packet_norm", "warp_publish_down"]
    for name in names:
        text = rewrite_kernel(text, name)
    for name in names:
        if name == "warp_publish_down":
            old = "5120,4352,8,global,peers(buffers),rank);"
            new = "5120,4352,8,global,peers(buffers),rank,\n"
            new += "      reinterpret_cast<const void*>(buffers.at(rank)));"
        else:
            begin = text.index(name + "<float><<<")
            end = text.index(";", begin)
            call = text[begin:end]
            old = call
            assert call.count("rank,1e-6f)") == 1
            new = call.replace(
                "rank,1e-6f)",
                "rank,reinterpret_cast<const void*>(buffers.at(rank)),1e-6f)",
            )
        assert text.count(old) == 1
        text = text.replace(old, new)
    return text
