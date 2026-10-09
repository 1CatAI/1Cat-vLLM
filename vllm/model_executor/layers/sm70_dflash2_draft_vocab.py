# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original FP8 draft head over complete ASCII/code and CJK token subsets."""

import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open

from vllm import _sm70_ops as sm70_ops
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)
_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
# Also retain radicals, strokes, Bopomofo, compatibility and presentation forms.
_CJK_RANGES = (
    (0x1100, 0x11FF),
    (0x2E80, 0xA4CF),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE1F),
    (0xFE30, 0xFE4F),
    (0xFF00, 0xFFEF),
    (0x16FE0, 0x16FFF),
    (0x1AFF0, 0x1AFFF),
    (0x1B000, 0x1B16F),
    (0x1F200, 0x1F2FF),
    (0x20000, 0x3FFFF),
)


def _byte_decoder() -> dict[str, int]:
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(values)
    offset = 0
    for value in range(256):
        if value not in values:
            values.append(value)
            characters.append(256 + offset)
            offset += 1
    return {chr(character): value for value, character in zip(values, characters)}


def build_draft_token_ids(tokenizer: dict, base_ids: list[int]) -> list[int]:
    """Keep original IDs, all ASCII/code, CJK, byte fragments and specials."""
    if (
        tokenizer["model"]["type"] != "BPE"
        or tokenizer["decoder"]["type"] != "ByteLevel"
    ):
        raise ValueError("Draft vocabulary requires a byte-level BPE tokenizer")
    vocab = tokenizer["model"]["vocab"]
    specials = {token["id"] for token in tokenizer["added_tokens"]}
    valid = set(vocab.values()) | specials
    selected = set(base_ids) & valid
    selected.update(specials)
    decoder = _byte_decoder()
    for token, token_id in vocab.items():
        try:
            text = bytes(decoder[character] for character in token).decode("utf-8")
        except (UnicodeDecodeError, KeyError):
            # A whole-token Unicode test alone misses UTF-8 pieces needed for
            # rare CJK characters. Conservatively retain every such fragment.
            selected.add(token_id)
            continue
        # A frequency-selected English seed omits rare words and code pieces.
        # Retain every ASCII token independently of that seed's corpus.
        if text.isascii() or any(
            any(low <= ord(c) <= high for low, high in _CJK_RANGES) for c in text
        ):
            selected.add(token_id)
    return sorted(selected)


class SM70DFlash2DraftVocabHead(torch.nn.Module):
    def __init__(self, codes, scales, meta, token_ids, logical_rows):
        super().__init__()
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("scales", scales, persistent=False)
        self.register_buffer("token_ids", token_ids, persistent=False)
        self.k_ld = int(meta[0].item())
        self.q_ld = int(meta[1].item())
        self.logical_rows = logical_rows

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
        logits = torch.empty(
            (x.shape[0], self.token_ids.numel()), device=x.device, dtype=x.dtype
        )
        sm70_ops.fp8_gemm_sm70_out(
            logits, x, self.codes, self.scales, 128, self.k_ld, self.q_ld, False
        )
        return logits[:, : self.logical_rows]


def maybe_prepare_draft_vocab(draft, vllm_config) -> None:
    """Prepare a separate proposal-only head; target logits keep the full head."""
    spec = vllm_config.speculative_config
    head = draft.lm_head
    model = vllm_config.model_config
    text_config = model.hf_text_config
    if not (
        current_platform.is_device_capability(70)
        and vllm_config.parallel_config.tensor_parallel_size == 4
        and model.dtype == torch.float16
        and spec is not None
        and spec.num_speculative_tokens == 7
        and hasattr(draft.model, "candidate_selector")
        and getattr(text_config, "model_type", None) == "qwen3_5_text"
        and text_config.hidden_size == 5120
        and text_config.vocab_size == 248320
        and getattr(head, "sm70_fp8_turbomind", False)
        and getattr(head, "sm70_fp8_channel_scale", False)
        and not getattr(head, "sm70_fp8_qpn8", False)
    ):
        return
    folder = Path(model.model)
    tokenizer_path = folder / "tokenizer.json"
    checkpoint = folder / "model.safetensors"
    if not tokenizer_path.is_file() or not checkpoint.is_file():
        return
    tokenizer_bytes = tokenizer_path.read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != _TOKENIZER_SHA256:
        return
    # The English/code seed is from MIT-licensed Strata, commit
    # 6f32ec070f23ced9f50e704d854d775da52591ab. CJK coverage is recomputed
    # from this checkpoint's tokenizer; foreign and padded IDs are discarded.
    assets = Path(__file__).parents[2] / "assets"
    seed = (assets / "sm70_dflash2_draft_vocab_en.bin").read_bytes()
    base_ids = [value for (value,) in struct.iter_unpack("<i", seed)]
    selected = build_draft_token_ids(json.loads(tokenizer_bytes), base_ids)
    rank = get_tensor_model_parallel_rank()
    # Balance the new head independently of the target's contiguous TP shard.
    # Only the following compact value/ID pairs cross cards.
    local_ids = selected[rank::4]
    logical_rows = len(local_ids)
    physical_rows = (logical_rows + 127) // 128 * 128
    prefix = "lm_head."
    with safe_open(str(checkpoint), framework="pt", device="cpu") as tensors:
        weight_slice = tensors.get_slice(prefix + "weight")
        if (
            weight_slice.get_shape() != [248320, 5120]
            or weight_slice.get_dtype() != "F8_E4M3"
        ):
            return
        scales = tensors.get_tensor(prefix + "weight_scale")
        if scales.shape != (248320, 1):
            return
        # Bound CPU staging instead of materializing a full vocabulary head.
        codes = torch.zeros(physical_rows, 5120, dtype=torch.uint8)
        cursor = 0
        for start in range(0, 248320, 4096):
            stop = min(start + 4096, 248320)
            end = cursor
            while end < logical_rows and local_ids[end] < stop:
                end += 1
            if end > cursor:
                indices = torch.tensor(local_ids[cursor:end]) - start
                codes[cursor:end].copy_(
                    weight_slice[start:stop].view(torch.uint8)[indices]
                )
            cursor = end
        channel_scales = torch.zeros(physical_rows, 1, dtype=torch.float32)
        channel_scales[:logical_rows].copy_(scales[local_ids].float())
    device = head.weight.device
    packed, packed_scales, meta = sm70_ops.fp8_sm70_prepare(
        codes.to(device).view(torch.float8_e4m3fn),
        channel_scales.to(device),
        128,
        False,
    )
    mapping = torch.full((physical_rows,), -1, device=device, dtype=torch.int64)
    mapping[:logical_rows].copy_(torch.tensor(local_ids, device=device))
    draft.sm70_draft_vocab_head = SM70DFlash2DraftVocabHead(
        packed, packed_scales, meta, mapping, logical_rows
    )
    logger.info(
        "SM70 DFlash2 original-FP8 draft vocabulary enabled: %d global tokens, "
        "%d local tokens; complete ASCII/CJK/UTF-8-fragment coverage.",
        len(selected),
        logical_rows,
    )
