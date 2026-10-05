# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rank a draft shortlist by training-corpus frequency, never by evaluation.

Inspired by Strata's MIT-licensed draft-vocabulary approach; no Strata source
is copied. Coverage is a report, not an admission condition. Only measured
speculative acceptance and head latency can admit the resulting candidate.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


def rank_vocab(counts, size, special_ids, language_weights):
    mandatory = set(special_ids)
    if size < len(mandatory):
        raise ValueError("Subset cannot omit required special tokens")
    scores = defaultdict(float)
    for language, weight in language_weights.items():
        group = counts.get((language, "train"), Counter())
        total = sum(group.values())
        if not total:
            raise ValueError(f"No training tokens for {language}")
        for token, count in group.items():
            scores[token] += weight * count / total
    ranking = sorted(scores, key=lambda token: (-scores[token], token))
    if len(set(ranking) | mandatory) < size:
        raise ValueError("Insufficient observed training vocabulary")
    selected = list(sorted(mandatory))
    selected.extend(token for token in ranking if token not in mandatory)
    return sorted(selected[:size])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32768)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    documents = json.loads(args.corpus.read_text())
    counts = defaultdict(Counter)
    seen = set()
    for document in documents:
        text = document["text"]
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest in seen:
            raise ValueError("Duplicate text across corpus splits")
        seen.add(digest)
        split = document["split"]
        if split not in ("train", "heldout"):
            raise ValueError("Expected disjoint train/heldout corpus")
        counts[document["language"], split].update(
            tokenizer.encode(text, add_special_tokens=False)
        )
    # Normalize each language before mixing, so article length/count does not
    # implicitly erase the Chinese domain. Evaluation never changes this mix.
    weights = {"en": 0.4, "zh": 0.4, "code": 0.15, "ja": 0.025, "ko": 0.025}
    ids = rank_vocab(counts, args.size, tokenizer.all_special_ids, weights)
    selected = set(ids)
    statistics = [
        {
            "language": language,
            "split": split,
            "tokens": sum(group.values()),
            "unique_tokens": len(group),
            "coverage": sum(n for t, n in group.items() if t in selected)
            / sum(group.values()),
        }
        for (language, split), group in sorted(counts.items())
    ]
    result = {
        "token_ids": ids,
        "subset_size": len(ids),
        "language_weights": weights,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (args.tokenizer / "tokenizer.json").read_bytes()
        ).hexdigest(),
        "statistics": statistics,
        "coverage_is_admission_gate": False,
        "model_admission": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "token_ids"}))


if __name__ == "__main__":
    main()
