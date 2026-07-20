# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Higgs-Audio-v3 text-to-speech on Tesla V100 (SM70).

Runs ``bosonai/higgs-audio-v3-tts-4b`` with the ``FLASH_ATTN_V100`` backend and
the Stage-0 CUDA graph (low-latency) profile, which reaches real-time generation
on a single V100. See README.md for the kernel / vllm-omni requirements.

Example:
    python examples/generate/multimodal/higgs_audio_v3/tts.py \
        --text "Hello from a Tesla V100." \
        --deploy-config examples/generate/multimodal/higgs_audio_v3/higgs_v100_low_latency.yaml \
        --out higgs_out.wav
"""

import argparse

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default="Hello! This is Higgs Audio version three, "
                                     "generating speech on a Tesla V100.")
    p.add_argument("--model", default="bosonai/higgs-audio-v3-tts-4b")
    p.add_argument("--deploy-config", required=True,
                   help="Stage deploy YAML (see higgs_v100_low_latency.yaml).")
    p.add_argument("--out", default="higgs_out.wav")
    args = p.parse_args()

    from transformers import AutoTokenizer

    from vllm_omni.entrypoints.omni import Omni
    from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_tokenizer import (
        HiggsAudioV3TokenizerAdapter,
    )

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_token_ids = HiggsAudioV3TokenizerAdapter(tok).build_prompt(args.text)

    omni = Omni(model=args.model, deploy_config=args.deploy_config,
                trust_remote_code=True)
    out = omni.generate([{"prompt_token_ids": prompt_token_ids}])[0]

    mm = out.multimodal_output
    audio = mm.tensors["audio"].detach().cpu().float().numpy().reshape(-1)
    sr = int(mm.tensors["sr"])
    print(f"durations: {getattr(out, 'stage_durations', None)}")
    print(f"audio: {audio.size / max(sr, 1):.2f}s @ {sr} Hz")

    if audio.size:
        import soundfile as sf
        sf.write(args.out, audio, sr)
        print(f"wrote {args.out}")
    omni.close()


if __name__ == "__main__":
    main()
