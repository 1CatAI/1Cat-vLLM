# Omni workflow coverage and H3-first migration

Audited 2026-09-08 against vLLM-Omni
`b58ff5cb8b17250b76f9cdf9b9b46385cdda4376`, committed at 09:10:51 UTC.
Its [supported-model table](https://github.com/vllm-project/vllm-omni/blob/b58ff5cb8b17250b76f9cdf9b9b46385cdda4376/docs/models/supported_models.md)
contains 90 rows, including separate variants of the same architecture. That
is not a claim of 90 distinct model families, nor of SM70 support.

The user prioritizes H3, while asking that future coverage follow official
workflows. Track complete pipelines rather than only model-name registrations.
The inherited `--omni` CLI forwards to a separately installed Omni package;
that delegation does not itself port any generator or validate it on V100.
The native `vllm video` engine currently specializes in H3.

## Official workflow groups

The examples below group the pinned official table. They are an inventory of
upstream capabilities, not claims that these pipelines run on native 1Cat.

| Workflow group | Official examples | Native 1Cat status / adaptation boundary |
| --- | --- | --- |
| Joint text/image/video/audio to video+audio | H3 T2VA, FL2VA, mixed Ref2VA | First priority; native pipeline exists, workflow and Turbo expansion is tracked below |
| Video generation and image conditioning | Wan2.1/2.2 T2V, TI2V, I2V; HunyuanVideo-1.5; SANA-Video; LingBot-Video | No corresponding native video pipeline in this scope; port preprocessing, DiT, scheduler and VAE together |
| Video control and speech-driven video | Wan S2V/VACE; LongCat-Video-Avatar A2V/AI2V/AVC | Distinct condition encoders/masks; ordinary T2V support is insufficient |
| One/two-stage and distilled video | LTX-2/2.3/2.5, Helios Base/Mid/Distilled; MAGI-2 preview | Track base/LoRA/upsampler combinations and stage-specific schedules independently |
| Images, editing and layers | Qwen-Image/Edit/Layered, FLUX/Kontext/klein, SDXL/SD3, Z-Image, GLM-Image, Krea, LongCat-Image, Boogu, ERNIE, HiDream | Separate text-to-image, image-edit and layered-output pipelines; no native H3-based alias |
| Multimodal understanding plus generation | Qwen2.5/3-Omni, Ming-flash-omni, BAGEL, MammothModa, SenseNova, MiniCPM-o, Dynin | Thinker/text support in vLLM does not establish talker/vocoder/image generation support |
| Speech synthesis and voice conditioning | Qwen3-TTS CustomVoice/VoiceDesign/Base, CosyVoice3, VoxCPM2, IndexTTS, Fish Speech, Higgs, MOSS, GLM-TTS, OmniVoice | Model-specific AR/flow/vocoder stage graphs and voice-conditioning contracts |
| Audio understanding, music and sound | MiMo Audio/ASR, Covo-Audio, Stable-Audio, MiniMax-Music3, MOSS-SoundEffect | Dedicated audio preprocessing, decoders and transport |
| World/action models | Cosmos3, SANA-WM, LingBot-World, DreamZero, Pi0, GR00T, InternVLA | Action/state conditioning and outputs require their own acceptance criteria |

## H3 adaptation history and ownership

Relevant tasks inspected:

- **确认 vLLM-Omni 支持 MiniMax H3**: initial upstream support audit; request for
  original BF16 and pruned signed INT8, TurboMind-style W8A16 and TP4; native
  port, FP32 range fixes, real text/DiT/VAE checks; subsequent FlashAttention-V100
  D128 and complete-denoise optimization.
- **确认 vLLM-Omni 支持 MiniMax H3 (2)**: the fork continuing FlashInfer operand
  feeding and the independent >80 useful TFLOPS/card optimization target.

Published review scopes at audit time:

- [#557](https://github.com/1CatAI/1Cat-vLLM/pull/557): native H3 model/pipeline/video service.
- [#558](https://github.com/1CatAI/1Cat-vLLM/pull/558): SM70 H3 kernels, stacked on #557.
- [#559](https://github.com/1CatAI/1Cat-vLLM/pull/559): quality and complete-denoise gates.
- [#564](https://github.com/1CatAI/1Cat-vLLM/pull/564): separate FlashInfer optimization.

The initial port came from Omni `7be014bce6374f06c95b703763bdbac4c6198f31`
and the Comfy loader in upstream PR #6894. It already contained text, keyframe
and reference packing, but excluded LoRA. Real short generation evidence was
primarily text-only. Original BF16 weights require FP16 matrix inputs plus FP32
residuals/modulation/output islands on V100. INT8 checkpoints additionally need
signed row scales, ConvRot256, QKV layout handling and pruned AdaLN tables.

Retained earlier short-run evidence reports 77.34 seconds / 44.78 useful
TFLOPS per card for FlashAttention-V100 and 86.20 seconds / 40.18 for a
FlashInfer control, at 39 frames and 20 updates. These are earlier development
runs, not current best-performance claims. Two-step base-model samples had
severe artifacts; this history is why a low step count must use actual distilled
weights. Human audio/temporal review and full 243-frame acceptance remain open.

## Current expansion

[H3 workflows and distilled LoRA](minimax_h3/WORKFLOWS.md) documents the executable
CLI/API contracts, eight LightX2V artifacts, FlashGen loading and FastH3 Dense
fusion. FlashGen/FastH3 have CPU validation; their GPU acceptance remains pending.
This branch is stacked on PR #557 and changes workflow/LoRA code, leaving kernel
optimization to the other tasks. It adds strict full-layout loading, TP-aware deltas,
artifact-specific sampling defaults, request-scale bypass, explicit task/shift
controls, reference-video offsets, and media validation before dispatch.

Acceptance has distinct levels:

1. Validate metadata, tasks, media limits, sigma schedule and adapter binding.
2. Check TP1/2/4 algebra, original-versus-rotated bases, full tensor consumption,
   staged buffers and zero-scale behavior.
3. Generate actual T2VA, keyframe FL2VA and mixed Ref2VA videos with the target
   weights; verify every rank's call count and decode fresh audio/video.
4. Review temporal/semantic/audio quality and repeat the agreed full-resolution,
   duration and seed gates before promotion.

H3's hosted Context-IR and Regenerate-2K services are described by MiniMax's
[official system overview](https://github.com/MiniMax-AI/MiniMax-H3).
They are not local open-weight stages in the audited Omni H3 recipe and are
not counted as ported by the native generator.

## Next implementation order

This is a proposed sequence based on reuse of the current code and the user's
H3-first priority, not a hardware-support assertion:

1. Finish H3 artifact/partition quality coverage, including FlashGen's dense
   AdaLN restoration and FastH3 Dense original-weight fusion. Extend FastH3
   INT8/VSA only with the corresponding weight and kernel validation.
   Add combined-partition serving only with
   a demonstrated host/GPU residency budget.
2. Extend diffusion coverage to Wan T2V/I2V and Qwen-Image/Edit with their official
   preprocessing and schedulers. Validate FP16 range and SM70 operator routing
   before importing further workflow variants.
3. Add LTX one/two-stage and distilled variants, including the matching LoRA and
   upsampler; independently check the second-stage output.
4. Bring in speech and any-to-any stage graphs with explicit talker/vocoder
   support. Treat interactive streaming, voice cloning and asynchronous stages
   as separate capabilities from offline generation.

For every added workflow retain source/model revisions, required adapters,
input/output schemas, quantization, TP/GPU set, scheduler contract, actual
stage/call counts, quality evidence and a smallest reproducible command.
