# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Export contracts without requiring an unleased GPU during CPU CI."""

import pytest
import torch

from vllm.model_executor.models.minimax_h3.config import H3Config, H3InputError
from vllm.video import media


def test_nvenc_receives_rgba_in_rgb_order_and_preserves_device_selection(
    monkeypatch, tmp_path
):
    class Process:
        def __init__(self):
            self.stdin = self
            self.frames = []

        def write(self, frame):
            self.frames.append(bytes(frame))

        def close(self):
            pass

        def wait(self):
            return 0

    process = Process()
    commands = []

    def popen(command, **kwargs):
        commands.append(command)
        return process

    monkeypatch.setattr(media.subprocess, "Popen", popen)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "/test/ffmpeg-nvenc")
    # Non-contiguous input exposes incorrect stride handling and channel order.
    video = torch.arange(24, dtype=torch.uint8).reshape(1, 2, 2, 2, 3)
    video = video.transpose(2, 3)
    audio = torch.zeros(1, 1, 2667)
    result = media.export_video(
        video, audio, tmp_path, encoder="h264_nvenc", gpu_index=2
    )
    received = torch.tensor(list(b"".join(process.frames)), dtype=torch.uint8)
    received = received.reshape(1, 2, 2, 2, 4)
    assert torch.equal(received[..., :3], video)
    assert (received[..., 3] == 255).all()
    command = commands[0]
    assert command[0] == "/test/ffmpeg-nvenc"
    assert command[command.index("-gpu") + 1] == "2"
    assert command[command.index("-c:v") + 1] == "h264_nvenc"
    assert command[command.index("-rgb_mode") + 1] == "yuv420"
    # Only the raw input format is declared: an output -pix_fmt yuv420p
    # would insert a software conversion ahead of NVENC.
    assert command.count("-pix_fmt") == 1
    assert command[command.index("-pix_fmt") + 1] == "rgba"
    assert "-crf" not in command
    assert result["encoding_device"] == "cuda:2"
    assert result["preparation_device"] == "cpu"
    assert result["frames"] == 2
    assert set(result["export_stage_seconds"]) == {
        "prepare_and_copy",
        "audio_wav",
        "ffmpeg_start_and_feed",
        "ffmpeg_finish",
    }


def test_nvenc_failure_does_not_silently_switch_to_cpu(monkeypatch, tmp_path):
    class Process:
        stdin = None
        killed = False
        launches = 0

        def __init__(self, *args, **kwargs):
            self.stdin = self
            Process.launches += 1

        def write(self, frame):
            raise BrokenPipeError("GPU encoder unavailable")

        def kill(self):
            Process.killed = True

        def wait(self):
            return 1

    monkeypatch.setattr(media.subprocess, "Popen", Process)
    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", "/test/ffmpeg-nvenc")
    with pytest.raises(RuntimeError, match="h264_nvenc.*encode.log"):
        media.export_video(
            torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
            torch.zeros(1, 1, 1333),
            tmp_path,
            encoder="h264_nvenc",
        )
    assert Process.killed
    assert Process.launches == 1


def test_unknown_encoder_rejected_in_deployment_contract():
    with pytest.raises(H3InputError, match="video encoder"):
        H3Config(video_encoder="unrecognized")
