# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native MiniMax H3 video/audio generation (optional ``video`` dependencies).

Keep this module lightweight: importing vLLM's text models must not import
video codecs, diffusion models, or initialize a CUDA device.
"""
