# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Named diagnostic RPCs; default worker execution remains unchanged."""


class MtpAdmissionExtension:
    def install_mtp_node_annotations(self, folder):
        from benchmarks.sm70_mtp_node_trace import install

        return install(self, folder)

    def uninstall_mtp_node_annotations(self):
        from benchmarks.sm70_mtp_node_trace import uninstall

        return uninstall(self)

    def install_mtp_teacher_forcing(
        self, token_ids, prompt_length, prompt_sha256, folder
    ):
        from benchmarks.sm70_mtp_teacher_forcing import install

        return install(self, token_ids, prompt_length, prompt_sha256, folder)

    def flush_mtp_teacher_forcing(self, *, discard=False):
        from benchmarks.sm70_mtp_teacher_forcing import flush

        return flush(self, discard=discard)
