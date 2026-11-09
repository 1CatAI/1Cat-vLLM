# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serve the native Z-Image job API from verified local ModelScope components."""

from vllm.entrypoints.cli.types import CLISubcommand


class ImageSubcommand(CLISubcommand):
    name = "image"

    @staticmethod
    def cmd(args):
        import uvicorn

        from vllm.image.config import ImageConfig
        from vllm.image.server import create_app

        app = create_app(ImageConfig(args.model, args.checkpoint), args.output_dir)
        uvicorn.run(app, host=args.host, port=args.port)

    def subparser_init(self, subparsers):
        parser = subparsers.add_parser(
            self.name, help="Native image generation service"
        )
        parser.add_argument(
            "--model", required=True, help="Verified local Z-Image directory"
        )
        parser.add_argument(
            "--checkpoint",
            choices=["z-image-turbo", "z-image"],
            default="z-image-turbo",
        )
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--host", default="127.0.0.1")
        parser.add_argument("--port", type=int, default=8090)
        return parser


def cmd_init():
    return [ImageSubcommand()]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    command = ImageSubcommand()
    command.subparser_init(parser.add_subparsers(required=True))
    command.cmd(parser.parse_args())
