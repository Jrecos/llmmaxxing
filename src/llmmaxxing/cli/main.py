"""Command dispatch for the llmmaxxing daemons."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from llmmaxxing import __version__


def _run_gateway(_args: argparse.Namespace) -> int:
    raise NotImplementedError("gateway daemon ships in a later implementation task")


def _run_control(_args: argparse.Namespace) -> int:
    raise NotImplementedError("control daemon ships in a later implementation task")


DAEMONS: dict[str, Callable[[argparse.Namespace], int]] = {
    "gateway": _run_gateway,
    "control": _run_control,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmmaxxing",
        description="LiteLLM admission, fair-queue and routing-policy control plane",
    )
    parser.add_argument("--version", action="version", version=f"llmmaxxing {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="{gateway,control}")
    for name in DAEMONS:
        subparsers.add_parser(name, help=f"run the {name} daemon")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return DAEMONS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
