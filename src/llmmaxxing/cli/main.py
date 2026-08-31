"""Command dispatch for the llmmaxxing daemons."""

from __future__ import annotations

import argparse
import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import uvicorn

from llmmaxxing import __version__
from llmmaxxing.gateway.app import GatewayApp, create_app
from llmmaxxing.gateway.emergency import GatewayRuntime, GatewaySingleton


@dataclass(frozen=True, slots=True)
class GatewayLaunch:
    """Mandatory production dependency injection for Task8's strict ``create_app``."""

    app_kwargs: dict[str, Any]
    runtime_factory: Callable[[GatewayApp], Any]

    def __post_init__(self) -> None:
        if not self.app_kwargs or not callable(self.runtime_factory):
            raise ValueError("Gateway launch requires complete app dependencies and runtime")


def _load_gateway_launch(
    factory_path: str | None,
    singleton: GatewaySingleton,
) -> GatewayLaunch:
    if not factory_path:
        raise ValueError(
            "gateway requires --factory module:callable (or LLMMAXXING_GATEWAY_FACTORY)"
        )
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Gateway factory must be module:callable")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError("Gateway factory target is not callable")
    launch = factory(singleton)
    if not isinstance(launch, GatewayLaunch):
        raise TypeError("Gateway factory must return GatewayLaunch")
    return launch


def _run_gateway(args: argparse.Namespace) -> int:
    singleton = GatewaySingleton(args.data_dir)
    if not singleton.acquire():
        raise RuntimeError("another Gateway dispatcher owns the singleton")
    try:
        launch = _load_gateway_launch(args.factory, singleton)
        app = create_app(**launch.app_kwargs)
        runtime = launch.runtime_factory(app)
        if isinstance(runtime, GatewayRuntime) and runtime.singleton is not singleton:
            raise ValueError("Gateway runtime must reuse the pre-acquired singleton")
        config = uvicorn.Config(
            runtime,
            host=args.host,
            port=args.port,
            workers=1,
            reload=False,
            lifespan="on",
            timeout_graceful_shutdown=600,
        )
        uvicorn.Server(config).run()
        return 0
    finally:
        singleton.release()


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
    gateway = subparsers.add_parser("gateway", help="run the gateway daemon")
    gateway.add_argument(
        "--factory",
        default=os.environ.get("LLMMAXXING_GATEWAY_FACTORY"),
        help="mandatory module:callable returning GatewayLaunch",
    )
    gateway.add_argument(
        "--data-dir",
        default=os.environ.get("LLMMAXXING_GATEWAY_DATA_DIR", "/var/lib/llmmaxxing"),
    )
    gateway.add_argument("--host", default="0.0.0.0")
    gateway.add_argument("--port", type=int, default=4000)
    subparsers.add_parser("control", help="run the control daemon")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return DAEMONS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
