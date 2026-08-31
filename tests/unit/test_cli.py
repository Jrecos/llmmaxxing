from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

import pytest

import llmmaxxing.cli.main as cli
from llmmaxxing.cli.main import GatewayLaunch, build_parser


def test_cli_has_two_daemons():
    parser = build_parser()
    assert parser.parse_args(["gateway"]).command == "gateway"
    assert parser.parse_args(["control"]).command == "control"


def test_gateway_launches_created_app_with_one_uvicorn_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created = object()
    runtime = object()
    module = ModuleType("gateway_fixture")
    module.build = lambda: GatewayLaunch(  # type: ignore[attr-defined]
        app_kwargs={"contract": "mandatory-dependencies"},
        runtime_factory=lambda app: runtime if app is created else None,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cli, "create_app", lambda **_kwargs: created)
    observed: dict[str, Any] = {}

    class Config:
        def __init__(self, app: object, **kwargs: Any) -> None:
            observed["app"] = app
            observed.update(kwargs)

    class Server:
        def __init__(self, config: object) -> None:
            observed["config"] = config

        def run(self) -> None:
            observed["ran"] = True

    monkeypatch.setattr(cli.uvicorn, "Config", Config)
    monkeypatch.setattr(cli.uvicorn, "Server", Server)

    assert (
        cli.main(
            [
                "gateway",
                "--factory",
                "gateway_fixture:build",
                "--host",
                "127.0.0.1",
                "--port",
                "4400",
            ]
        )
        == 0
    )
    assert observed["app"] is runtime
    assert observed["workers"] == 1
    assert observed["reload"] is False
    assert observed["lifespan"] == "on"
    assert observed["timeout_graceful_shutdown"] == 600
    assert observed["host"] == "127.0.0.1"
    assert observed["port"] == 4400
    assert observed["ran"] is True


@pytest.mark.parametrize(
    "factory",
    ["missing-colon", "gateway_fixture:missing", "gateway_fixture:not_callable"],
)
def test_gateway_factory_is_mandatory_and_exact(
    monkeypatch: pytest.MonkeyPatch,
    factory: str,
) -> None:
    module = ModuleType("gateway_fixture")
    module.not_callable = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises((SystemExit, ValueError, TypeError, AttributeError, ModuleNotFoundError)):
        cli.main(["gateway", "--factory", factory])
