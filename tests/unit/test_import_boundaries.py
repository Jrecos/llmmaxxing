"""CI-enforced layer boundaries via AST import analysis (not substring matching)."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN: dict[str, tuple[str, ...]] = {
    "src/llmmaxxing/core": ("fastapi", "starlette", "sqlalchemy", "httpx"),
    "src/llmmaxxing/gateway": ("llmmaxxing.control", "llmmaxxing.storage"),
    "src/llmmaxxing/control": ("llmmaxxing.gateway",),
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def _violations(root: str, forbidden: tuple[str, ...]) -> list[str]:
    found = []
    for path in sorted(Path(root).rglob("*.py")):
        for module in sorted(_imported_modules(path)):
            for name in forbidden:
                if module == name or module.startswith(name + "."):
                    found.append(f"{path}: imports {module!r} (forbidden: {name!r})")
    return found


def test_layer_import_boundaries() -> None:
    for root, names in FORBIDDEN.items():
        directory = Path(root)
        assert directory.is_dir(), f"layer directory {root} is missing"
        assert list(directory.rglob("*.py")), f"layer directory {root} contains no python"
        assert not _violations(root, names), "\n".join(_violations(root, names))
