from pathlib import Path

FORBIDDEN = {
    "src/llmmaxxing/core": ("fastapi", "starlette", "sqlalchemy", "httpx"),
    "src/llmmaxxing/gateway": ("llmmaxxing.control", "llmmaxxing.storage"),
    "src/llmmaxxing/control": ("llmmaxxing.gateway",),
}

def test_layer_import_boundaries():
    for root, names in FORBIDDEN.items():
        text = "\n".join(p.read_text() for p in Path(root).rglob("*.py"))
        assert not any(name in text for name in names)
