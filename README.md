# LLMMaxxing

Apache-2.0 self-hosted admission, fair-queue, routing-policy and operations
control plane for [LiteLLM](https://github.com/BerriAI/litellm).

```text
Agents   → LLMMaxxing Gateway → private LiteLLM → providers
Operators → LLMMaxxing Control → immutable policy publication
```

One repository and one OCI image expose two long-running commands:

```bash
llmmaxxing gateway
llmmaxxing control
```

The Gateway is the only inference data plane. Control is out of band.
Gateway never reads Control, SQLite or PostgreSQL while handling inference.

## Status

Early skeleton. Normative design: [`docs/design.md`](docs/design.md).

## Development

```bash
uv sync --all-groups
uv run pytest
uv run llmmaxxing --version
```

## License

Apache-2.0. See [LICENSE](LICENSE).
