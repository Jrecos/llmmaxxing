# syntax=docker/dockerfile:1

# --- Stage 1: build UI assets with Node ---
FROM node:22-alpine AS ui-build
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm test && npm run build

 # --- Stage 2: install Python package and dependencies pinned by uv.lock ---
 FROM python:3.12-slim AS python-build
 # same uv version that generated uv.lock, so --frozen never re-resolves
 RUN pip install --no-cache-dir uv==0.12.3
 ENV UV_LINK_MODE=copy
 WORKDIR /app
 COPY pyproject.toml uv.lock README.md LICENSE ./
 COPY src ./src
 # --frozen forbids re-resolution: every dependency comes from uv.lock
 RUN uv sync --frozen --no-dev --no-editable

 # --- Stage 3: runtime ---
 FROM python:3.12-slim AS runtime
 RUN groupadd -r llmmaxxing && useradd -r -g llmmaxxing llmmaxxing \
     && mkdir -p /var/lib/llmmaxxing \
     && chown llmmaxxing:llmmaxxing /var/lib/llmmaxxing
 COPY --from=python-build --chown=llmmaxxing:llmmaxxing /app/.venv /app/.venv
 COPY --from=ui-build /build/ui/dist /app/ui/dist
 ENV LLMMAXXING_STATE_DIR=/var/lib/llmmaxxing \
     PATH="/app/.venv/bin:$PATH"
 VOLUME /var/lib/llmmaxxing
 USER llmmaxxing
 WORKDIR /app
 ENTRYPOINT ["llmmaxxing"]
 CMD ["--version"]
