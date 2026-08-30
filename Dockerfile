# syntax=docker/dockerfile:1

# --- Stage 1: build UI assets with Node ---
FROM node:22-alpine AS ui-build
WORKDIR /build/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui/ ./
RUN npm test && npm run build

# --- Stage 2: build the Python wheel ---
FROM python:3.12-slim AS wheel-build
WORKDIR /build
RUN pip install --no-cache-dir build hatchling
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m build --wheel --no-isolation --outdir /wheels

# --- Stage 3: runtime ---
FROM python:3.12-slim AS runtime
RUN groupadd -r llmmaxxing && useradd -r -g llmmaxxing llmmaxxing \
    && mkdir -p /var/lib/llmmaxxing \
    && chown llmmaxxing:llmmaxxing /var/lib/llmmaxxing
COPY --from=wheel-build /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl
COPY --from=ui-build /build/ui/dist /app/ui/dist
ENV LLMMAXXING_STATE_DIR=/var/lib/llmmaxxing
VOLUME /var/lib/llmmaxxing
USER llmmaxxing
WORKDIR /app
ENTRYPOINT ["llmmaxxing"]
CMD ["--version"]
