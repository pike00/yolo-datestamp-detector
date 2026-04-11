FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY --from=ghcr.io/casey/just:latest /just /usr/local/bin/just

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV UV_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV UV_EXTRA_INDEX_URL=https://pypi.org/simple/
ENV UV_INDEX_STRATEGY=first-index

COPY justfile ./
COPY scripts/train.py scripts/infer_all.py scripts/

CMD ["just", "cycle"]
