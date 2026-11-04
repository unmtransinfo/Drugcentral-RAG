# Local-mode Python environment for the smoke test: Python 3.11 + requirements-local.lock only (no AWS packages).
# python:3.11-slim-trixie, multi-platform index pinned for reproducibility.
FROM python:3.11-slim-trixie@sha256:db20d8506e2f15b6db39b40b6015ca46bae855dec6dd326d462cfe79ae972893
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY --from=locallock requirements-local.lock /tmp/requirements-local.lock
RUN pip install -r /tmp/requirements-local.lock
RUN useradd --uid 10001 --create-home app
USER app
WORKDIR /src
