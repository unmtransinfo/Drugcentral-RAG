# Local-mode smoke test (opt-in)

An end-to-end check of `RAG_PIPELINE=local` on a synthetic three-drug database: embedding, ingestion, collection
stamping, readiness, Swagger UI, bearer authentication, chat and the CLI. It runs only when started explicitly;
`python -m unittest discover -s tests -t .` does not collect it. It uses no AWS account and no DrugCentral data.

```bash
tests/integration/run.sh                                   # from any directory; defaults below
ITEST_API_PORT=18080 tests/integration/run.sh              # another localhost port
ITEST_CHAT_MODEL=qwen2.5:14b ITEST_OLLAMA_MEMORY=14g tests/integration/run.sh
```

Exit status: 0 all checks passed, 1 a check failed, 2 refused to start (prerequisite, name or port problem).

## Chat model: smoke-test default vs. application default

| | Chat model | Status |
|---|---|---|
| This smoke test (`ITEST_CHAT_MODEL`) | `qwen2.5:3b` (~1.9 GB) | Passed on the platform below |
| The application (`OLLAMA_CHAT_MODEL` in `ollama.env.example`) | `qwen2.5:14b` (~9 GB) | Not run by this test so far |

The smoke test defaults to the smaller model so that it fits next to other workloads; a pass shows that the
components work together, not that `qwen2.5:14b` (or any model) gives good answers. The embedding model is always
`nomic-embed-text`, as in the application.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `ITEST_CHAT_MODEL` | `qwen2.5:3b` | Ollama chat model pulled and used for answers |
| `ITEST_API_PORT` | `18000` | Host port; the API is published on `127.0.0.1` only. The run stops if the port is in use |
| `ITEST_OLLAMA_MEMORY` | `4g` | Memory limit of the Ollama container (`qwen2.5:14b` needs about 12–14 GB) |
| `ITEST_PROJECT` | `dc-itest-<timestamp>-<pid>` | Compose project name; must start with `dc-itest-` and must not exist yet |
| `ITEST_KEEP` | `0` | `1` keeps the project for inspection and prints the command that removes it |

## Prerequisites

* Docker Engine on Linux (tested) or Docker Desktop (untested), with Docker Compose v2 supporting `!reset`
  (Compose 2.24 or later); `bash`, `curl`.
* Optional: `python3` (3.9 or later) on the host for the CLI check; it is skipped otherwise.
* Network access to Docker Hub, PyPI and the Ollama model registry.
* Disk: about 8 GB on the first run (Ollama image 4.3 GB, pgvector 0.5 GB, Python image with the local lock 0.4 GB,
  models 2.2 GB). Memory: about 6 GB free with the default model.

## What it does

1. Builds a Python 3.11 image from `environments/local/requirements-local.lock` (`app.Dockerfile`; no AWS packages).
2. Starts `db` and `ollama` from `environments/local/docker-compose.yml` (pinned Ollama digest) plus the overlay
   `compose.yml`: a source database loaded from `source-fixture.sql`, the API, and a small forwarder (`forward.py`).
3. Pulls `nomic-embed-text` and the chat model, then runs the checks in `checks.py`: isolation, embedding, ingestion
   (`python -m ingestion.vectors`), `/api/health`, `/api/ready`, `/docs`, `/openapi.json`, chat without, with a wrong
   and with the token, `/api/metrics`; then `/docs` and `python3 -m cli.ask` from the host on `127.0.0.1:$ITEST_API_PORT`.

Isolation: ingestion and the API run on an internal Docker network with no internet, no EC2 instance metadata and no
access to other Compose projects; only the forwarder publishes a localhost port. Database passwords and the API token
are generated per run in a private temporary directory, passed to `cli.ask` through its environment and never
printed; values in the caller's shell are ignored.

Cleanup: on exit, `docker compose -p <project> down -v --rmi local --remove-orphans` removes only that project's
containers, networks, volumes and the images it built; shared base images and models in other volumes are kept.
Models are therefore downloaded again on every run.

## Expected runtime

Measured on Linux arm64, 4 vCPU, no GPU, with `qwen2.5:3b`: 3 to 3.5 minutes per run with base images cached
(model download about 20 s on a fast connection; ingestion of 84 chunks under a minute; one chat answer about 80 s).
The first run adds the image downloads and the Python image build. Larger chat models and slower CPUs take longer.

## Tested and untested

Tested: Linux arm64 (AWS Graviton), Docker Compose 5.6.0, Ollama 0.40.14 in Docker, `qwen2.5:3b` +
`nomic-embed-text`, CPU only. Recorded runs: [docs/LOCAL_INTEGRATION_TEST.md](../../docs/LOCAL_INTEGRATION_TEST.md).

Not tested: amd64 (the pinned images support it); macOS or Windows with Docker Desktop; GPUs; Podman or rootless
Docker; Compose versions other than 5.6.0; `qwen2.5:14b`, `qwen2.5:7b` or any other chat model; a native (non-Docker)
Ollama; the real DrugCentral database; answer quality.
