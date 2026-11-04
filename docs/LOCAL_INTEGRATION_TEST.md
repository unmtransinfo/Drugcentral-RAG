# Local-mode integration test record

**Date:** 2026-11-04. **Host:** Linux aarch64 (arm64), 4 vCPU, 15 GB RAM, no GPU. **Result:** all checks passed.

A small, isolated test of `RAG_PIPELINE=local` on disposable synthetic data. It is not full-data ingestion and not a
quality evaluation (see [What has been tested](LOCAL_DEVELOPMENT.md#what-has-been-tested)).

## Images and models

| Item | Value |
|---|---|
| Ollama image | `ollama/ollama@sha256:ebaef4f4c5d93660e1f25f8ad085a112f51ccfb0e8458073e71793959dba91c9`, reports `ollama version is 0.40.14`. Multi-platform index: `linux/amd64` (`sha256:bdf98cbc…2ea4`) and `linux/arm64` (`sha256:a2be4eed…46c5`); only arm64 was run |
| pgvector image | `pgvector/pgvector:pg17` (`sha256:ac08538c…a75d`), extension `vector` 0.8.7 |
| Python environment | `python:3.11-slim-trixie` (Python 3.11.18) + `environments/local/requirements-local.lock` (installed set identical to the lock; no `boto3`) |
| Embedding model | `nomic-embed-text:latest` (Ollama ID `0a109f422b47`), 768 dimensions |
| Chat model | `qwen2.5:3b` (Ollama ID `357c53fb659c`), **not** the default `qwen2.5:14b`: the host had about 8 GB free alongside other services and the Ollama container was limited to 4 GB |

## Isolation and data

* Services came from `environments/local/docker-compose.yml` under a separate Compose project (`dc-itest`) with
  fresh volumes. Ingestion and the API ran in containers attached only to a Docker `--internal` network.
* From that network: EC2 instance metadata, the internet, the host's other services and other Compose projects were
  unreachable; no `AWS_*` variables, no `~/.aws`, no AWS SDK installed.
* Source data: a synthetic PostgreSQL fixture with the 24 tables `ingestion/vectors.py` reads and three drugs
  (metformin, lisinopril, warfarin), with illustrative values. It is not DrugCentral data and is not committed.
* Database passwords and the API token were generated for the run and never printed. Containers, volumes and the
  network were removed afterwards.

## Commands (repository root; `.env` values as in `ollama.env.example`, hosts adapted to the network)

```bash
C="docker compose -p dc-itest -f environments/local/docker-compose.yml --env-file compose.env"
$C --profile ollama up -d db ollama
$C exec -T ollama ollama pull nomic-embed-text && $C exec -T ollama ollama pull qwen2.5:3b
docker network create --internal dc-itest-internal      # db, ollama and a synthetic source DB attached to it
# In the local-lock environment on that network (RAG_PIPELINE=local, OLLAMA_CHAT_MODEL=qwen2.5:3b, API_AUTH_TOKEN set):
python -c "from chatbot.embedding_config import create_embeddings; print(len(create_embeddings().embed_query('x')))"
python -m ingestion.vectors
python -m uvicorn api:app --host 0.0.0.0 --port 8000    # inside its container; no host port published
curl /api/health, /api/ready, /docs, /openapi.json; POST /api/chat without, with a wrong and with the token
python -m cli.ask --url http://api:8000 --check && python -m cli.ask --url http://api:8000 "Which drugs target ACE?"
```

## Results

| Check | Result |
|---|---|
| Embedding | `nomic-embed-text`, 768-dimensional vector |
| Ingestion | 3 dossiers → 84 chunks stored and committed; collection stamped `backend=ollama model=nomic-embed-text dimensions=768` with probe vector |
| `GET /api/health` | 200 `{"status":"ok"}` |
| `GET /api/ready` | 200 `provider=ollama`, `drugs=3`, `dataset_version=["itest-synthetic-3-drugs"]` |
| `GET /docs`, `GET /openapi.json` | 200 Swagger UI; bearer security declared on `/api/chat` |
| `POST /api/chat` without / with wrong token | 401 / 401 |
| `POST /api/chat` with token, "What is metformin indicated for?" | 200 in 79 s: "Metformin is indicated for Type 2 diabetes mellitus … Biguanide pharmacologic class", with `Sources:` |
| `GET /api/metrics` with token | 200 |
| `cli.ask --check` and a question | health ok, ready; "Lisinopril targets Angiotensin-converting enzyme (ACE)" with sources |

The API was reached over the internal network rather than `localhost`, because port 8000 on this host is in use.

## Unit tests per dependency group (same date, `--network none`, repository mounted read-only)

| Environment | Result |
|---|---|
| Python 3.11.18 + `requirements-local.lock` | 31 run, OK, 4 skipped (Bedrock request tests need `boto3`) |
| Python 3.11.18 + `requirements-aws.lock` | 31 run, OK, 1 skipped (local client test needs `langchain-ollama`) |

## Not established

The default chat model `qwen2.5:14b`; amd64; native (non-Docker) Ollama; ingestion from the real DrugCentral
database; answer quality; latency on other hardware.

## Later runs: `tests/integration/run.sh` (2026-11-04)

The fixture used above is now included as `tests/integration/source-fixture.sql`, with the harness described in
[tests/integration/README.md](../tests/integration/README.md). Same host, images and models as above (Docker
Compose 5.6.0, `qwen2.5:3b`, `nomic-embed-text`), unique Compose project per run, API published on `127.0.0.1`.

| Run | Result |
|---|---|
| `tests/integration/run.sh` (port 18000) | All service checks passed (isolation, ingestion of 84 chunks, readiness, `/docs`, 401/401/200 chat, metrics, host `/docs` and `cli.ask`); the embedding check failed on a harness import-path bug, since fixed. 202 s |
| `ITEST_API_PORT=18080 tests/integration/run.sh` | All checks passed, `RESULT: PASS`. 178 s (images cached); chat answer 81 s |
| Guards | Refused (exit 2) a project name outside `dc-itest-*`, an existing project, a port in use and a non-numeric port, before creating or removing anything |

After each run no containers, volumes, networks or built images of the run's project remained.
