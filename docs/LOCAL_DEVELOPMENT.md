# Running your own instance

This guide runs the FastAPI service on your machine in one of two modes. Both serve Swagger UI at
**http://localhost:8000/docs**. Nothing here connects to the maintainers' deployment, database or AWS account.

| | Local mode (`RAG_PIPELINE=local`) | Bedrock mode (`RAG_PIPELINE=verified-bedrock`) |
|---|---|---|
| Models | Ollama: `qwen2.5:14b` (chat), `nomic-embed-text` (embeddings) | Your AWS account: Cohere Embed v4, Cohere Rerank 3.5, Qwen3 Next |
| AWS | None: no credentials, no AWS calls, no AWS packages installed | Standard AWS credential chain of your shell; Bedrock usage is billed to you |
| Conda environment | `drugcentral-rag-local` from `environments/local/environment.yml` (packages: `requirements-local.lock`) | `drugcentral-rag-aws` from `environments/aws/environment.yml` (packages: `requirements-aws.lock`) |
| Example config | `environments/local/ollama.env.example` | `environments/local/bedrock.env.example` |
| Vector collection | `drugcentral_local_nomic_embed_text` (Ollama vectors) | `drugcentral_dev_cohere_v4_1024` + structured snapshot (Cohere vectors) |
| Answer pipeline | Vector retrieval + Ollama generation with JSON-schema output and an entailment check | Graph (optional) → hybrid retrieval → rerank → per-component generation → claim validation |

## Prerequisites

Clone the repository (`git clone <repository URL> drugcentral-rag && cd drugcentral-rag`) and run these commands from
its root.

1. [Install Conda](https://www.anaconda.com/docs/getting-started/miniconda/install).
2. Create the environment:
   - Local models: `conda env create -f environments/local/environment.yml`
   - AWS Bedrock: `conda env create -f environments/aws/environment.yml`
3. Activate the environment:
   - Local models: `conda activate drugcentral-rag-local`
   - AWS Bedrock: `conda activate drugcentral-rag-aws`
4. Follow the documented model, database, and configuration setup for your selected mode.

Each environment file installs Python 3.11 and pip from conda-forge (`nodefaults`, so no Anaconda `defaults` packages
or terms acceptance are involved) and then `pip install -r` of the lock file in its own directory; versions are pinned
only in the lock files, which the Docker images also use. Conda resolves the `-r` path relative to the environment
file and writes a temporary requirements file next to it, so create environments from a writable clone; creating
from a read-only checkout fails. Validated on Linux arm64 with Conda 26.9.4 (Miniforge) and 26.7.1 (Miniconda):
Python 3.11.18, installed versions identical to the lock files, unit tests passing. Not validated on amd64, macOS or
Windows.

Also required:

* Docker with Compose v2, about 25 GB free disk for local mode (table below).
* Local mode: [Ollama](https://ollama.com) installed natively, or the `ollama` Compose profile (CPU only as configured).
* Bedrock mode: an AWS account where you enabled Bedrock model access for the models above (Cohere Embed v4 and
  Qwen3 Next in `us-east-2`, Cohere Rerank 3.5 in `us-west-2`), and credentials available to your shell
  (`aws configure sso` / `aws configure`, then `export AWS_PROFILE=...`). The IAM permissions needed are
  `bedrock:InvokeModel` and `bedrock:Rerank` (see `environments/ec2/bedrock-policy.json`; tighten `Resource` to your
  model ARNs).

| Item | Size |
|---|---|
| `qwen2.5:14b` / `qwen2.5:7b` (alternative) | ~9.0 GB / ~4.7 GB |
| `nomic-embed-text` | ~0.3 GB |
| DrugCentral source database (Compose profile `source`) | ~5.5 GB of data plus the image |
| Local vector collection (~151,500 chunks × 768 dimensions + text) | ~1–2 GB |
| Conda environment (Python 3.11 + lock-file packages) | ~0.45 GB |

## 1. Configure

With the environment active, from the repository root:

```bash
cp environments/local/ollama.env.example .env                  # or bedrock.env.example
```

Edit `.env`: replace the `change-me-…` passwords (they only protect your local containers). `.env` is git-ignored.

## 2. Start the local database (both modes)

```bash
docker compose -f environments/local/docker-compose.yml --env-file .env up -d db
```

pgvector listens on `127.0.0.1:55432` (data in the `localpgdata` volume). The `vector` extension is created on
first start; ingestion creates the tables.

## 3. Models

Local mode, native Ollama:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:14b          # or qwen2.5:7b, then set OLLAMA_CHAT_MODEL=qwen2.5:7b in .env
```

Or with Docker: `docker compose -f environments/local/docker-compose.yml --env-file .env --profile ollama up -d ollama`
and `docker compose -f environments/local/docker-compose.yml --env-file .env exec ollama ollama pull nomic-embed-text`
(likewise for the chat model). The Compose service is pinned to the Ollama image available when this guide was
written, `ollama/ollama@sha256:ebaef4f4c5d93660e1f25f8ad085a112f51ccfb0e8458073e71793959dba91c9`, which reports
version **0.40.14**. The digest is a multi-platform index for `linux/amd64` and `linux/arm64` (Linux containers
only; on macOS and Windows it runs inside Docker Desktop's Linux VM, without GPU acceleration); it has been
tested on arm64 only ([LOCAL_INTEGRATION_TEST.md](LOCAL_INTEGRATION_TEST.md)). It listens on `127.0.0.1:11434`
only. A native Ollama install may be a different version. Bedrock mode needs no downloads.

## 4. Build your collection

DrugCentral data is not distributed with this repository. Start the public DrugCentral source database (amd64
image; first start initialises it) and point `SOURCE_DATABASE_URL` in `.env` at it:

```bash
docker compose -f environments/local/docker-compose.yml --env-file .env --profile source up -d source-db
```

On ARM machines without amd64 emulation, build `environments/ec2/source-db.Dockerfile` instead and restore a
DrugCentral PostgreSQL dump into it (`environments/ec2/restore_source.sh` shows the commands), then set
`SOURCE_DB_IMAGE`/`SOURCE_DB_PLATFORM` in `.env`.

**Local mode** — embed all chunks with Ollama into the local collection (no AWS):

```bash
python -m ingestion.vectors        # reads SOURCE_DATABASE_URL, writes VECTOR_COLLECTION
```

Embedding speed decides the time: `nomic-embed-text` on a 4-vCPU ARM CPU (no GPU) embedded about 0.9 chunks per
second in testing, so the full ~151,500 chunks take roughly two days on such a machine; a GPU is much faster. To check
the data first without embedding anything: `INGEST_PREVIEW_ONLY=1 python -m ingestion.vectors`.

**Bedrock mode** — follow [FULL_COVERAGE.md](FULL_COVERAGE.md): build the previews (no Bedrock calls), publish the
structured snapshot as `STRUCTURED_COLLECTION`, then embed it with `ingestion.cohere_migration` into
`VECTOR_COLLECTION` (about USD 5.50 of Cohere Embed usage on your account at the time of writing).

Optional for both modes: `data/labels.sqlite` (product labels; `python -m ingestion.labels`) and, for local mode,
`data/relations.sqlite` (exact drug–target lookups; `python -m ingestion.relations`). Both read a DrugCentral SQL dump;
see `data/README.md`.

A collection is stamped with the embedder that built it. Pointing local mode at a Bedrock collection (or the
reverse) stops the API with an identity-mismatch error, and ingestion refuses to overwrite a collection stamped by a
different embedder: use a separate `VECTOR_COLLECTION` per embedding model.

## 5. Run the API

```bash
uvicorn api:app --host 127.0.0.1 --port 8000
```

* Swagger UI: **http://localhost:8000/docs** (ReDoc: `/redoc`, schema: `/openapi.json`).
* `curl -s http://localhost:8000/api/ready` reports `"provider": "ollama"` or `"bedrock"` once the collection is
  loaded and the database (and, in local mode, the Ollama chat model) is available.
* Authentication: with `API_AUTH_TOKEN` empty, `/api/chat` is open on your machine. Set it in `.env` (for example
  `openssl rand -base64 32`) and restart to require `Authorization: Bearer <token>` on `/api/chat`, `/api/metrics` and
  the evaluation routes; in Swagger UI use **Authorize**. Keep the default `127.0.0.1` binding unless you mean to
  expose the API, and never expose it without a token.
* Local development binds to localhost only: the command above, and every port in
  `environments/local/docker-compose.yml`, use `127.0.0.1`. `cli.ask` and the evaluation tools default to
  `http://127.0.0.1:8000`; nothing in this setup defaults to a hosted API or to the maintainers' credentials.

## 6. Ask questions

```bash
python -m cli.ask --check
python -m cli.ask "What are the common side effects of metformin?"
export API_AUTH_TOKEN=...          # same value as in .env, only when you enabled authentication
python -m cli.ask                  # interactive
```

or `curl -s localhost:8000/api/chat -H 'Content-Type: application/json' -d '{"question":"..."}'` (add
`-H "Authorization: Bearer $API_AUTH_TOKEN"` when authentication is on).

## 7. Tests

```bash
python -m unittest discover -s tests -t .
```

They need no models, AWS access or database (Bedrock clients and databases are replaced by stubs). Run them in
both Conda environments, `drugcentral-rag-local` and `drugcentral-rag-aws`: in the local one the 4 Bedrock request
tests are skipped, and in the AWS one the local client test is skipped, so both runs are needed for full coverage.

The opt-in local-mode smoke test runs the whole local stack in Docker on synthetic data (about 3–4 minutes, about
8 GB of downloads on the first run) and is not part of the command above:

```bash
tests/integration/run.sh                        # chat model qwen2.5:3b, API on 127.0.0.1:18000
ITEST_CHAT_MODEL=qwen2.5:14b ITEST_OLLAMA_MEMORY=14g ITEST_API_PORT=18080 tests/integration/run.sh
```

Its default chat model, `qwen2.5:3b`, is a smoke-test setting; the application default remains `qwen2.5:14b`.
Prerequisites, settings and untested platforms: [tests/integration/README.md](../tests/integration/README.md).

## What has been tested

| Check | What it establishes | What it does not establish |
|---|---|---|
| Unit tests (`tests/`) | Mode selection, collection identity guards, bearer authentication, CLI behaviour. The Bedrock request tests are **mocked**: they check the request bodies sent to stub embedding, rerank and Converse clients and make no Bedrock calls | That your AWS account, model access, regions or credentials work |
| Small local integration test (`tests/integration/run.sh`, [record](LOCAL_INTEGRATION_TEST.md); pinned Ollama 0.40.14 image on arm64, `nomic-embed-text`, `qwen2.5:3b`, synthetic 3-drug data, no AWS) | Embedding, ingestion and collection stamping, readiness, `/docs`, bearer authentication and grounded chat answers work together in local mode | **Not full-data ingestion**: the ~151,500 DrugCentral chunks were not embedded with Ollama (about two days on CPU, see step 4). **Not a quality evaluation**: local-mode answer quality has not been measured. The default chat model `qwen2.5:14b` and amd64 were not run |
| Verified benchmark (`evaluation/DATASETS.md`) | Answer quality of Bedrock mode on the full-coverage collections, for the frozen question set | Anything about local mode; clinical safety; behaviour under load |

## Troubleshooting

| Symptom | Cause |
|---|---|
| `Set RAG_PIPELINE to 'local' … or 'verified-bedrock'` | `.env` not in the repository root, or `RAG_PIPELINE` missing |
| `CHAT_BACKEND='bedrock' conflicts with RAG_PIPELINE='local'` | Mixed configuration; remove the explicit backend or change the mode |
| `Embedding collection identity mismatch` / `holds … vectors from …` | `VECTOR_COLLECTION` was built by another embedding model |
| `/api/ready` 503 `Ollama is not ready` | Ollama not running at `OLLAMA_HOST`, or `OLLAMA_CHAT_MODEL` not pulled |
| `/api/ready` 503 `Storage is not ready` | Local database not running, or `PG*` settings wrong |
| `NoCredentialsError` / `AccessDeniedException` (Bedrock mode) | No AWS credentials in your shell, or model access not enabled for that account and region |
