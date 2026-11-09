# Drugcentral-RAG

Question answering over [DrugCentral](https://drugcentral.org) with cited answers. A FastAPI service retrieves
evidence from PostgreSQL/pgvector and generates answers that are checked against that evidence before they are
returned. It runs in two modes:

| Mode (`RAG_PIPELINE`) | Models | Needs | Use it for |
|---|---|---|---|
| `local` | Ollama on your machine (chat + embeddings) | Docker, Ollama, a locally built vector collection | Development without any cloud account or cost |
| `verified-bedrock` | Amazon Bedrock: Cohere Embed v4, Cohere Rerank 3.5, Qwen3 Next (generation + claim validation) | **Your own** AWS account with Bedrock model access, your own collections | The production pipeline: hybrid retrieval, rerank, graph answers, per-claim validation |

The two modes use separate, separately stamped vector collections; a collection built by one embedding model is
never served by, or overwritten from, the other.

## Three different things

1. **Hosted API: unavailable pending maintainer configuration.** The maintainers' deployment has no public
   endpoint: its API binds to `127.0.0.1` on the instance and is reached only by its operators through an SSH tunnel.
   There is no public Swagger URL. If one is published later, its protected routes will still require a bearer
   token that is not part of this repository.
2. **Run your own instance** on your machine, in either mode. It serves its own Swagger UI at
   **http://localhost:8000/docs**, with authentication you control. Start with
   [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md).
3. **Deploy to AWS** the way the maintainers do (one EC2 instance, Docker Compose, Bedrock):
   [docs/AWS_DEPLOYMENT.md](docs/AWS_DEPLOYMENT.md) and the runbook [environments/ec2/README.md](environments/ec2/README.md).

Cloning this repository gives you no credentials, tokens, database access or access to the maintainers' AWS
account. Local mode makes no AWS calls; Bedrock mode uses only the credentials of your own environment. The CLI and
evaluation tools default to `http://127.0.0.1:8000`; no setting defaults to a hosted API.

## Prerequisites

Clone the repository and run these commands from its root.

1. [Install Conda](https://www.anaconda.com/docs/getting-started/miniconda/install).
2. Create the environment:
   - Local models: `conda env create -f environments/local/environment.yml`
   - AWS Bedrock: `conda env create -f environments/aws/environment.yml`
3. Activate the environment:
   - Local models: `conda activate drugcentral-rag-local`
   - AWS Bedrock: `conda activate drugcentral-rag-aws`
4. Follow the documented model, database, and configuration setup for your selected mode.

Each environment file installs Python 3.11 from conda-forge and then the pinned packages of its lock file
(`requirements-local.lock` or `requirements-aws.lock`, next to the environment file). Docker workflows install the same
lock files without Conda. Validation and limitations: [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md#prerequisites).

## Quick start (your own instance)

With the environment active:

```bash
cp environments/local/ollama.env.example .env                 # or bedrock.env.example; edit the passwords
docker compose -f environments/local/docker-compose.yml --env-file .env up -d db
```

Then pull the models (local mode), build your collection, start the API and open http://localhost:8000/docs. Every
step, with download sizes, ingestion times and disk space, is in [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md).
The endpoints and authentication are described in [docs/API.md](docs/API.md).

## Asking an API from the command line

`cli/ask.py` needs only the Python 3 standard library. It reads the token from `API_AUTH_TOKEN` (or the variable
named by `--token-env`), never prints it and never takes it as an argument.

```bash
python -m cli.ask --check                                          # your instance: health and readiness
python -m cli.ask "What are the common side effects of metformin?"
python -m cli.ask                                                  # interactive; follow-ups share one session
read -rs API_AUTH_TOKEN && export API_AUTH_TOKEN                   # only if your instance requires a token
python -m cli.ask --url https://api.example.org --check            # any other instance you are authorised to use
```

The default URL is http://127.0.0.1:8000 (or `DRUGCENTRAL_API_URL`). Exit status: 0 success, 1 request failed,
2 usage error.

## Models

| Setting | Local mode default | Bedrock mode default | Role |
|---|---|---|---|
| `OLLAMA_EMBED_MODEL` / `BEDROCK_EMBED_MODEL` (+ `BEDROCK_EMBED_DIMENSIONS`) | `nomic-embed-text` (768) | `us.cohere.embed-v4:0` (1024) | Query and document embeddings |
| `BEDROCK_RERANK_MODEL` | — | `cohere.rerank-v3-5:0` (`us-west-2`) | Rerank retrieved evidence |
| `OLLAMA_CHAT_MODEL` / `BEDROCK_GENERATION_MODEL` | `qwen3.5:9b` | `qwen.qwen3-next-80b-a3b` | Answer generation |
| `BEDROCK_VALIDATION_MODEL` | — (local mode checks answers with the same chat model) | same as generation | Claim validation |
| `python -m evaluation.judge_verified --model …` | — | `us.amazon.nova-lite-v1:0` | Benchmark judge (evaluation only) |

* **Embeddings are tied to the collection.** Each collection is stamped with its backend, model and dimensions
  (Bedrock: `bedrock:<model>:dim=<n>:document=search_document:query=search_query:float:truncate=NONE`; Ollama: the
  model name). The API refuses to start on a mismatching collection, and ingestion refuses to overwrite a collection
  stamped by another embedder. Changing the embedding model means building a new collection.
* Reranking, generation and validation models can change without re-embedding: the reranker must accept the Cohere
  Rerank request format; Bedrock generation/validation models must support the Converse API and return the JSON the
  prompts ask for; Ollama chat models must support structured (JSON schema) output. Re-run the benchmark after changes.
* Accepted values: `RAG_PIPELINE` = `local` or `verified-bedrock` (required); `CHAT_BACKEND`/`EMBED_BACKEND` default from
  it and must match it; `VECTOR_BACKEND` = `pgvector`. Anything else stops the API at startup with an explicit error.

## Repository layout

| Path | Contents |
|---|---|
| `api.py` | FastAPI service (`/api/chat`, `/api/health`, `/api/ready`, `/api/metrics`, Swagger at `/docs`) |
| `agent.py`, `chatbot/` | Startup checks and both answer pipelines (retrieval, routing, generation, validation) |
| `cli/ask.py` | Developer CLI for any running API (standard library only) |
| `ingestion/` | Builders for vector/evidence collections, Neo4j graph, label and relation SQLite files |
| `evaluation/` | Benchmark builders, runners and the Nova Lite judge (datasets not included; `evaluation/DATASETS.md`) |
| `environments/local/` | Developer Compose file (pgvector, optional Ollama and source DB), env examples, local-mode Conda environment and dependency lock |
| `environments/aws/` | Bedrock-mode Conda environment, dependency spec and lock (the lock is also used by the production image) |
| `environments/prod/` | Production API `Dockerfile`, non-secret Bedrock settings, alert script |
| `environments/ec2/` | Production Compose file, runbook, restore/bundle/collection scripts, IAM policy, examples |
| `docs/` | Local development, local integration test record, AWS deployment, API reference, architecture, full-coverage runbook, source map |
| `tests/` | Auth, CLI, mode-selection and collection-guard tests (no models, AWS or databases needed); `tests/integration/`: opt-in local-mode smoke test in Docker |

## Tests

```bash
python -m unittest discover -s tests -t .     # in each Conda environment; no network, AWS or database needed
```

Run it once per dependency group: tests that need only the other group's packages are skipped. The Bedrock request
tests are mocked: they check request formats against stub clients and make no Bedrock calls.
`tests/integration/run.sh` is a separate, opt-in smoke test of local mode on synthetic data
([tests/integration/README.md](tests/integration/README.md)). What the unit tests, the small local integration test and the Bedrock benchmark each do and do not establish is
listed in [docs/LOCAL_DEVELOPMENT.md](docs/LOCAL_DEVELOPMENT.md#what-has-been-tested).

## Evaluation

Question sets and results are not committed; see `evaluation/DATASETS.md` for hashes, regeneration and the reported
benchmark. `evaluation.verified_benchmark` + `evaluation.judge_verified` reproduce it in Bedrock mode;
`evaluation.evaluate_live_api` collects answers from any running instance started with `ENABLE_EVALUATION_API=true`.

## Not included

Secrets (`.env`), DrugCentral-derived data (SQLite files, dumps, restore bundles, question sets), benchmark results,
logs and Docker image archives.

## License

No license has been chosen for this code yet. DrugCentral data is subject to DrugCentral's own terms; obtain it
from the DrugCentral project.
