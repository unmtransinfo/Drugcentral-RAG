# AWS deployment architecture

How the maintainers run the service, and how to reproduce it **in your own AWS account**. Account identifiers,
addresses and credentials of the maintainers' deployment are not part of this repository.

## Architecture

```
  Developer / operator laptop
        │  SSH (port 22, your IP only) + tunnel  ──►  127.0.0.1:8000
        ▼
┌──────────────────────── EC2 instance (Graviton, t4g.xlarge, 100 GB gp3) ────────────────────────┐
│  Docker Compose (environments/ec2/docker-compose.yml)                                              │
│                                                                                                    │
│   api  (environments/prod/Dockerfile, target bedrock)  ── 127.0.0.1:8000, bearer-token auth        │
│     │  RAG_PIPELINE=verified-bedrock; labels.sqlite baked into the image                           │
│     ├──► db     pgvector/pgvector:pg17   evidence + Cohere vectors (volume pgdata)                  │
│     │           roles: rag_api (read-only, used by the API), rag_ingest (writes, tools only)       │
│     ├──► neo4j  neo4j:5-community        relationship graph, optional (profile graph)              │
│     └──► logs   evidence audit trail (volume auditlogs)                                            │
│   source-db (profile source) and tools (profile tools): stopped; used only for rebuilds            │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │  instance role (bedrock:InvokeModel, bedrock:Rerank), IMDSv2, hop limit 2
        ▼
  Amazon Bedrock: Cohere Embed v4 + Qwen3 Next (us-east-2), Cohere Rerank 3.5 (us-west-2)
```

* **One instance, no managed services** besides Bedrock: no load balancer, RDS, ECS or Secrets Manager.
* **Request path:** the API answers from the graph when a question matches one of its fixed templates; otherwise it
  embeds the question (Cohere), runs hybrid dense + lexical search in pgvector, reranks (Cohere Rerank), generates per
  question component (Qwen3 Next via the Converse API) and validates each claim against the cited evidence before
  answering. Every answer writes an evidence audit record.
* **Network exposure:** the security group admits SSH from the operators' addresses only. The API, database and
  Neo4j ports bind to `127.0.0.1`; operators reach Swagger UI through an SSH tunnel. Protected routes require the
  bearer token configured in `environments/ec2/.env` on the instance (never committed).
* **Credentials:** containers obtain short-lived credentials from the EC2 instance role through IMDSv2 (hop limit 2 so
  containers can reach it). The role allows only Bedrock invocation (`environments/ec2/bedrock-policy.json`).
* **Data:** the RAG database is restored from a bundle (`rag.dump`, `graph.dump`, `labels.sqlite`, `SHA256SUMS`) or
  rebuilt from the DrugCentral source database (`docs/FULL_COVERAGE.md`). Backups are EBS snapshots plus offline
  dumps.

## Hosted API

**Unavailable pending maintainer configuration.** The maintainers' deployment exposes no public endpoint and no
public Swagger URL: the API binds to `127.0.0.1` and operators open http://localhost:8000/docs through the SSH tunnel
above. Calling `/api/chat`, `/api/metrics` or the evaluation routes also requires the deployment's bearer token,
which this repository does not contain.

## Deploying your own copy

Follow [environments/ec2/README.md](../environments/ec2/README.md) in your own AWS account:

1. Enable Bedrock model access for the models above in your account and regions.
2. Create an IAM role/instance profile from `environments/ec2/bedrock-policy.json` (tighten `Resource` to your model
   and inference-profile ARNs), launch an ARM64 Amazon Linux 2023 instance with IMDSv2 required and hop limit 2, and
   allow SSH only from your address.
3. Create `environments/ec2/.env` from `env.example` with your own generated passwords and API token.
   `API_AUTH_TOKEN` is required: the `api` service refuses to start while it is unset or empty, so the deployed API
   always requires `Authorization: Bearer <token>` on its protected routes.
4. Provide data: `data/labels.sqlite` (`data/README.md`) and either your own restore bundle or a rebuild from the
   source database. The collection names in `environments/prod/bedrock-production.env` are those of the reference
   deployment; set `VECTOR_COLLECTION`/`STRUCTURED_COLLECTION` (or `environments/ec2/collection.env`) to the
   collections you build.
5. `environments/ec2/restore.sh bundle`, then `docker compose … up -d --build api` and check `/api/ready`.

The optional `docker-compose.keys.yml` overlay passes IAM user keys from your `.env` instead of an instance role;
prefer the role.
