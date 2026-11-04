# EC2 + Bedrock deployment runbook

One EC2 instance runs the API, PostgreSQL/pgvector and (optionally) Neo4j in Docker Compose. Amazon Bedrock is the only
other AWS service. No ALB, RDS, ECS, ECR, DynamoDB, Secrets Manager or Cognito.

Everything below was rehearsed on the development machine with the same compose file, scripts and bundle: restore from
bundle → API start → graph questions answered over HTTP with an audit record. Only the AWS-side steps (1–3) are new on
the host.

## What gets deployed

| Piece | Where it comes from |
|---|---|
| API image (`drugcentral-rag-api:local`) | built on the host from this repo (`environments/prod/Dockerfile`, target `bedrock`) |
| RAG database (evidence + vectors) | `rag.dump` in the bundle → `restore.sh` |
| Relationship graph (optional) | `graph.dump` in the bundle → `restore.sh` (no source database needed on the host) |
| Product-label database | `labels.sqlite` (baked into the image at build) |
| Models | Bedrock via the instance role (`bedrock-policy.json`) |

The full DrugCentral source database is **not** needed to answer questions. Keep it on the development machine (or start
the `source` profile only when rebuilding vectors or the graph).

## 0. On the development machine

```bash
# Active collections only (~1/3 of the full database). KEEP_COLLECTIONS=ALL ships everything.
NEO4J_CONTAINER=drugcentral-neo4j-test environments/ec2/bundle.sh ./bundle      # include the graph
environments/ec2/bundle.sh ./bundle                                              # or without it
```

If the graph container no longer exists, rebuild it first from the local source database
(`python -m ingestion.graph_build --rebuild`, about six seconds, see `docs/ARCHITECTURE_IMPLEMENTATION.md`).
The bundle ends with `SHA256SUMS`, verified again on the host.

## 1. AWS: account side (console or CLI, your credentials)

1. **Region** `us-east-2` (matches `environments/prod/bedrock-production.env`).
2. **Bedrock model access**: enable Cohere Embed v4 and the Qwen3 Next model in `us-east-2`, and Cohere Rerank 3.5 in
   `us-west-2` (the env file uses both). Run a one-off test from the console playground before blaming the instance.
3. **IAM role** for EC2 with the policy in `bedrock-policy.json`; create an instance profile from it. Tighten
   `Resource` to the model and inference-profile ARNs once everything works.
4. **Security group**: inbound **SSH (22) from your IP only**. No inbound 8000/7687/5432: the API binds to `127.0.0.1`
   and you reach it through an SSH tunnel.
5. **Key pair** for SSH.

### Without an IAM role (access keys instead)

A role is optional. Skip steps 1.3 and the instance profile and the IMDS hop-limit setting, and instead put an IAM user's
keys in `environments/ec2/.env` (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `chmod 600`), then add
`-f environments/ec2/docker-compose.keys.yml` after the main compose file in every `docker compose` command, e.g.
`C="docker compose -f environments/ec2/docker-compose.yml -f environments/ec2/docker-compose.keys.yml --env-file environments/ec2/.env"`.
The cost is that long-lived keys live on the instance disk. If you do this, use a user that can only call Bedrock
(`bedrock-policy.json`), never an administrator's keys, and rotate them if the instance is ever exposed.

## 2. Launch the instance

* AMI: Amazon Linux 2023 **ARM64**. Type: `t4g.large` (2 vCPU / 8 GB). Storage: **60 GB gp3** (database ≈ 2 GB trimmed,
  image ≈ 4.6 GB, Neo4j and logs small; keep headroom for restores).
* Attach the instance profile from step 1.
* **Metadata options: IMDSv2 required, hop limit = 2.** Containers sit one network hop behind the host, so with the
  default hop limit of 1 they cannot fetch the role credentials and every Bedrock call fails with `NoCredentialsError`
  (the API then refuses to start because its startup embedding probe cannot authenticate).
* Do not assign a public IP if you reach it through a bastion or SSM; otherwise a public IP is fine because the security
  group only allows your SSH.

Cost note: Graviton `t4g` avoids the amd64-only source image problem, since the runtime stack here is multi-arch.
Stopping the instance stops compute billing; the EBS volume keeps billing (about $0.08 per GB-month).

## 3. Prepare the host

```bash
sudo dnf install -y docker git rsync
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user && newgrp docker
# Compose plugin (ARM64)
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-aarch64 \
     -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
sudo mkdir -p /srv/drugcentral && sudo chown ec2-user /srv/drugcentral
```

## 4. Copy code and data (from the development machine)

```bash
HOST=ec2-user@EC2_PUBLIC_OR_PRIVATE_IP
rsync -avP --exclude '.git' --exclude 'data/' --exclude 'benchmarks/' --exclude 'logs/' --exclude '.env' \
      --exclude 'environments/ec2/.env' --exclude '__pycache__' ./ $HOST:/srv/drugcentral/Drugcentral-RAG-EC2/
rsync -avP ./bundle/ $HOST:/srv/drugcentral/Drugcentral-RAG-EC2/bundle/
```

The deployment runs from this single folder; the bundle lives inside it as `bundle/`. On the host, place the label
database where the image build expects it:

```bash
cd /srv/drugcentral/Drugcentral-RAG-EC2
mkdir -p data
cp bundle/labels.sqlite data/labels.sqlite
```

These are separate copies, not hard links: `data/labels.sqlite` and `bundle/labels.sqlite` each take disk space (about
2.6 GB). Only `data/labels.sqlite` is read by the image build; `bundle/` is the restore source.

## 5. Configure

```bash
cd /srv/drugcentral/Drugcentral-RAG-EC2
cp environments/ec2/env.example environments/ec2/.env && chmod 600 environments/ec2/.env
# Fill POSTGRES_ADMIN_PASSWORD, RAG_API_PASSWORD, RAG_INGEST_PASSWORD, API_AUTH_TOKEN (long random),
# NEO4J_PASSWORD (if using the graph). Leave GRAPH_ENABLED=0 until the graph is restored and checked.
```

Generate secrets with `openssl rand -base64 32`. The `.env` file never leaves the host. `API_AUTH_TOKEN` must be
nonempty: Compose refuses to start the `api` service without it.

## 6. Start the database and restore

```bash
C="docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env"
$C up -d db
environments/ec2/restore.sh bundle          # run from /srv/drugcentral/Drugcentral-RAG-EC2
```

`restore.sh` verifies checksums, creates the separate `rag_ingest` (owner) and `rag_api` (read-only) roles, restores the
database, grants `SELECT` only to `rag_api`, restores the graph if present, and prints the verification: collection
row counts, the embedding-model stamp, the migration status and the `rag_api` privileges (`t|f` = can read, cannot write).
It refuses to run over an existing `drugcentral_rag` database.

## 7. Start the API

```bash
$C up -d --build api                          # builds on the host, ~10 minutes the first time
for i in $(seq 1 40); do curl -fsS localhost:8000/api/health && break; sleep 5; done
curl -sS localhost:8000/api/ready -H "Authorization: Bearer $API_AUTH_TOKEN"
```

Expect `"status":"ready"` with `"drugs":5116`. A startup embedding probe makes a couple of tiny Bedrock calls; a failure
here means the role, hop limit or Bedrock access is wrong, not the data.

## 8. Enable the graph (optional)

```bash
$C --profile graph up -d neo4j
# edit environments/ec2/.env: GRAPH_ENABLED=1
$C --profile graph up -d api
curl -sS -X POST localhost:8000/api/chat -H "Authorization: Bearer $API_AUTH_TOKEN" -H 'Content-Type: application/json' \
     -d '{"question":"Which drugs target EGFR and are indicated for lung cancer?"}'
```

The graph route makes no Bedrock calls and answers in tens of milliseconds. If Neo4j is down the API falls back to
normal retrieval instead of failing.

## 9. Check a real RAG answer, then the baseline

```bash
curl -sS -X POST localhost:8000/api/chat -H "Authorization: Bearer $API_AUTH_TOKEN" -H 'Content-Type: application/json' \
     -d '{"question":"What are the targets and indications of metformin?"}'     # multi-part: one set of Bedrock calls per component
```

Baseline before changing anything (costs Bedrock; start small):

```bash
$C exec api python -m evaluation.run_complex_benchmark --questions evaluation/complex-110/questions.json --estimate
$C exec api python -m evaluation.run_complex_benchmark --questions evaluation/complex-110/questions.json \
     --output benchmarks/complex-first10.jsonl --limit 10
$C exec api python -m evaluation.complex_metrics --questions evaluation/complex-110/questions.json \
     --results benchmarks/complex-first10.jsonl
```

## 10. Reaching the API from your laptop

```bash
ssh -N -L 8000:127.0.0.1:8000 ec2-user@HOST       # then http://localhost:8000/api/chat
```

Swagger UI is then at http://localhost:8000/docs on your laptop; use **Authorize** with the token from `.env`.

Use Tailscale or a reverse proxy only if other people need access; never open 8000 to the internet without TLS and the
token.

## 11. Backups and cost control

* **EBS snapshots** (daily, keep 7) are the simplest backup: Data Lifecycle Manager policy on the volume.
* The audit log is in the `auditlogs` volume (`/app/logs/evidence-audit.jsonl` in the container).
* Idle: `aws ec2 stop-instances` (compute stops, EBS keeps billing). Bedrock bills per token only.
* Do not stop Docker volumes with `docker compose down -v`: `-v` deletes the database.

## 12. Rollback and teardown

* Roll back a bad release: re-deploy the previous code directory and `$C up -d --build api`; the data volumes are untouched.
* Recover the database: restore from an EBS snapshot, or drop `drugcentral_rag` and re-run `restore.sh`.
* Full teardown: terminate the instance, delete its volume and snapshots, delete the IAM role/profile and the security
  group. Nothing else exists.

## Troubleshooting

| Symptom | Cause |
|---|---|
| API restarts, log shows `NoCredentialsError` | instance profile missing, or IMDS hop limit is 1 (set it to 2) |
| `AccessDeniedException` on invoke | Bedrock model access not enabled in that region, or role `Resource` too narrow for the inference profile |
| every answer is “could not be verified” and the log shows `reason=AUDIT_ERROR` | the audit volume is not writable (the image creates `/app/logs` owned by `app`; use the compose volume, not a host bind mount) |
| `/api/ready` 503 | database not restored, or `rag_api` password mismatch with `.env` |
| graph questions fall through to normal retrieval | `GRAPH_ENABLED=0`, Neo4j not started, or wrong `NEO4J_PASSWORD` |
| out-of-memory on `t4g.medium` | use `t4g.large`, or leave Neo4j off |

## Optional: full source database on the host

Only for rebuilding vectors, the graph or the inventory. On an **x86** instance use the published image
(`SOURCE_DB_IMAGE=unmtransinfo/drugcen-2027-db:latest`, `SOURCE_DB_PLATFORM=linux/amd64`, set `SOURCE_DB_PASSWORD`,
`--profile source`). On Graviton build `source-db.Dockerfile` (PostgreSQL 16 + RDKit 4.8.0, built here from the same
recipe), then restore the archive (`pg_restore` of `data/source/drugcen-2027-image.dump`, shipped with
`bundle.sh … --with-source`). Jobs run through the `tools` profile, e.g.
`$C --profile graph --profile source --profile tools run --rm tools python -m ingestion.graph_build --rebuild`.
