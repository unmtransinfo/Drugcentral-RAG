#!/usr/bin/env bash
# Run ON the host from the repository root after `up -d db`, when you build the collection ON the server instead of
# restoring a bundle. Creates the separate ingestion (owner) and read-only API roles and an EMPTY drugcentral_rag database.
# It refuses to touch an existing database; restore.sh is the other way to fill it.
set -euo pipefail
COMPOSE=(docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env)
set -a; . environments/ec2/.env; set +a
: "${RAG_INGEST_PASSWORD:?}" "${RAG_API_PASSWORD:?}"
sql() { "${COMPOSE[@]}" exec -T db psql -v ON_ERROR_STOP=1 -U postgres "$@"; }
if sql -Atc "SELECT 1 FROM pg_database WHERE datname='drugcentral_rag'" | grep -q 1; then
  echo "drugcentral_rag already exists; refusing to change it" >&2; exit 1
fi
sql -c "CREATE ROLE rag_ingest LOGIN PASSWORD '$RAG_INGEST_PASSWORD'" -c "CREATE ROLE rag_api LOGIN PASSWORD '$RAG_API_PASSWORD'" \
    -c "CREATE DATABASE drugcentral_rag OWNER rag_ingest"
sql -d drugcentral_rag -c "CREATE EXTENSION vector" -c "REVOKE CREATE ON SCHEMA public FROM PUBLIC" \
    -c "GRANT USAGE ON SCHEMA public TO rag_api, rag_ingest" -c "GRANT CREATE ON SCHEMA public TO rag_ingest"
echo "Empty drugcentral_rag created with roles rag_ingest (writes) and rag_api (read-only)."
