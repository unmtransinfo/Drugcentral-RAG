#!/usr/bin/env bash
# Run ON the EC2 host from the repository root, after the bundle is copied and `db` is running.
# Creates the separate runtime/ingestion roles, restores the RAG database, restores the graph (if present)
# and verifies the result. Safe to re-run only on an empty database; it refuses to overwrite.
# Usage: environments/ec2/restore.sh bundle   (run from /srv/drugcentral/Drugcentral-RAG-EC2)
set -euo pipefail
BUNDLE="${1:?bundle directory}"
COMPOSE=(docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env)
set -a; . environments/ec2/.env; set +a
( cd "$BUNDLE" && sha256sum -c SHA256SUMS )
psql_db() { "${COMPOSE[@]}" exec -T db psql -v ON_ERROR_STOP=1 -U postgres "$@"; }

if psql_db -Atc "SELECT 1 FROM pg_database WHERE datname='drugcentral_rag'" | grep -q 1; then
  echo "drugcentral_rag already exists; refusing to overwrite" >&2; exit 1
fi
psql_db -c "CREATE ROLE rag_ingest LOGIN PASSWORD '${RAG_INGEST_PASSWORD:?}'" \
        -c "CREATE ROLE rag_api LOGIN PASSWORD '${RAG_API_PASSWORD:?}'" \
        -c "CREATE DATABASE drugcentral_rag OWNER rag_ingest"
psql_db -d drugcentral_rag -c "CREATE EXTENSION vector" -c "REVOKE CREATE ON SCHEMA public FROM PUBLIC" \
        -c "GRANT USAGE ON SCHEMA public TO rag_api, rag_ingest" -c "GRANT CREATE ON SCHEMA public TO rag_ingest"
echo "Restoring rag.dump (this takes a while) ..."
# Parallel restore needs a seekable file, so copy the archive into the container first.
"${COMPOSE[@]}" cp "$BUNDLE/rag.dump" db:/tmp/rag.dump
"${COMPOSE[@]}" exec -T db pg_restore -U postgres -d drugcentral_rag --no-owner --role=rag_ingest -j 2 /tmp/rag.dump \
  || { echo "pg_restore reported errors; check the output above before continuing" >&2; exit 1; }
"${COMPOSE[@]}" exec -T db rm -f /tmp/rag.dump
psql_db -d drugcentral_rag -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_api" -c "ANALYZE"

echo "Verifying ..."
psql_db -d drugcentral_rag -Atc "SELECT collection, count(*) FROM drugcentral_chunks GROUP BY 1 ORDER BY 1" \
        -c "SELECT collection, model FROM drugcentral_collections ORDER BY 1" \
        -c "SELECT source, status FROM drugcentral_migrations"
psql_db -d drugcentral_rag -Atc "SELECT has_table_privilege('rag_api','drugcentral_chunks','SELECT'), has_table_privilege('rag_api','drugcentral_chunks','INSERT')"

if [ -f "$BUNDLE/graph.dump" ]; then
  : "${NEO4J_PASSWORD:?set NEO4J_PASSWORD to restore the graph}"
  echo "Restoring graph ..."
  "${COMPOSE[@]}" --profile graph create neo4j >/dev/null          # creates the labelled data volume
  "${COMPOSE[@]}" --profile graph stop neo4j >/dev/null 2>&1 || true
  VOLUME="$("${COMPOSE[@]}" --profile graph config --format json | python3 -c "import json,sys;c=json.load(sys.stdin);print(c['name']+'_neo4jdata')")"
  mkdir -p /tmp/graph-load && cp "$BUNDLE/graph.dump" /tmp/graph-load/neo4j.dump
  docker run --rm -v "$VOLUME":/data -v /tmp/graph-load:/dump neo4j:5-community \
    neo4j-admin database load neo4j --from-path=/dump --overwrite-destination=true
  rm -rf /tmp/graph-load
  "${COMPOSE[@]}" --profile graph up -d neo4j
fi
echo "Done. Start the API: ${COMPOSE[*]} up -d --build api"
