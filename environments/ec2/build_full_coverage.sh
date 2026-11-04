#!/usr/bin/env bash
# Run ON the host from the repository root, after restore_source.sh. Builds the full-coverage collection:
#   1. base dossier chunks + new chunks (no Bedrock calls) and the cost estimate,
#   2. publishes the structured snapshot (local database only),
#   3. embeds into a NEW vector collection (spends Bedrock tokens: needs --apply and a typed confirmation),
#   4. creates the search indexes.
# Without --apply it stops after step 1 and writes nothing to the database.
# With --apply it also grants the read-only API role access, and builds the relationship graph when NEO4J_PASSWORD is set.
# Optional environment: REUSE_VECTORS=<complete vector collection> copies vectors for unchanged chunk text (pay only for changes);
#   SKIP_BASE_PREVIEW=1 keeps the existing base-preview.jsonl so unchanged base chunks stay byte-identical (needed for reuse to hit).
# Usage: environments/ec2/build_full_coverage.sh [--apply] [YYYYMMDD]
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && { APPLY=1; shift; }
TAG="${1:-$(date +%Y%m%d)}"
STRUCT="drugcentral-sections-full-$TAG"; VEC="drugcentral_full_cohere_v4_1024_$TAG"
# COMPOSE_OVERRIDE adds a compose file, e.g. environments/ec2/docker-compose.keys.yml when no instance role is attached.
COMPOSE=(docker compose -f environments/ec2/docker-compose.yml ${COMPOSE_OVERRIDE:+-f "$COMPOSE_OVERRIDE"} --env-file environments/ec2/.env --profile source --profile tools)
run() { "${COMPOSE[@]}" run --rm -T tools "$@"; }
mkdir -p benchmarks/full-coverage
DB="${SOURCE_DB_NAME:-drugcentral}"
VERSION="$("${COMPOSE[@]}" exec -T source-db psql -U postgres -d "$DB" -Atc "SELECT 'drugcentral-' || version || '-' || to_char(dtime,'YYYY-MM-DD') FROM dbversion ORDER BY dtime DESC LIMIT 1")"
echo "Dataset version: $VERSION   structured: $STRUCT   vectors: $VEC"
export DATASET_VERSION="$VERSION"
echo "== 1/4 previews (no Bedrock calls)"
if [ "${SKIP_BASE_PREVIEW:-0}" = "1" ] && [ -s benchmarks/full-coverage/base-preview.jsonl ]; then
  echo "Keeping the existing base preview (SKIP_BASE_PREVIEW=1)."
else
"${COMPOSE[@]}" run --rm -T -e DATASET_VERSION="$VERSION" -e INGEST_PREVIEW_ONLY=1 -e INGEST_PREVIEW_PATH=benchmarks/full-coverage/base-preview.jsonl tools python -m ingestion.vectors | tail -3
fi
"${COMPOSE[@]}" run --rm -T -e DATASET_VERSION="$VERSION" tools python -m ingestion.full_coverage preview \
    --output benchmarks/full-coverage/extra-preview.jsonl | tee benchmarks/full-coverage/cost.json | tail -12
if [ "$APPLY" != "1" ]; then
  echo "Stopped before any database write or Bedrock spend. Review the estimate above, then rerun with --apply $TAG"; exit 0
fi
echo "== 2/4 publish structured snapshot $STRUCT"
# A snapshot is published in one transaction, so if it exists it is complete: skip, which makes a rerun after an interruption resume embedding.
EXISTING="$("${COMPOSE[@]}" exec -T db psql -U postgres -d drugcentral_rag -Atc "SELECT count(*) FROM drugcentral_evidence WHERE collection='$STRUCT'" 2>/dev/null || echo 0)"
if [ "${EXISTING:-0}" -gt 0 ]; then
  echo "Snapshot $STRUCT already published ($EXISTING rows); skipping publish."
else
  run python -m ingestion.full_coverage publish --base benchmarks/full-coverage/base-preview.jsonl \
      --extra benchmarks/full-coverage/extra-preview.jsonl --collection "$STRUCT" | tail -3
fi
echo "== 3/4 embedding into $VEC spends Bedrock tokens (see the estimated_usd in the cost report above)."
read -r -p "Type YES to spend it: " answer
[ "$answer" = "YES" ] || { echo "Not confirmed; the snapshot is published but nothing was embedded."; exit 1; }
run python -m ingestion.cohere_migration --source "$STRUCT" --target "$VEC" --report "benchmarks/full-coverage/migration-$TAG.json" ${REUSE_VECTORS:+--reuse "$REUSE_VECTORS"} --apply | tail -3
echo "== 4/6 indexes"
run python -m ingestion.index_collection --collection "$VEC"
echo "== 5/6 read-only access for the API role"
"${COMPOSE[@]}" exec -T db psql -v ON_ERROR_STOP=1 -U postgres -d drugcentral_rag -c "GRANT SELECT ON ALL TABLES IN SCHEMA public TO rag_api" -Atc \
    "SELECT has_table_privilege('rag_api','drugcentral_chunks','SELECT'), has_table_privilege('rag_api','drugcentral_chunks','INSERT')"
set -a; . environments/ec2/.env; set +a
if [ -n "${NEO4J_PASSWORD:-}" ]; then
  echo "== 6/6 relationship graph from the same source database"
  "${COMPOSE[@]}" --profile graph up -d neo4j
  for i in $(seq 1 40); do "${COMPOSE[@]}" --profile graph exec -T neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" 'RETURN 1' >/dev/null 2>&1 && break; sleep 3; done
  run python -m ingestion.graph_build --rebuild | tail -25
else
  echo "== 6/6 skipped: NEO4J_PASSWORD is not set"
fi
echo "Done. Evaluate $VEC before switching the API to it (docs/FULL_COVERAGE.md step 4); export_collection.sh is NOT needed on this host."
echo "Stop the source database when finished: ${COMPOSE[*]} stop source-db"
# Write the switch file with the REAL names so nobody has to type them (a placeholder like YYYYMMDD in this file crashes the API).
printf 'VECTOR_COLLECTION=%s\nSTRUCTURED_COLLECTION=%s\n' "$VEC" "$STRUCT" > environments/ec2/collection.env.new
echo "To switch: mv environments/ec2/collection.env.new environments/ec2/collection.env && unset VECTOR_COLLECTION STRUCTURED_COLLECTION && recreate the api container."
