#!/usr/bin/env bash
# Run ON the host from the repository root. Starts the full DrugCentral SOURCE database (PostgreSQL 16 + RDKit 4.8.0)
# and restores the source archive into it. Needed only to (re)build vectors, the graph or the inventory; answering
# questions never touches it. On x86 hosts the published amd64 image restores itself and this script is not needed.
#
# ARM (Graviton) hosts: load the image built from source-db.Dockerfile first
#   (docker load < source-db-image.tar.gz), then set in environments/ec2/.env:
#   SOURCE_DB_IMAGE=drugcentral-source-db:rdkit-2026.03  SOURCE_DB_PLATFORM=linux/arm64  SOURCE_DB_PASSWORD=...
# Usage: environments/ec2/restore_source.sh /srv/drugcentral/Drugcentral-RAG-EC2/source-bundle/source.dump
set -euo pipefail
DUMP="${1:?path to source.dump (a pg_restore custom archive)}"
COMPOSE=(docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env --profile source)
set -a; . environments/ec2/.env; set +a
: "${SOURCE_DB_PASSWORD:?set SOURCE_DB_PASSWORD in environments/ec2/.env}"
DB="${SOURCE_DB_NAME:-drugcentral}"
[ -s "$DUMP" ] || { echo "$DUMP not found or empty" >&2; exit 1; }
[ -f "$(dirname "$DUMP")/SHA256SUMS" ] && ( cd "$(dirname "$DUMP")" && sha256sum -c SHA256SUMS --ignore-missing )
"${COMPOSE[@]}" up -d source-db
for i in $(seq 1 60); do "${COMPOSE[@]}" exec -T source-db pg_isready -U postgres >/dev/null 2>&1 && break; sleep 2; done
sql() { "${COMPOSE[@]}" exec -T source-db psql -v ON_ERROR_STOP=1 -U postgres "$@"; }
if sql -Atc "SELECT 1 FROM pg_database WHERE datname='$DB'" | grep -q 1; then
  echo "Database $DB already exists; refusing to overwrite. Drop it deliberately if you want a fresh restore." >&2; exit 1
fi
sql -c "CREATE DATABASE $DB"
sql -d "$DB" -c "CREATE EXTENSION rdkit"
echo "RDKit extension: $(sql -d "$DB" -Atc "SELECT extversion FROM pg_extension WHERE extname='rdkit'")"
echo "Restoring (about 5 minutes) ..."
"${COMPOSE[@]}" cp "$DUMP" source-db:/tmp/source.dump
"${COMPOSE[@]}" exec -T source-db sh -c "pg_restore -l /tmp/source.dump | grep -vE '^[0-9]+; [0-9]+ [0-9]+ (EXTENSION|COMMENT - EXTENSION)' > /tmp/source.toc \
  && pg_restore -U postgres -d $DB -L /tmp/source.toc --no-owner --no-privileges -j 2 /tmp/source.dump" \
  || { echo "pg_restore reported errors; do not use this database until they are explained" >&2; exit 1; }
"${COMPOSE[@]}" exec -T source-db rm -f /tmp/source.dump /tmp/source.toc
echo "Verifying ..."
sql -d "$DB" -Atc "SELECT 'tables', count(*) FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE'" \
               -c "SELECT 'views', count(*) FROM information_schema.views WHERE table_schema='public'" \
               -c "SELECT 'structures', count(*) FROM structures" -c "SELECT 'act_table_full', count(*) FROM act_table_full" \
               -c "SELECT 'structures with molecule', count(*) FROM structures WHERE m IS NOT NULL" \
               -c "SELECT 'dbversion', version, dtime FROM dbversion"
echo "Done. Expect 70 tables, 4 views, 5116 structures. Source database: $DB (service source-db)."
