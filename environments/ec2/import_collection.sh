#!/usr/bin/env bash
# Run ON the EC2 host from the repository root. Adds an exported collection to the running database without
# touching the existing one, builds its search indexes, and prints how to switch to it. Refuses to overwrite.
# Usage: environments/ec2/import_collection.sh /srv/drugcentral/Drugcentral-RAG-EC2/collection-bundle
set -euo pipefail
DIR="${1:?collection directory}"
# DB_EXEC overrides how to reach the database (used for local rehearsals): e.g. DB_EXEC="docker exec -i drugcentral-pgvector"
RAG_DB="${RAG_DB:-drugcentral_rag}"
if [ -n "${DB_EXEC:-}" ]; then read -r -a COMPOSE <<< "$DB_EXEC"; DB_EXEC_ARGS=(); else
  COMPOSE=(docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env exec -T db)
fi
( cd "$DIR" && sha256sum -c SHA256SUMS )
set -a; . "$DIR/collection.env"; set +a
db() { "${COMPOSE[@]}" psql -v ON_ERROR_STOP=1 -U postgres -d "$RAG_DB" "$@"; }
[ "$(db -Atc "SELECT count(*) FROM drugcentral_chunks WHERE collection='$VECTOR_COLLECTION'")" = "0" ] \
  || { echo "$VECTOR_COLLECTION already exists; refusing to overwrite" >&2; exit 1; }
for table in drugcentral_evidence drugcentral_chunks drugcentral_collections drugcentral_migrations; do
  echo "Importing $table ..."
  gunzip -c "$DIR/$table.csv.gz" | "${COMPOSE[@]}" psql -v ON_ERROR_STOP=1 -U postgres -d "$RAG_DB" \
      -c "\\copy public.$table FROM STDIN WITH (FORMAT csv)"
done
TAG="$(printf '%s' "$VECTOR_COLLECTION" | sha256sum | cut -c1-12)"
db -c "CREATE INDEX IF NOT EXISTS dc_scope_$TAG ON public.drugcentral_chunks (collection, (metadata->>'drug_name'), (metadata->>'section')) WHERE collection = '$VECTOR_COLLECTION'" \
   -c "CREATE INDEX IF NOT EXISTS dc_section_$TAG ON public.drugcentral_chunks (collection, (metadata->>'section')) WHERE collection = '$VECTOR_COLLECTION'" \
   -c "CREATE INDEX IF NOT EXISTS dc_lexical_$TAG ON public.drugcentral_chunks USING gin (to_tsvector('simple', document)) WHERE collection = '$VECTOR_COLLECTION'" \
   -c "ANALYZE public.drugcentral_chunks"
db -Atc "SELECT collection, count(*) FROM drugcentral_chunks GROUP BY 1 ORDER BY 1" \
   -c "SELECT collection, status FROM drugcentral_migrations ORDER BY 1" \
   -c "SELECT has_table_privilege('rag_api','drugcentral_chunks','SELECT')"
cp "$DIR/collection.env" environments/ec2/collection.env.new
echo "Imported. To switch the API to it: mv environments/ec2/collection.env.new environments/ec2/collection.env && recreate the api container."
echo "To go back: delete environments/ec2/collection.env and recreate the api container. The old collection is untouched."
