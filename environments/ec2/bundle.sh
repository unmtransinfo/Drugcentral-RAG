#!/usr/bin/env bash
# Run on the SOURCE machine (this one). Produces the transfer bundle for the EC2 host:
#   rag.dump          pg_dump -Fc of the pgvector RAG database (evidence + vectors)
#   graph.dump        offline neo4j-admin dump of the derived relationship graph (optional)
#   labels.sqlite     product-label database
#   SHA256SUMS        checksums, verified again on the host
# Usage: environments/ec2/bundle.sh [OUTPUT_DIR] [--with-source]
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT="${1:-.ec2-bundle}"; WITH_SOURCE="${2:-}"
RAG_CONTAINER="${RAG_CONTAINER:-drugcentral-pgvector}"
RAG_DB="${RAG_DB:-drugcentral_rag}"
NEO4J_CONTAINER="${NEO4J_CONTAINER:-}"           # set to include the graph; it is stopped for the offline dump
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5-community}"
[ -e "$OUT" ] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"       # absolute, so bind mounts below are correct for relative and absolute inputs
# Only the active collections ship by default (legacy trial collections are ~3x the size). Set KEEP_COLLECTIONS=ALL for everything.
if [ -z "${KEEP_COLLECTIONS+x}" ]; then
  KEEP_COLLECTIONS="$(grep -E '^(VECTOR|STRUCTURED)_COLLECTION=' environments/prod/bedrock-production.env | cut -d= -f2 | paste -sd,)"
fi
SRC_DB="$RAG_DB"
if [ "$KEEP_COLLECTIONS" != "ALL" ]; then
  TMP_DB="${RAG_DB}_bundle_tmp"
  echo "Building a trimmed copy keeping: $KEEP_COLLECTIONS"
  LIST="$(printf "'%s'," ${KEEP_COLLECTIONS//,/ })"; LIST="${LIST%,}"
  docker exec "$RAG_CONTAINER" psql -U postgres -qc "DROP DATABASE IF EXISTS $TMP_DB" -c "CREATE DATABASE $TMP_DB"
  docker exec "$RAG_CONTAINER" sh -c "pg_dump -U postgres -Fc --no-owner $RAG_DB | pg_restore -U postgres -d $TMP_DB --no-owner" || true
  for TABLE in drugcentral_chunks drugcentral_evidence drugcentral_collections drugcentral_migrations; do
    docker exec "$RAG_CONTAINER" psql -U postgres -d "$TMP_DB" -qc "DELETE FROM public.$TABLE WHERE collection NOT IN ($LIST)"
  done
  docker exec "$RAG_CONTAINER" psql -U postgres -d "$TMP_DB" -Atc "SELECT collection, count(*) FROM drugcentral_chunks GROUP BY 1"
  SRC_DB="$TMP_DB"
fi
echo "Dumping $SRC_DB from $RAG_CONTAINER ..."
docker exec "$RAG_CONTAINER" pg_dump -U postgres -Fc --no-owner --exclude-extension=vector "$SRC_DB" > "$OUT/rag.dump"
[ "$SRC_DB" != "$RAG_DB" ] && docker exec "$RAG_CONTAINER" psql -U postgres -qc "DROP DATABASE $SRC_DB"
cp data/labels.sqlite "$OUT/labels.sqlite"
if [ -n "$NEO4J_CONTAINER" ]; then
  echo "Stopping $NEO4J_CONTAINER for an offline graph dump ..."
  VOLUME="$(docker inspect "$NEO4J_CONTAINER" --format '{{range .Mounts}}{{if eq .Destination "/data"}}{{.Name}}{{end}}{{end}}')"
  docker stop "$NEO4J_CONTAINER" >/dev/null
  trap 'docker start "$NEO4J_CONTAINER" >/dev/null 2>&1 || true' EXIT   # never leave the graph stopped
  # ":z" relabels the bind mount for SELinux hosts; chown hands the dump back to the invoking user.
  docker run --rm -v "$VOLUME":/data -v "$OUT":/dump:z --entrypoint sh "$NEO4J_IMAGE" -c \
    "neo4j-admin database dump neo4j --to-path=/dump && chown $(id -u):$(id -g) /dump/neo4j.dump"
  mv "$OUT/neo4j.dump" "$OUT/graph.dump"
fi
if [ "$WITH_SOURCE" = "--with-source" ]; then cp data/source/drugcen-2027-image.dump "$OUT/source.dump"; fi
(cd "$OUT" && sha256sum * > SHA256SUMS)
du -sh "$OUT"/*
echo "Bundle ready in $OUT. Copy it to the host, e.g.: rsync -avP $OUT/ ec2-user@HOST:/srv/drugcentral/Drugcentral-RAG-EC2/bundle/"
