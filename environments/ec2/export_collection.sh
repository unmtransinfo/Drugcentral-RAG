#!/usr/bin/env bash
# Run on the DEVELOPMENT machine. Exports ONE vector collection (and its structured snapshot) as compressed CSV
# files for import on the EC2 host, so the full 4.8 GB database does not have to be shipped again.
# Usage: environments/ec2/export_collection.sh VECTOR_COLLECTION STRUCTURED_COLLECTION OUTPUT_DIR
set -euo pipefail
VEC="${1:?vector collection}"; STRUCT="${2:?structured collection}"; OUT="${3:?output directory}"
RAG_CONTAINER="${RAG_CONTAINER:-drugcentral-pgvector}"; RAG_DB="${RAG_DB:-drugcentral_rag}"
[ -e "$OUT" ] && { echo "Refusing to overwrite $OUT" >&2; exit 1; }
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
for spec in "drugcentral_chunks:$VEC" "drugcentral_evidence:$STRUCT" "drugcentral_collections:$VEC" "drugcentral_migrations:$VEC"; do
  table="${spec%%:*}"; coll="${spec#*:}"
  echo "Exporting $table ($coll) ..."
  docker exec "$RAG_CONTAINER" psql -U postgres -d "$RAG_DB" -v ON_ERROR_STOP=1 \
      -c "COPY (SELECT * FROM public.$table WHERE collection = '$coll' ORDER BY 1) TO STDOUT WITH (FORMAT csv)" | gzip > "$OUT/$table.csv.gz"
  [ "${PIPESTATUS[0]}" = "0" ] || { echo "export of $table failed" >&2; exit 1; }
done
printf 'VECTOR_COLLECTION=%s\nSTRUCTURED_COLLECTION=%s\n' "$VEC" "$STRUCT" > "$OUT/collection.env"
(cd "$OUT" && sha256sum * > SHA256SUMS); du -sh "$OUT"/*
echo "Done. Copy $OUT to the host and run environments/ec2/import_collection.sh"
