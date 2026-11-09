#!/usr/bin/env bash
# Opt-in local-mode smoke test (RAG_PIPELINE=local) on synthetic data. See tests/integration/README.md.
#   tests/integration/run.sh                          # chat model qwen3.5:2b, API on 127.0.0.1:18000
#   ITEST_API_PORT=18080 tests/integration/run.sh     # another localhost port
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1

export ITEST_CHAT_MODEL=${ITEST_CHAT_MODEL:-qwen3.5:2b}   # smoke-test default; the application default is qwen3.5:9b
export ITEST_API_PORT=${ITEST_API_PORT:-18000}
export ITEST_OLLAMA_MEMORY=${ITEST_OLLAMA_MEMORY:-4g}
PROJECT=${ITEST_PROJECT:-dc-itest-$(date +%Y%m%d%H%M%S)-$$}
KEEP=${ITEST_KEEP:-0}

fail() { echo "ERROR: $*" >&2; exit 2; }
[[ $PROJECT =~ ^dc-itest-[a-z0-9_-]+$ ]] || fail "ITEST_PROJECT must match dc-itest-[a-z0-9_-]+ (got '$PROJECT')"
[[ $ITEST_API_PORT =~ ^[0-9]+$ ]] || fail "ITEST_API_PORT must be a port number"
command -v docker >/dev/null || fail "docker is required"
docker compose version >/dev/null 2>&1 || fail "Docker Compose v2 is required"
[ -z "$(docker ps -aq --filter "label=com.docker.compose.project=$PROJECT")" ] || fail "Compose project $PROJECT already exists"
if (exec 3<>"/dev/tcp/127.0.0.1/$ITEST_API_PORT") 2>/dev/null; then fail "127.0.0.1:$ITEST_API_PORT is in use; set ITEST_API_PORT"; fi

# Never inherit secrets from the caller's shell; generate disposable ones in a private directory.
unset PGPASSWORD SOURCE_DB_PASSWORD API_AUTH_TOKEN
SECRETS=$(umask 077 && mktemp -d)
rnd() { od -An -tx1 -N24 /dev/urandom | tr -d ' \n'; }
( umask 077; { echo "PGPASSWORD=$(rnd)"; echo "SOURCE_DB_PASSWORD=$(rnd)"; echo "API_AUTH_TOKEN=$(rnd)"; } > "$SECRETS/env" )
C=(docker compose -p "$PROJECT" -f environments/local/docker-compose.yml -f tests/integration/compose.yml
   --env-file "$SECRETS/env" --profile ollama)

cleanup() {
  if [ "$KEEP" = 1 ]; then
    echo "ITEST_KEEP=1: project $PROJECT kept (secrets in $SECRETS). Remove with:"
    echo "  ${C[*]} down -v --rmi local --remove-orphans && rm -rf $SECRETS"
    return
  fi
  echo "== Cleanup (project $PROJECT only)"
  "${C[@]}" down -v --rmi local --remove-orphans >/dev/null 2>&1
  rm -rf "$SECRETS"
}
trap cleanup EXIT
FAILED=0
step() { echo; echo "== $*"; }
run() { "$@" || { FAILED=$((FAILED + 1)); echo "FAIL  $*" | sed "s#$SECRETS#<secrets>#g"; }; }
APP=("${C[@]}" run --rm --no-deps -T api)

echo "project=$PROJECT chat_model=$ITEST_CHAT_MODEL embed_model=nomic-embed-text api=http://127.0.0.1:$ITEST_API_PORT"
step "Build the local-lock Python image"
"${C[@]}" build -q api proxy || exit 1
step "Start pgvector, Ollama (pinned digest) and the synthetic source database"
"${C[@]}" up -d --wait db ollama src || exit 1
docker inspect "$("${C[@]}" ps -q ollama)" --format 'ollama image: {{.Config.Image}}'
"${C[@]}" exec -T ollama ollama --version
step "Pull models"
for model in nomic-embed-text "$ITEST_CHAT_MODEL"; do
  "${C[@]}" exec -T ollama ollama pull "$model" >/dev/null 2>&1 || fail "could not pull $model"
done
"${C[@]}" exec -T ollama ollama list

step "Isolation of ingestion and API containers"
run "${APP[@]}" python tests/integration/checks.py isolation
step "Embedding"
run "${APP[@]}" python tests/integration/checks.py embedding
step "Ingestion"
run "${APP[@]}" python -m ingestion.vectors
"${C[@]}" exec -T db psql -U rag_local -d drugcentral_rag -Atc \
  "SELECT 'chunks=' || count(*) || ' drugs=' || count(DISTINCT metadata->>'drug_name') FROM drugcentral_chunks;
   SELECT 'stamp: backend=' || backend || ' model=' || model || ' dimensions=' || dimensions FROM drugcentral_collections;"

step "API (in-network)"
"${C[@]}" up -d api proxy || exit 1
run "${APP[@]}" python tests/integration/checks.py api

step "Host checks on 127.0.0.1:$ITEST_API_PORT"
BASE=http://127.0.0.1:$ITEST_API_PORT
if curl -fsS "$BASE/docs" | grep -q swagger-ui; then echo "PASS  $BASE/docs"; else FAILED=$((FAILED + 1)); echo "FAIL  $BASE/docs"; fi
if command -v python3 >/dev/null; then
  set -a; . "$SECRETS/env"; set +a   # token reaches cli.ask through its environment, never argv
  run python3 -m cli.ask --url "$BASE" --check
  run python3 -m cli.ask --url "$BASE" "Which drugs target ACE?"
  unset PGPASSWORD SOURCE_DB_PASSWORD API_AUTH_TOKEN
else
  echo "SKIP  cli.ask (python3 not found on the host)"
fi

step "API errors"
"${C[@]}" logs api 2>&1 | grep -iE "traceback|error" | head -5 || true
echo
echo "chat_model=$ITEST_CHAT_MODEL runtime=${SECONDS}s failed_steps=$FAILED"
[ "$FAILED" = 0 ] && echo "RESULT: PASS" || echo "RESULT: FAIL"
exit $([ "$FAILED" = 0 ] && echo 0 || echo 1)
