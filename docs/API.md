# API reference

The same FastAPI application serves both modes. Interactive documentation:

| Instance | Swagger UI | Who can call protected routes |
|---|---|---|
| Your own instance (either mode) | http://localhost:8000/docs (ReDoc `/redoc`, schema `/openapi.json`) | Anyone with the token you set in your `.env` (or anyone, if you left `API_AUTH_TOKEN` empty) |
| The maintainers' hosted API | Unavailable pending maintainer configuration (no public endpoint; the API binds to `127.0.0.1` on the instance) | Only the maintainers' operators, through an SSH tunnel and with their token; none is published |

## Endpoints

| Method and path | Auth | Purpose |
|---|---|---|
| `GET /api/health` | public | Process liveness; calls no model or database |
| `GET /api/ready` | public | `200` with `provider` (`ollama` or `bedrock`), `drugs`, `dataset_version` once data, database and (local mode) the Ollama chat model are available; `503` otherwise |
| `POST /api/chat` | bearer token when `API_AUTH_TOKEN` is set | Body `{"question": "...", "session_id": "<uuid>"?}` → `{"answer": "..."}`; the answer ends with `Sources: [1] [label](url) · …` |
| `GET /api/metrics` | bearer token when set | Request counts, chat latency percentiles, memory |
| `GET /api/evaluation/info`, `POST /api/evaluation/jobs`, `GET /api/evaluation/jobs/{id}` | always a token | Evaluation jobs with answer traces; `404` unless the server sets `ENABLE_EVALUATION_API=true` |
| `GET /docs`, `/redoc`, `/openapi.json` | public | Documentation |

## Authentication

* The server enables bearer authentication by setting `API_AUTH_TOKEN`. Clients send
  `Authorization: Bearer <token>`; a missing, malformed or wrong token returns `401` with `WWW-Authenticate: Bearer`.
* With `API_AUTH_TOKEN` unset, `/api/chat` and `/api/metrics` are open. The evaluation routes return `404` unless
  `ENABLE_EVALUATION_API=true`, and `503` if they are enabled without a token.
* Leave `API_AUTH_TOKEN` unset only on a machine where the API is bound to `127.0.0.1`.
* In Swagger UI, **Authorize** stores the token in your browser session only for requests to that instance.

## Errors

| Status | Meaning |
|---|---|
| `401` | Token required or invalid |
| `422` | Invalid request body (empty question, more than 4000 characters, unknown fields) |
| `503` | Not ready, busy (`Retry-After` header), or the model provider/database is unavailable |

## Examples (your own instance)

```bash
curl -s http://localhost:8000/api/ready
curl -s http://localhost:8000/api/chat -H 'Content-Type: application/json' \
     -H "Authorization: Bearer $API_AUTH_TOKEN" -d '{"question":"Which drugs target EGFR?"}'
python -m cli.ask "Which drugs target EGFR?"      # reads API_AUTH_TOKEN from the environment
```
