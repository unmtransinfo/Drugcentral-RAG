"""HTTP interface for the local DrugCentral RAG backend."""

import logging
import math
import os
import secrets
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, BoundedSemaphore
from uuid import UUID
from time import monotonic
import httpx
from chatbot.conversation import SessionContext, create_session_context
from chatbot.model_config import chat_backend, chat_model_id

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()
logger = logging.getLogger(__name__)
QUEUE_WAIT_SECONDS = min(30.0, max(0.0, float(os.getenv("API_QUEUE_WAIT_SECONDS", "2"))))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the database once per process, before accepting requests.
    from agent import agent_executor, invoke_english_only, evidence_index

    app.state.answer_question = lambda question: invoke_english_only(
        agent_executor, question
    )
    app.state.fast_answer = None
    app.state.answer_mode = "vector-rag"
    app.state.evidence_index = evidence_index
    from chatbot.embedding_config import embedding_model_id
    app.state.embedding_model = embedding_model_id()
    app.state.chat_model = chat_model_id()
    app.state.chat_backend = chat_backend()
    app.state.sessions = create_session_context()
    yield


API_DESCRIPTION = """Verified, cited answers over DrugCentral.

**Authentication.** When the server sets `API_AUTH_TOKEN`, `/api/chat`, `/api/metrics` and the evaluation routes
require `Authorization: Bearer <token>`; use **Authorize** below. `/api/health`, `/api/ready` and these docs are public.
Viewing the documentation of a hosted instance does not grant permission to call its protected routes.

**Modes.** `RAG_PIPELINE=local` answers with Ollama models on the developer's machine; `RAG_PIPELINE=verified-bedrock`
uses Amazon Bedrock with the AWS credentials of the environment running the server. `/api/ready` reports which one.
"""

app = FastAPI(title="DrugCentral RAG API", version="1.0.0", description=API_DESCRIPTION, lifespan=lifespan)
app.state.inference_lock = Lock()
app.state.request_slots = BoundedSemaphore(4)
app.state.sessions = SessionContext()
# Declare Swagger authentication without rejecting optional auth or changing the
# existing require_auth/require_evaluation_auth checks and error precedence.
bearer_auth = HTTPBearer(auto_error=False)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("API_CORS_ORIGINS", "").split(",")
        if origin.strip()
    ],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None


class ChatResponse(BaseModel):
    answer: str


class Metrics:
    """In-memory request and latency counters; question text is never stored."""
    def __init__(self, window=1000):
        self.lock = Lock()
        self.counts = Counter()
        self.seconds = deque(maxlen=window)
        self.started = monotonic()

    def record(self, route, status, seconds):
        with self.lock:
            self.counts[f"{route}:{status}"] += 1
            self.seconds.append(seconds)

    @staticmethod
    def _percentile(ordered, fraction):
        return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)] if ordered else 0.0

    def snapshot(self):
        with self.lock:
            ordered = sorted(self.seconds)
            counts = dict(self.counts)
        rss_kb = 0
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
        except OSError:
            pass
        return {"uptime_seconds": round(monotonic() - self.started, 1),
                "requests": counts,
                "chat_seconds": {"count": len(ordered), "p50": self._percentile(ordered, 0.5),
                                 "p95": self._percentile(ordered, 0.95)},
                "memory_rss_mb": round(rss_kb / 1024, 1)}


app.state.metrics = Metrics()


def require_auth(request: Request):
    """Optional bearer auth; active only when API_AUTH_TOKEN is configured."""
    expected = os.getenv("API_AUTH_TOKEN", "")
    if not expected:
        return
    header = request.headers.get("authorization", "")
    provided = header[7:].strip() if header.lower().startswith("bearer ") else ""
    if not provided or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Unauthorized",
                            headers={"WWW-Authenticate": "Bearer"})


@app.get("/api/health")
def health():
    """Process liveness only; does not invoke models or the database."""
    return {"status": "ok"}


@app.get("/api/ready")
def ready(request: Request):
    index = getattr(request.app.state, "evidence_index", None)
    if index is None or not index.ids:
        raise HTTPException(status_code=503, detail="Dataset is not ready")
    # Model invocation is checked by the deployment smoke test, not each health probe.
    # A successful data read detects a database outage without paid inference.
    try:
        from chatbot.storage import _connect
        from contextlib import closing
        with closing(_connect()) as conn, conn, conn.cursor() as cursor:
            cursor.execute("SELECT 1")
    except Exception:
        raise HTTPException(status_code=503, detail="Storage is not ready") from None
    provider = getattr(request.app.state, "chat_backend", "unknown")
    if provider == "ollama":
        # Local mode: the configured chat model must be pulled into the local Ollama server.
        try:
            response = httpx.get(os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/") + "/api/tags", timeout=3, trust_env=False)
            response.raise_for_status()
            models = {m["name"] for m in response.json()["models"]}
            model = request.app.state.chat_model
            if model not in models and model + ":latest" not in models:
                raise ValueError("Chat model unavailable")
        except Exception:
            raise HTTPException(status_code=503, detail="Ollama is not ready") from None
    return {"status": "ready", "provider": provider, "drugs": len(index.ids),
            "dataset_version": sorted(index.versions)}


@app.get("/api/metrics", dependencies=[Security(bearer_auth)])
def metrics(request: Request):
    """Request counts, chat latency percentiles, and process memory."""
    require_auth(request)
    return request.app.state.metrics.snapshot()


@app.post("/api/chat", response_model=ChatResponse, dependencies=[Security(bearer_auth)])
def chat(payload: ChatRequest, request: Request):
    require_auth(request)
    started = monotonic()
    status = 200
    try:
        question = payload.question
        if payload.session_id and hasattr(request.app.state, "evidence_index"):
            try:
                question = request.app.state.sessions.prepare(str(payload.session_id), question, request.app.state.evidence_index)
            except Exception:
                logger.error("Session storage unavailable")
                raise HTTPException(status_code=503, detail="Conversation storage is temporarily unavailable") from None
        fast_answer = getattr(request.app.state, "fast_answer", None)
        if fast_answer is not None:
            answer = fast_answer(question)
            if answer is not None:
                logger.info("chat_completed route=direct seconds=%.3f", monotonic() - started)
                return ChatResponse(answer=answer)
        # Only model inference joins the bounded queue; exact lookups stay available.
        lock = request.app.state.inference_lock
        slots = request.app.state.request_slots
        if not slots.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="The assistant queue is full.", headers={"Retry-After": "5"})
        if not lock.acquire(timeout=QUEUE_WAIT_SECONDS):
            slots.release()
            raise HTTPException(
                status_code=503,
                detail="The assistant is busy. Please try again shortly.",
                headers={"Retry-After": "5"},
            )
        try:
            with capture_trace() as trace:
                answer = request.app.state.answer_question(question)
            record_trace(trace['route'], trace['evidence'], trace['observations'], reason=trace['reason'])
            logging.getLogger('uvicorn.error').info(
                "chat_completed route=%s reason=%s evidence_records=%d seconds=%.3f",
                trace['route'], trace['reason'] or 'none', len(trace['evidence']),
                monotonic() - started)
            from chatbot.intent import classify as classify_intent
            from chatbot.query_log import build_record, log_query
            log_query(build_record(question, trace, monotonic() - started,
                                   intent=classify_intent(question)))
            return ChatResponse(answer=answer)
        except Exception:
            logger.exception("DrugCentral inference failed")
            raise HTTPException(
                status_code=503,
                detail="The assistant is temporarily unavailable. Please try again later.",
            ) from None
        finally:
            lock.release()
            slots.release()
    except HTTPException as exc:
        status = exc.status_code
        raise
    finally:
        request.app.state.metrics.record("chat", status, monotonic() - started)


# Evaluation jobs poll over short HTTP requests so tunnel timeouts do not cut off inference.
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
from chatbot.evaluation_trace import capture_trace, record_trace

_eval_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag-evaluation")
_eval_jobs = {}
_eval_lock = Lock()
_eval_slots = BoundedSemaphore(4)


def require_evaluation_auth(request):
    if os.getenv("ENABLE_EVALUATION_API", "").lower() != "true":
        raise HTTPException(status_code=404, detail="Evaluation API is disabled")
    if not os.getenv("API_AUTH_TOKEN"):
        raise HTTPException(status_code=503, detail="Evaluation API requires API_AUTH_TOKEN")
    require_auth(request)


def evaluation_metadata(request):
    state = request.app.state
    return {"chat_model": getattr(state, "chat_model", "unknown"),
            "embedding_model": getattr(state, "embedding_model", "unknown"),
            "chat_backend": getattr(state, "chat_backend", "unknown"),
            "dataset_versions": sorted(getattr(getattr(state, "evidence_index", None), "versions", [])),
            "answer_mode": getattr(state, "answer_mode", "unknown"),
            "trace_version": 2}


@app.get("/api/evaluation/info", dependencies=[Security(bearer_auth)])
def evaluation_info(request: Request):
    require_evaluation_auth(request)
    return evaluation_metadata(request)


def _run_evaluation(job_id, payload, request):
    try:
        with _eval_lock:
            _eval_jobs[job_id]["status"] = "running"
        with capture_trace() as trace:
            response = chat(payload, request)
        result = {"answer": response.answer, "trace": trace, "metadata": evaluation_metadata(request)}
        with _eval_lock:
            _eval_jobs[job_id].update(status="done", result=result)
    except Exception:
        logger.exception("Evaluation request failed")
        with _eval_lock:
            _eval_jobs[job_id].update(status="error", error="Chatbot request failed; inspect local server logs")
    finally:
        _eval_slots.release()


@app.post("/api/evaluation/jobs", status_code=202, dependencies=[Security(bearer_auth)])
def start_evaluation(payload: ChatRequest, request: Request):
    require_evaluation_auth(request)
    if payload.session_id is not None:
        raise HTTPException(status_code=422, detail="Evaluation questions must be independent; omit session_id")
    if not _eval_slots.acquire(blocking=False):
        raise HTTPException(status_code=503, detail="Evaluation queue is full", headers={"Retry-After": "5"})
    job_id = str(uuid4())
    with _eval_lock:
        # Bound memory, retain completed jobs for up to an hour (or the latest 256).
        for key, value in list(_eval_jobs.items()):
            if value["status"] in {"done", "error"} and monotonic() - value["created"] > 3600:
                del _eval_jobs[key]
        while len(_eval_jobs) >= 256:
            key = next((k for k,v in _eval_jobs.items() if v["status"] in {"done", "error"}), None)
            if key is None:
                break
            del _eval_jobs[key]
        _eval_jobs[job_id] = {"status": "queued", "created": monotonic()}
    try:
        _eval_pool.submit(_run_evaluation, job_id, payload, request)
    except Exception:
        _eval_slots.release()
        with _eval_lock:
            del _eval_jobs[job_id]
        raise HTTPException(status_code=503, detail="Evaluation worker unavailable") from None
    return {"job_id": job_id}


@app.get("/api/evaluation/jobs/{job_id}", dependencies=[Security(bearer_auth)])
def get_evaluation(job_id: UUID, request: Request):
    require_evaluation_auth(request)
    with _eval_lock:
        job = _eval_jobs.get(str(job_id))
        if job is None:
            raise HTTPException(status_code=404, detail="Job expired or server restarted; submit again")
        return {k:v for k,v in job.items() if k != "created"}
