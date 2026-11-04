FROM python:3.11-slim-trixie AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl postgresql-client \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --silent --show-error https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem -o /opt/rds-ca.pem
WORKDIR /app
COPY environments/aws/requirements-aws.lock .
RUN pip install -r requirements-aws.lock
COPY *.py ./
COPY chatbot ./chatbot
COPY config/intent_fields.json ./config/intent_fields.json
COPY --chmod=0444 data/labels.sqlite /opt/drugcentral/labels.sqlite
RUN chmod 0755 /opt/drugcentral
ENV LABEL_DATABASE=/opt/drugcentral/labels.sqlite LABEL_REQUIRED=1
COPY ingestion ./ingestion
COPY evaluation ./evaluation
RUN useradd --uid 10001 --create-home app && mkdir -p /app/benchmarks /app/logs && chown -R app:app /app
ENV PGSSLROOTCERT=/opt/rds-ca.pem SOURCE_PGSSLROOTCERT=/opt/rds-ca.pem
USER app
RUN python -c "from types import SimpleNamespace; from chatbot.label_retrieval import search_labels; records, clarification = search_labels(SimpleNamespace(ids={'metformin': 1725}), 'What are the common side effects of metformin?', 'metformin'); assert not clarification and any(r['metadata']['section'] == 'adverse_reactions' for r in records), 'Label retrieval smoke test failed'"
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM base AS bedrock
