# data/

Not tracked in Git. The production image build needs `data/labels.sqlite`; the build fails with
`"/data/labels.sqlite": not found` until it exists. Running the API locally works without either file.

| File | Needed for | How to produce |
|---|---|---|
| `labels.sqlite` (~2.6 GB) | **Docker build** (copied to `/opt/drugcentral/labels.sqlite`; the build runs a metformin label smoke test against it); optional product-label answers in local mode | `python -m ingestion.labels SOURCE.sql.gz --output data/labels.sqlite` |
| `relations.sqlite` (~2 MB) | Optional, local mode only: exact drug–target lookups (Bedrock mode uses the Neo4j graph) | `python -m ingestion.relations SOURCE.sql.gz --output data/relations.sqlite` |

`SOURCE.sql.gz` is a DrugCentral PostgreSQL dump (the reference build used the 2026-09-02 release,
`dataset_version` `drugcentral-56-2026-09-02`). Obtain it from DrugCentral (https://drugcentral.org) under
DrugCentral's own terms; this repository does not redistribute DrugCentral data or anything derived from it.

Run the commands from the repository root in either virtual environment (docs/LOCAL_DEVELOPMENT.md). Both tools
build a temporary file, check it with `PRAGMA integrity_check`, and replace the output atomically.

The vector collections (pgvector), the Neo4j graph and the restore bundle are not files in this folder; see
`docs/LOCAL_DEVELOPMENT.md` (your own instance) and `environments/ec2/README.md` (AWS deployment).
