# Implementing the economical AWS RAG architecture

Maps [`drugcentral_aws_rag_architecture_plan.md`](drugcentral_aws_rag_architecture_plan.md) to the code in this repository, with the switches and the checks
that prove each part. Everything here runs on one host (EC2 or this machine) plus Bedrock.

## Plan section → implementation

| Plan section | Implementation | Status |
|---|---|---|
| 2. Full-source inventory and source-to-retrieval map | `ingestion/schema_inventory.py` → `docs/source-inventory.json`, `docs/SOURCE_RETRIEVAL_MAP.md` | Done; 70 tables, 43 not yet in any retrieval store |
| 2. Derived records trace back to source | `chatbot/provenance.py`; `Evidence.as_dict()['provenance']` (source tables, `struct_id`, dataset version, section) | Done |
| 4–7. Hybrid retrieval, RRF, rerank, Qwen, validation | Already present: `chatbot/hybrid_search.py`, `production_models.py`, `production_pipeline.py` | Unchanged |
| 8–10. Router, decomposition, per-component retrieval/generation/validation, deterministic composer | `chatbot/components.py` (`plan_question` planner, composer), `chatbot/production_pipeline.py` (`_answer` routing, `run_component`, `answer_components`) | Done; switch `RAG_MULTIPART=0` restores the single path |
| 11. Partial answers | Components without validated claims are named under “Not verified…”; supported ones survive | Done |
| 12. Evidence indexes instead of opaque ids | `generation_evidence`, `resolve_indexes` in `chatbot/components.py`; the model never sees `ev_…` ids | Done |
| 13. Layered validation, one constrained repair | Unchanged deterministic + semantic validation; repair is now per component (`generate_validated`) | Done |
| 14–17. Neo4j graph, constrained queries, multi-hop | `ingestion/graph_build.py`, `chatbot/graph_query.py` (five fixed Cypher templates; no model-written Cypher) | Done; `GRAPH_ENABLED=1` |
| 18–19. Router: graph (RELATIONSHIP / MULTI_HOP), then SIMPLE / MULTI_PART | `chatbot/production_pipeline.py::_answer`: with `GRAPH_ENABLED=1`, `graph_query.answer_graph` runs first (`parse_graph_query` picks one of the fixed templates) and returns nothing when the question does not match or the graph is off or down; otherwise `components.plan_question` returns SIMPLE or MULTI_PART, and MULTI_PART goes to `answer_components` unless `RAG_MULTIPART=0`. No separate classifier module | Done |
| 6, 20–22. EC2 + EBS + Docker Compose, no ECS/RDS/ALB | `environments/ec2/` (compose, bundle/restore scripts, runbook) | Done; migration is manual |
| 2. Every table and view answerable (all 70 tables + 4 views) | `ingestion/full_coverage.py`, `ingestion/index_collection.py`, runbook `docs/FULL_COVERAGE.md`; coverage guard test | Built and tested locally; embedding (~$5.50) not yet run |
| 24. Complex benchmark and metrics | `evaluation/build_complex_questions.py`, `complex_metrics.py`, `run_complex_benchmark.py`; frozen set in `evaluation/complex-110/` | Built; baseline not yet run |

## What the graph contains

Built from PostgreSQL only; every relationship stores `source_table`, `source_pk` and `dataset_version`.

Nodes: Drug, Target, Gene, Disease, ATC, Approval, DrugClass (Product optional via `--with-products`).
Relationships: TARGETS, ENCODED_BY, INDICATED_FOR, CONTRAINDICATED_FOR, OFF_LABEL_FOR, TREATS_SYMPTOMS_OF,
REDUCES_RISK_OF, HAS_ATC, HAS_APPROVAL, IN_CLASS, INTERACTS_WITH.

DrugCentral records interactions between **drug classes**, not between individual drugs, so the graph and the answers say
“interaction at the drug-class level”. Adverse events (≈473k FAERS rows) are deliberately not in the graph; they stay in
hybrid RAG.

Supported graph questions (anything else goes through normal RAG):

* Which drugs target X [and are indicated for Y]? (1 hop)
* Which drugs share targets with D? / same ATC class as D? / What ATC classes do drugs targeting G have? (2–3 hops)
* Which drug classes interact with D?

## Switches

| Variable | Default | Effect |
|---|---|---|
| `RAG_MULTIPART` | `1` | `0` disables decomposition (single retrieval for the whole question) |
| `GRAPH_ENABLED` | `0` | `1` routes relationship questions to Neo4j (needs `NEO4J_URI`, `NEO4J_PASSWORD`) |
| `NEO4J_DATABASE`, `GRAPH_TIMEOUT_SECONDS` | `neo4j`, `15` | graph connection and query timeout |
| `RETRIEVAL_CANDIDATES`, `RERANK_TOP_K` | `20`, `8` | per-component evidence budget; change only with benchmark evidence |

## Cost behaviour to expect

A multi-part question runs retrieval, rerank, generation and verification **once per component**, so a four-field
question costs roughly four times a single-field one. Component retrieval is concurrent, so latency grows far less.
The benchmark runner has `--estimate` (call upper bound) and `--limit` so the first run can be small. Graph answers make
no model calls at all.

## Commands

```bash
# Inventory (needs the source database)
SOURCE_DATABASE_URL=postgresql://postgres:…@HOST:PORT/drugcen python -m ingestion.schema_inventory
# Graph
NEO4J_URI=bolt://HOST:7687 NEO4J_PASSWORD=… python -m ingestion.graph_build --rebuild
# Benchmark (builds gold from source; Bedrock is only used by the runner)
python -m evaluation.build_complex_questions --output evaluation/complex-110
python -m evaluation.run_complex_benchmark --questions evaluation/complex-110/questions.json --estimate
python -m evaluation.run_complex_benchmark --questions evaluation/complex-110/questions.json --output benchmarks/complex-first10.jsonl --limit 10
python -m evaluation.complex_metrics --questions evaluation/complex-110/questions.json --results benchmarks/complex-first10.jsonl
```

## Local source database with real RDKit

The published `unmtransinfo/drugcen-2027-db` image is amd64-only. `environments/ec2/source-db.Dockerfile` builds the
same stack (PostgreSQL 16 + RDKit `Release_2026_03_6`, extension 4.8.0) for arm64 or amd64, so the full archive
(`pg_restore` of `data/source/drugcen-2027-image.dump`) restores with no errors and the `mol`/`bfp` columns intact.

## Known limits (by design, pending measurements)

* The planner is deterministic. There is no LLM planner fallback; add one only if the benchmark shows rule misses.
* Product-label sections (warnings, dosage, populations) are not planner fields; long-label questions use the existing
  label-aware single path. The benchmark's `long_label` category will show whether they need components.
* `evaluation/complex-110` is unreviewed: gold facts come straight from source tables, but a person should review the
  questions before the scores are treated as results.
* The graph answers only the five template shapes; broader multi-hop (e.g. disease → gene → protein → drug) is not
  possible because DrugCentral has no disease–gene association table.
