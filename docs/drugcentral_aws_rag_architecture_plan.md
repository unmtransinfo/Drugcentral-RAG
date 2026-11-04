# DrugCentral RAG Backend: Economical AWS Deployment and Retrieval Architecture

## 1. Purpose

This document defines a practical deployment and RAG architecture for
the DrugCentral chatbot/backend. The goal is to keep the AWS footprint
small and economical while supporting:

-   DrugCentral structured database access
-   Hybrid dense + lexical retrieval
-   Cohere embedding and reranking through Amazon Bedrock
-   Qwen-based answer generation and validation through Amazon Bedrock
-   Reliable multi-part and long-question handling
-   Optional knowledge-graph retrieval for relationship and multi-hop
    questions
-   Evidence-grounded answers with safe partial-answer behavior
-   Simple deployment and maintenance

A public website is **not** part of this plan.

The recommended AWS footprint is:

1.  **Amazon EC2** --- application, PostgreSQL/pgvector,
    retrieval/orchestration, and optionally Neo4j
2.  **Amazon Bedrock** --- embeddings, reranking, generation, planning,
    and semantic validation

Amazon EBS is used as the persistent disk attached to EC2. The existing
DrugCentral database image remains in Docker Hub and is pulled directly
to EC2.

------------------------------------------------------------------------

## 2. Full-Source Database Requirement

This project must be implemented against the **complete DrugCentral
source database**.

The system must not assume that the existing vectorized evidence
collection represents the complete knowledge available in DrugCentral.
The current RAG evidence collection is one retrieval representation of
source data; it is not the authoritative database itself.

Before extending multi-part, relationship, or multi-hop retrieval,
perform a complete database inventory covering:

-   all schemas and tables relevant to DrugCentral
-   primary and foreign keys
-   drug and active-ingredient entities
-   synonyms and identifiers
-   targets and target relationships
-   genes and proteins where represented
-   indications
-   contraindications
-   ATC classifications and DDD data
-   approvals and regulatory information
-   pharmaceutical products/formulations
-   structures and chemical properties
-   ADME/pharmacology data
-   drug-drug interactions
-   adverse-event data
-   FDA/product-label data
-   source metadata and provenance
-   relationship/junction tables
-   other tables containing facts that may be useful to chatbot
    questions

The inventory should produce a source-to-retrieval map such as:

  -----------------------------------------------------------------------
  Source data type                    Preferred retrieval mechanism
  ----------------------------------- -----------------------------------
  Exact structured facts              Direct PostgreSQL query

  Textual/long-form evidence          PostgreSQL FTS + pgvector hybrid
                                      RAG

  Entity aliases/identifiers          Direct PostgreSQL resolution

  Explicit relational data            PostgreSQL and optional Neo4j graph
                                      projection

  Long product-label sections         Section-aware hybrid retrieval

  Provenance/source records           Direct PostgreSQL lookup and
                                      evidence binding
  -----------------------------------------------------------------------

The architecture should therefore be understood as:

``` text
                FULL DRUGCENTRAL POSTGRESQL DATABASE
                              |
                              v
                   Schema / Source Inventory
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        Structured         Textual        Relationships
          Facts            Evidence
             |                |                |
             v                v                v
        PostgreSQL       FTS + pgvector    PostgreSQL /
        direct query       Hybrid RAG      Neo4j projection
             |                |                |
             +----------------+----------------+
                              |
                              v
                       Retrieval Router
                              |
                              v
                      Evidence Assembly
                              |
                              v
                        Amazon Bedrock
                              |
                              v
                       Validated Answer
```

### Source-of-truth rule

The following hierarchy should be enforced:

``` text
PostgreSQL full DrugCentral source DB
              |
              +--> pgvector collection
              |
              +--> PostgreSQL FTS indexes
              |
              +--> Neo4j knowledge graph
              |
              +--> caches
```

The lower layers are **derived representations**.

They must never silently become independent sources of truth.

Every derived record should retain enough information to trace it back
to the authoritative database, such as:

``` text
source_table
source_primary_key
drug_id / structure_id where applicable
evidence_id
source_id
dataset_version
section / field
source_type
```

### Full-source coverage does not mean vectorizing every field

Using the full source database does **not** mean every database column
should be converted into a vector.

The retrieval method should match the data:

``` text
Exact ID / formula / numeric field
        -> PostgreSQL

Narrative label / descriptive evidence
        -> hybrid RAG

Drug-target-disease relationship
        -> PostgreSQL relationship query or KG

Complex multi-part question
        -> decomposition + multiple appropriate retrieval paths
```

This prevents unnecessary embedding cost and preserves exact structured
values.

### Knowledge-graph coverage

If Neo4j is introduced, its schema must be designed from the **full
PostgreSQL schema inventory**, not only from the current vector
collection.

For example, if important drug-target, indication, approval, ATC,
interaction, gene, product, or other relationships exist in source
tables that are not represented in the vector evidence collection, those
relationships should still be eligible for the graph.

Neo4j remains a derived relationship index. PostgreSQL remains
authoritative.

------------------------------------------------------------------------

## 3. Design Principles

The architecture should remain simple until scale or availability
requirements justify additional managed services.

Key principles:

-   Keep PostgreSQL as the authoritative DrugCentral data source.
-   Reuse the existing Docker Hub DrugCentral database image.
-   Run the application and database on a CPU-based EC2 instance.
-   Do not run large language models on EC2.
-   Use Bedrock for model inference so no GPU EC2 instance is required.
-   Keep the existing hybrid retrieval design: dense retrieval +
    PostgreSQL full-text search + RRF + reranking.
-   Decompose multi-part questions before retrieval instead of
    increasing context size blindly.
-   Use a graph only for questions that benefit from explicit
    relationship traversal.
-   Preserve source provenance and validate final claims.
-   Prefer partial supported answers over unnecessary full abstentions.
-   Add AWS services only when they solve a demonstrated operational
    problem.

------------------------------------------------------------------------

## 4. High-Level Architecture

``` text
                    API Client / Chatbot Consumer
                              |
                              v
                 +---------------------------+
                 |        Amazon EC2         |
                 |                           |
                 | DrugCentral RAG API       |
                 | Python / FastAPI          |
                 |                           |
                 | Query Router / Planner    |
                 | Retrieval Orchestration   |
                 | RRF Fusion                |
                 | Evidence Assembly         |
                 | Validation Logic          |
                 | Response Composition      |
                 |                           |
                 | +-----------------------+ |
                 | | PostgreSQL            | |
                 | | DrugCentral DB        | |
                 | | pgvector              | |
                 | | PostgreSQL FTS        | |
                 | +-----------------------+ |
                 |                           |
                 | +-----------------------+ |
                 | | Neo4j (optional)      | |
                 | | Relationship / KG     | |
                 | +-----------------------+ |
                 |                           |
                 |       EBS gp3 storage     |
                 +-------------+-------------+
                               |
                               | AWS SDK
                               v
                 +---------------------------+
                 |      Amazon Bedrock       |
                 |                           |
                 | Cohere Embed v4           |
                 | Cohere Rerank 3.5         |
                 | Qwen3 Next 80B A3B        |
                 +---------------------------+
```

The database Docker image is pulled from Docker Hub directly onto EC2:

``` text
Docker Hub
    |
    | docker pull
    v
Amazon EC2
    |
    +-- DrugCentral PostgreSQL container
```

------------------------------------------------------------------------

## 5. Core AWS Services

### 4.1 Amazon EC2

EC2 is the main compute host.

It runs:

-   DrugCentral RAG API
-   Python/FastAPI application
-   PostgreSQL
-   pgvector
-   PostgreSQL full-text search
-   retrieval logic
-   reciprocal-rank fusion
-   question routing
-   multi-part decomposition orchestration
-   evidence management
-   deterministic validation
-   response composition
-   caching
-   optional Neo4j Community Edition

A GPU instance is not required because embedding, reranking, generation,
and semantic validation are sent to Bedrock.

### 4.2 Amazon EBS

Use an EBS gp3 volume for persistent application and database storage.

Store:

-   PostgreSQL data
-   pgvector collection
-   Docker volumes
-   optional Neo4j data
-   application configuration
-   logs or local caches as appropriate

The PostgreSQL database must not depend only on the writable layer of a
Docker container.

Conceptually:

``` text
PostgreSQL Docker Container
            |
            v
      Docker Volume
            |
            v
         EBS gp3
```

This allows the database container to be replaced or upgraded without
losing persistent data.

### 4.3 Amazon Bedrock

Bedrock provides the model layer.

Recommended responsibilities:

  ---------------------------------------------------------------------
  Function                           Model
  ---------------------------------- ----------------------------------
  Embedding                          Cohere Embed v4

  Reranking                          Cohere Rerank 3.5

  Answer generation                  Qwen3 Next 80B A3B

  Semantic entailment validation     Qwen3 Next 80B A3B

  Query planning/decomposition when  Qwen3 Next 80B A3B or a cheaper
  an LLM is required                 suitable model
  ---------------------------------------------------------------------

The application accesses Bedrock through the AWS SDK and IAM
credentials/instance role rather than hardcoded access keys.

------------------------------------------------------------------------

## 6. Docker Deployment

Docker Compose is sufficient for the current architecture. ECS, EKS, and
Kubernetes are not necessary.

A conceptual deployment is:

``` text
EC2
 |
 +-- drugcentral-api
 |
 +-- drugcentral-db
 |      |
 |      +-- PostgreSQL
 |      +-- pgvector
 |
 +-- neo4j        [optional]
 |
 +-- persistent EBS volumes
```

The existing database image can be pulled directly from Docker Hub:

``` bash
docker pull unmtransinfo/drugcen-2027-db
```

A simplified Compose structure could look like:

``` yaml
services:
  db:
    image: unmtransinfo/drugcen-2027-db
    restart: unless-stopped
    volumes:
      - drugcentral-db:/var/lib/postgresql/data

  api:
    image: drugcentral-rag-api
    restart: unless-stopped
    depends_on:
      - db
    environment:
      DATABASE_HOST: db

  neo4j:
    image: neo4j
    restart: unless-stopped
    profiles:
      - graph
    volumes:
      - neo4j-data:/data

volumes:
  drugcentral-db:
  neo4j-data:
```

Neo4j should be optional so the base system can run without allocating
resources to it.

------------------------------------------------------------------------

## 7. Existing Retrieval Stack

The existing retrieval architecture should be preserved because it
already performs strongly on the frozen evaluation set.

The retrieval path is:

``` text
Question
   |
   +---------------------+
   |                     |
   v                     v
Cohere Embed v4     PostgreSQL FTS
   |                     |
   v                     v
Dense Ranking       Lexical Ranking
   |                     |
   +----------+----------+
              |
              v
             RRF
              |
              v
      Candidate Evidence
              |
              v
     Cohere Rerank 3.5
              |
              v
       Selected Evidence
              |
              v
       Qwen Generation
              |
              v
          Validation
```

### Dense retrieval

-   Cohere Embed v4 creates query embeddings.
-   Embeddings are compared against DrugCentral evidence vectors in
    pgvector.
-   Current evaluated embeddings use 1024 dimensions.

### Lexical retrieval

PostgreSQL full-text search is used in parallel with dense retrieval.

This is valuable for:

-   exact drug names
-   gene symbols
-   ATC codes
-   numerical values
-   identifiers
-   terminology where lexical matching is strong

### Reciprocal-rank fusion

Dense and lexical rankings are combined using RRF.

This avoids relying on a raw cosine-similarity threshold.

### Reranking

The fused candidates are sent to Cohere Rerank 3.5.

A practical configuration remains approximately:

``` text
Initial candidates: ~20
Final evidence:     ~5-8
```

These values should remain configurable and should be changed only when
evaluation supports doing so.

------------------------------------------------------------------------

## 8. Query Routing

Not every question should use the same retrieval strategy.

The application should classify requests into several broad types:

``` text
                         Question
                            |
                            v
                       Query Router
                            |
          +-----------------+------------------+
          |                 |                  |
          v                 v                  v
       SIMPLE          MULTI-PART        RELATIONSHIP /
                                             MULTI-HOP
          |                 |                  |
          v                 v                  v
 Structured lookup    Decomposition       Knowledge Graph
 and/or Hybrid RAG    + Hybrid RAG        + source evidence
```

The router does not need to be unnecessarily complex. Deterministic
rules can handle obvious cases, with an LLM planner used when necessary.

------------------------------------------------------------------------

## 9. Simple Questions

Example:

> What is the molecular formula of metformin?

This should use the cheapest reliable path.

If an exact structured lookup is available, use it.

Otherwise:

``` text
Question
   |
   v
Hybrid Retrieval
   |
   v
Reranking
   |
   v
Evidence
   |
   v
Qwen
   |
   v
Validation
```

There is no need to invoke Neo4j or multi-part decomposition.

------------------------------------------------------------------------

## 10. Multi-Part and Long Questions

Multi-part questions are a major area where the architecture should
improve.

Example:

> What are the common adverse reactions, targets, indications, and ATC
> classification of metformin?

Do not treat this as one large retrieval request.

### 9.1 Decompose first

The planner creates independent components:

``` text
Original Question
       |
       v
Question Planner
       |
       +-- C1: common adverse reactions of metformin
       |
       +-- C2: targets of metformin
       |
       +-- C3: indications of metformin
       |
       +-- C4: ATC classification of metformin
```

A structured representation can be:

``` json
{
  "entity": "metformin",
  "components": [
    {
      "id": "c1",
      "field": "adverse_reactions"
    },
    {
      "id": "c2",
      "field": "targets"
    },
    {
      "id": "c3",
      "field": "indications"
    },
    {
      "id": "c4",
      "field": "atc"
    }
  ]
}
```

### 9.2 Retrieve per component

Each component receives its own retrieval budget:

``` text
C1 --> Hybrid retrieval --> Rerank --> E1
C2 --> Hybrid retrieval --> Rerank --> E2
C3 --> Hybrid retrieval --> Rerank --> E3
C4 --> Hybrid retrieval --> Rerank --> E4
```

Independent components should run concurrently where practical.

This prevents evidence for one requested field from crowding out another
field.

### 9.3 Generate and validate per component

Instead of generating one large answer and repairing the entire response
when one claim fails:

``` text
C1 --> Generate --> Validate
C2 --> Generate --> Validate
C3 --> Generate --> Validate
C4 --> Generate --> Validate
```

Only the failed component should require repair.

### 9.4 Compose the final answer

Validated component answers are combined after validation.

The final composer should be deterministic where possible.

``` text
Validated C1
Validated C2
Validated C3
Validated C4
     |
     v
Response Composer
     |
     v
Final Answer
```

This avoids introducing new unsupported facts in an unnecessary final
LLM synthesis call.

------------------------------------------------------------------------

## 11. Partial Answers

The system should distinguish between:

-   a fully supported component
-   a partially supported component
-   an unsupported component

Example:

> Give one regulatory agency and approval year for benzoin.

Evidence:

``` text
FDA (N/A)
```

A safe response is:

``` text
The recorded regulatory agency is FDA. An approval year is not
recorded in the available evidence.
```

The system should not discard the supported agency merely because the
year is unavailable.

For a four-component question, a failure in one component should not
cause the other three supported components to disappear.

------------------------------------------------------------------------

## 12. Evidence References

The LLM should not be required to reproduce long opaque evidence IDs.

Instead of:

``` json
{
  "claim": "...",
  "evidence_id": "drugcentral-section-1725-target-382712"
}
```

send compact local evidence indexes:

``` text
[0] evidence ...
[1] evidence ...
[2] evidence ...
```

Then the model returns:

``` json
{
  "claim": "Sulindac has a bioavailability of 88%.",
  "evidence": [1]
}
```

The backend maps index `1` to the real immutable evidence ID.

``` text
Model evidence index
        |
        v
Backend mapping
        |
        v
Real evidence ID
        |
        v
Original source
```

This reduces failures caused by the model incorrectly copying evidence
identifiers.

------------------------------------------------------------------------

## 13. Validation

Validation should remain layered.

### Layer 1: deterministic validation

Check items that software can verify reliably:

-   evidence ID/index exists
-   cited evidence belongs to the selected evidence set
-   drug/entity names
-   numerical values
-   units
-   value-unit pairs
-   identifiers
-   gene symbols
-   route
-   species
-   negation
-   qualifiers
-   relationship direction
-   source type where relevant

Claims that fail deterministic validation should not proceed as
supported claims.

### Layer 2: semantic entailment

For claims that pass deterministic validation, use the separate Bedrock
validation call to determine whether the selected evidence actually
supports the claim.

Conceptual states:

``` text
SUPPORTED
CONTRADICTED
INSUFFICIENT_EVIDENCE
```

### Repair

Allow one constrained repair attempt using only the already selected
evidence.

Do not perform uncontrolled repeated regeneration.

------------------------------------------------------------------------

## 14. Knowledge Graph for Relationship Questions

A graph database is useful when the question is fundamentally about
relationships rather than independent textual facts.

Example:

> Which drugs targeting EGFR are indicated for lung cancer?

Graph representation:

``` text
EGFR
 ^
 |
TARGETS
 |
Drug
 |
INDICATED_FOR
 |
 v
Lung Cancer
```

This is different from an ordinary multi-part question.

### Recommended graph technology

Initially run **Neo4j Community Edition as a Docker container on the
same EC2 instance**.

Do not introduce Amazon Neptune unless scale, operations, or
availability later justify it.

### PostgreSQL remains authoritative

Neo4j should be treated as a derived relationship index:

``` text
PostgreSQL
DrugCentral authoritative data
        |
        | graph build/update
        v
Neo4j
Derived relationship graph
```

The graph should be reproducible from authoritative DrugCentral data.

------------------------------------------------------------------------

## 15. Proposed Knowledge-Graph Schema

Potential nodes include:

``` text
Drug
Target
Gene
Disease
Indication
ATC
Approval
Ingredient
Product
AdverseEvent
```

Potential relationships include:

``` text
(Drug)-[:TARGETS]->(Target)

(Target)-[:ENCODED_BY]->(Gene)

(Drug)-[:INDICATED_FOR]->(Disease)

(Drug)-[:CONTRAINDICATED_FOR]->(Disease)

(Drug)-[:HAS_ATC]->(ATC)

(Drug)-[:INTERACTS_WITH]->(Drug)

(Drug)-[:HAS_ADVERSE_EVENT]->(AdverseEvent)

(Drug)-[:HAS_APPROVAL]->(Approval)

(Product)-[:CONTAINS]->(Drug)
```

Every important relationship should preserve provenance.

Example:

``` text
relationship.evidence_id
relationship.source_id
relationship.dataset_version
relationship.source_type
```

The graph therefore identifies relationships, while the final answer can
still be grounded in authoritative DrugCentral evidence.

------------------------------------------------------------------------

## 16. Relationship Query Execution

Do not allow the LLM to freely generate arbitrary Cypher and execute it
directly.

Use a constrained intermediate query representation.

Example:

``` json
{
  "intent": "RELATIONSHIP",
  "entity": {
    "type": "target",
    "value": "EGFR"
  },
  "relationship": "TARGETED_BY",
  "filter": {
    "indication": "lung cancer"
  },
  "return": "drug"
}
```

The backend maps this to a known Cypher template.

For example:

``` cypher
MATCH (t:Target {name: $target})
      <-[:TARGETS]-(d:Drug)
      -[:INDICATED_FOR]->(disease:Disease)
WHERE disease.name = $disease
RETURN DISTINCT d
```

Benefits:

-   easier validation
-   predictable performance
-   safer execution
-   easier testing
-   fewer hallucinated graph labels
-   reproducible behavior

------------------------------------------------------------------------

## 17. True Multi-Hop Questions

Some questions require one retrieval result to determine the next step.

Example:

> Which drugs target proteins encoded by genes associated with disease
> X?

Conceptually:

``` text
Disease
   |
   v
Gene
   |
   v
Protein / Target
   |
   v
Drug
```

These should use graph traversal or controlled iterative retrieval.

The router should distinguish:

``` text
SIMPLE
MULTI_PART
RELATIONSHIP
MULTI_HOP
```

Multi-part questions contain several independently answerable
components.

Multi-hop questions require chained relationships or dependent
retrieval.

------------------------------------------------------------------------

## 18. Complete Retrieval Strategy

The final backend can support three complementary mechanisms.

### Structured retrieval

Best for exact database facts.

Examples:

-   formula
-   identifiers
-   exact ATC code
-   approval record
-   exact structured values

### Hybrid RAG

Best for textual evidence.

Uses:

``` text
Cohere Embed v4
       +
PostgreSQL FTS
       |
       v
      RRF
       |
       v
Cohere Rerank
```

### Graph retrieval

Best for explicit relationship traversal.

Uses:

``` text
Neo4j
  |
  v
Graph paths
  |
  v
DrugCentral evidence IDs
```

The router chooses the appropriate path.

------------------------------------------------------------------------

## 19. Combined Architecture

``` text
                           QUESTION
                              |
                              v
                        QUERY PLANNER
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
       SIMPLE             MULTI-PART       RELATIONSHIP /
                                                MULTI-HOP
          |                   |                   |
          v                   v                   v
   Structured lookup      Decompose            Neo4j
      and/or RAG          components           traversal
          |                   |                   |
          |                   v                   |
          |              Hybrid RAG               |
          |                   |                   |
          +-------------------+-------------------+
                              |
                              v
                       EVIDENCE ASSEMBLY
                              |
                              v
                     CLAIM GENERATION
                       Amazon Bedrock
                              |
                              v
                       VALIDATION
                              |
                              v
                    RESPONSE COMPOSITION
                              |
                              v
                         FINAL ANSWER
```

------------------------------------------------------------------------

## 20. Services Not Required Initially

The economical architecture deliberately avoids unnecessary services.

### Amazon RDS

Not initially required because:

-   a working DrugCentral PostgreSQL Docker image already exists
-   PostgreSQL can run on EC2
-   pgvector can remain in the same database
-   this avoids another continuously billed managed service

RDS can be considered later if database high availability, automated
failover, and managed backups become operational priorities.

### Amazon ECR

Not required because the existing image is in Docker Hub.

EC2 can pull it directly.

### ECS / EKS / Fargate

Not required for a single-server deployment.

Docker Compose is sufficient and much simpler.

### Amazon OpenSearch

Not required because the architecture already has:

``` text
PostgreSQL FTS
+
pgvector
```

### Amazon Neptune

Not initially required.

Neo4j Community Edition can run on EC2 if graph retrieval is introduced.

Neptune can be evaluated later if the graph becomes large or requires
managed high availability.

### Bedrock Knowledge Bases

Not required because retrieval is already implemented and controlled by
the application.

### Application Load Balancer

Not necessary for one EC2 backend.

It becomes useful when multiple backend EC2 instances need traffic
distribution.

------------------------------------------------------------------------

## 21. Security

Use an EC2 IAM role for Bedrock access.

Do not place long-term AWS credentials in:

-   source code
-   Docker images
-   `.env` files committed to Git
-   configuration repositories

The application should use the normal AWS SDK credential chain.

Network exposure should also be minimized.

PostgreSQL and Neo4j should normally remain internal to the EC2
host/container network unless an explicit external requirement exists.

Example:

``` text
External API port
      |
      v
DrugCentral API

PostgreSQL 5432 ---- internal only
Neo4j ports    ---- internal only
```

------------------------------------------------------------------------

## 22. Cost Strategy

The main fixed infrastructure cost is:

``` text
EC2
+
EBS
```

AI inference remains usage-based through Bedrock.

This is preferable to maintaining a GPU EC2 instance continuously.

Cost controls should include:

-   CPU EC2 rather than GPU EC2
-   one server initially
-   gp3 storage
-   Bedrock on-demand inference
-   caching embeddings where appropriate
-   caching deterministic lookups
-   avoiding repeated generation
-   component-level repair rather than regenerating full answers
-   deterministic composition instead of an additional LLM call
-   routing simple questions to structured lookup when possible
-   invoking Neo4j only for graph-relevant requests
-   configurable candidate/evidence budgets

------------------------------------------------------------------------

## 23. Performance Strategy

### Parallel multi-part retrieval

Independent components should execute concurrently.

Instead of:

``` text
C1 -> C2 -> C3 -> C4
```

prefer:

``` text
      +-> C1
      |
      +-> C2
Q ----+
      +-> C3
      |
      +-> C4
```

Then combine results.

### Keep contexts small

Do not solve long questions by simply passing more and more evidence to
Qwen.

Each component should receive only the evidence relevant to that
component.

### Avoid unnecessary LLM calls

Use deterministic code for:

-   routing when obvious
-   exact database lookups
-   evidence-index mapping
-   claim metadata checks
-   response composition
-   graph query templates

Reserve LLM calls for tasks where language reasoning is actually needed.

------------------------------------------------------------------------

## 24. Evaluation Plan

Keep the existing frozen 240-question benchmark as the regression suite
for the existing system.

Add a separate complex-question benchmark.

Suggested composition:

  Question type                 Count
  ------------------------- ---------
  Independent 2-part               20
  Independent 3-part               15
  Multi-drug comparison            15
  Cross-section                    15
  True multi-hop                   15
  Long-label                       10
  Partially answerable             10
  Completely unanswerable          10
  **Total**                   **110**

### Retrieval metrics

Measure:

-   Recall@1
-   Recall@5
-   Recall@10
-   Recall@20
-   MRR
-   nDCG@10
-   component evidence recall

### Answer metrics

Measure:

-   faithfulness
-   factual correctness
-   citation accuracy
-   exact quote accuracy
-   unsupported claim rate

### Multi-part metrics

Add:

-   component completion rate
-   all-components-correct rate
-   unnecessary component omission rate
-   partial-answer accuracy
-   unnecessary full-abstention rate

For example, if a question asks for four components and the system
correctly returns three:

``` text
Component completion = 3 / 4 = 75%
```

This should be visible in evaluation rather than hidden inside a single
overall answer score.

### Operational metrics

Measure:

-   p50 latency
-   p95 latency
-   Bedrock calls/question
-   repair rate
-   abstention rate
-   approximate cost/question
-   approximate cost/1,000 questions

------------------------------------------------------------------------

## 25. Implementation Phases

### Phase 1 --- Minimal AWS backend

Deploy:

``` text
EC2
EBS
Docker
DrugCentral PostgreSQL image
DrugCentral RAG API
Bedrock integration
```

Do not add graph infrastructure yet.

### Phase 2 --- Multi-part question handling

Implement:

-   query decomposition
-   component schema
-   parallel component retrieval
-   component-scoped evidence
-   component-level generation
-   component-level validation
-   deterministic response composition
-   partial-answer handling

This should be the highest-priority architectural improvement because it
directly addresses the observed long/multi-part question behavior.

### Phase 3 --- Evidence-reference hardening

Replace model-generated opaque evidence IDs with local evidence indexes.

Implement:

``` text
model index
   ->
backend mapping
   ->
immutable evidence ID
```

### Phase 4 --- Complex-question evaluation

Build the separate multi-part/multi-hop benchmark and establish a
baseline before introducing graph retrieval.

### Phase 5 --- Knowledge graph

Add Neo4j only after the relationship/multi-hop benchmark demonstrates a
need.

Build the graph from PostgreSQL.

Initially support a limited set of high-value relationships:

``` text
Drug -> Target
Drug -> Indication
Drug -> ATC
Drug -> Approval
Drug -> Interaction
Target -> Gene
```

Attach provenance to graph relationships.

### Phase 6 --- Graph-aware routing

Add:

``` text
SIMPLE
MULTI_PART
RELATIONSHIP
MULTI_HOP
```

routing.

Use controlled graph query templates rather than unrestricted
model-generated Cypher.

### Phase 7 --- Optimize based on measured results

Only after evaluation should changes be considered to:

-   EC2 instance size
-   retrieval candidate count
-   final evidence count
-   model selection
-   graph coverage
-   caching
-   managed AWS services

------------------------------------------------------------------------

## 26. Recommended Initial Production Stack

  Layer                      Technology
  -------------------------- -----------------------------------------
  Compute                    Amazon EC2
  Persistent storage         Amazon EBS gp3
  Containers                 Docker / Docker Compose
  DrugCentral DB image       Docker Hub
  Database                   PostgreSQL
  Vector storage/search      pgvector
  Lexical retrieval          PostgreSQL FTS
  Dense embedding            Cohere Embed v4 via Bedrock
  Fusion                     Reciprocal-rank fusion
  Reranking                  Cohere Rerank 3.5 via Bedrock
  Generation                 Qwen3 Next 80B A3B via Bedrock
  Semantic validation        Qwen3 Next 80B A3B via Bedrock
  Backend                    Python / FastAPI
  Multi-part orchestration   Python + Bedrock planner where required
  Knowledge graph            Neo4j Community on EC2, optional
  Graph source of truth      PostgreSQL
  Authentication to AWS      EC2 IAM role / AWS SDK credential chain
  Deployment orchestration   Docker Compose

------------------------------------------------------------------------

## 27. Final Recommendation

The first production version should remain intentionally small:

``` text
                   Docker Hub
                       |
                       | DrugCentral DB image
                       v
             +----------------------+
             |      Amazon EC2      |
             |                      |
             | DrugCentral RAG API  |
             | PostgreSQL           |
             | pgvector             |
             | PostgreSQL FTS       |
             |                      |
             | Neo4j later          |
             |                      |
             | EBS persistent disk  |
             +----------+-----------+
                        |
                        | AWS SDK / IAM
                        v
             +----------------------+
             |   Amazon Bedrock     |
             |                      |
             | Cohere Embed v4      |
             | Cohere Rerank 3.5    |
             | Qwen3 Next 80B A3B   |
             +----------------------+
```

The immediate engineering priority should **not** be adding more AWS
services.

The recommended order is:

1.  Deploy the current RAG backend on EC2 with PostgreSQL/pgvector and
    Bedrock.
2.  Implement robust multi-part question decomposition and
    component-level retrieval.
3.  Harden evidence-reference handling.
4.  Build a complex-question evaluation set.
5.  Add Neo4j on the same EC2 instance only for relationship and true
    multi-hop questions.
6.  Introduce additional managed AWS services only if measured scale,
    reliability, or operational requirements justify their cost.

This keeps the system economical while preserving a clear upgrade path
from a strong hybrid RAG backend to a graph-aware DrugCentral retrieval
system.
