# Full-database coverage: all 70 tables and 4 views answerable

The base ingestion embeds 24 tables. `ingestion/full_coverage.py` adds the other 46 tables and the 4 views, so every
source object is reachable from retrieval. The two empty tables (`ijc_connect_items`, `ijc_connect_structures`) have no rows to embed.
`docs/SOURCE_RETRIEVAL_MAP.md` lists every object and where it is covered; a test fails if one is missing.

## What changes in the data

New content uses **new section names** only, so existing answers are unchanged (checked: all 71,624 existing
drug-and-section lists are identical, aliases identical).

| Kind | Tables / views | Stored as |
|---|---|---|
| Per drug, every row kept | `atc_ddd`, `identifier`, `inxight_drugs`, `structure_type`, `struct2parent`, `parentmol`, `pk_az`, `humanim`, `vet*`, `faers_male/female/ped/ger` | sections `ddd`, `identifiers`, `structure_info`, `pk_measurements`, `veterinary`, `adverse_events_male/female/pediatric/geriatric` |
| Views | `faers_top`, `ob_patent_view`, `ob_exclusivity_view`, `omop_relationship_doid_view` | `adverse_events_top`, `patent_details`, `doid_mapping` |
| LINCS | `lincs_signature` (8.76M pairwise rows) | one summary per drug: counts, cell lines, 10 most similar compounds (`lincs`) |
| Entity documents (title repeated in every chunk) | `target_dictionary`+`td2tc`+`target_component`+`tdgo2tc`+`tdkey2tc`+`act_table_full` | `target_profile` (one doc per target: class, components, GO terms, keywords, drugs) |
| | `doid`, `doid_xref` + view | `disease_profile` |
| | `target_go`, `target_keyword` | `ontology` (one doc per term) |
| | `reference` | `references` (one doc per reference) |
| | `ob_*_code`, small dictionaries (`action_type`, `id_type`, `data_source`, …) | `dictionary` |

Not embedded by design: the `molfile`/`molimg` columns (connection table and image bytes), and the 8.76M individual LINCS pairs.
Both are declared in `EXCLUDED_COLUMNS` / `COVERAGE`, not dropped silently.

Question routing: new cues in `chatbot/question_retrieval.py` (e.g. "UNII", "DDD", "GO terms", "PMID", "female patients"),
planner phrases and labels in `chatbot/components.py`, provenance (exact source tables per chunk) in `chatbot/provenance.py`.

## Measured size and cost (this machine, source = drugcen-2027 image, dataset version drugcentral-56-2026-09-02)

| | Chunks | Tokens (chars/4) | Embedding cost at $0.12/M |
|---|---:|---:|---:|
| Base dossier (24 tables, regenerated from the same source) | 151,517 | 9.7 M | $1.16 |
| New content | 134,311 | 36.0 M | $4.32 |
| **Whole new collection** | **285,828** | **45.7 M** | **≈ $5.50** |

The old collection stays untouched; the new one is a separate collection you can switch back from.
At the default throttle (`INGEST_TOKENS_PER_MINUTE=240000`) embedding takes about 3 hours. Raise the variable only if your Bedrock quota allows.
The embedding step is resumable: rerun the same command and it continues.

## Steps (run on this machine, where the source database is)

Previews already exist in `benchmarks/full-coverage/`. To regenerate them, both must use the **same** `DATASET_VERSION`:

```bash
cd /path/to/Drugcentral-RAG-EC2      # this repository, on the machine with the source database
export SOURCE_DATABASE_URL=postgresql://postgres:<source-db-password>@127.0.0.1:5435/drugcen DATASET_VERSION=drugcentral-56-2026-09-02
set -a; . environments/prod/bedrock-production.env; set +a; export AWS_REGION=us-east-2
PY=python      # with the drugcentral-rag-aws Conda environment active (environments/aws/environment.yml)
INGEST_PREVIEW_ONLY=1 INGEST_PREVIEW_PATH=benchmarks/full-coverage/base-preview.jsonl $PY -m ingestion.vectors      # no Bedrock calls
$PY -m ingestion.full_coverage preview --output benchmarks/full-coverage/extra-preview.jsonl                          # prints the cost
```

1. **Publish the structured snapshot** (local database only, no Bedrock):
   ```bash
   $PY -m ingestion.full_coverage publish --base benchmarks/full-coverage/base-preview.jsonl \
        --extra benchmarks/full-coverage/extra-preview.jsonl --collection drugcentral-sections-full-20261004
   ```
2. **Embed it into a new vector collection** (this spends about $5.50 of Bedrock tokens):
   ```bash
   AWS_PROFILE=<your-profile> $PY -m ingestion.cohere_migration --source drugcentral-sections-full-20261004 \
        --target drugcentral_full_cohere_v4_1024_20261004 --report benchmarks/full-coverage/migration-report.json --apply
   ```
3. **Create the search indexes** for the new collection:
   ```bash
   $PY -m ingestion.index_collection --collection drugcentral_full_cohere_v4_1024_20261004
   ```
4. **Evaluate before switching anything.** Point a local API at the new collection and compare it with the old one on the same
   questions (the 240-question set must not get worse; the 300-question full-schema set should answer far more):
   ```bash
   VECTOR_COLLECTION=drugcentral_full_cohere_v4_1024_20261004 STRUCTURED_COLLECTION=drugcentral-sections-full-20261004 \
     AWS_PROFILE=<your-profile> $PY -m uvicorn api:app --port 8010          # in one terminal
   ```
   then run `evaluation/evaluate_live_api.py` for each question set against `http://127.0.0.1:8010` (see
   `evaluation/full-schema-300-drugcen-2027-20261003/README.md`). The full-schema set needs independent review of its answers
   before any accuracy number is quoted; its mechanical checks only show whether the right facts were retrieved.
5. **Ship only the new collection to EC2** (no 4.8 GB reship):
   ```bash
   environments/ec2/export_collection.sh drugcentral_full_cohere_v4_1024_20261004 drugcentral-sections-full-20261004 ~/collection-bundle
   ```
   Upload `~/collection-bundle` to the host, then on the host:
   ```bash
   cd /srv/drugcentral/Drugcentral-RAG-EC2
   environments/ec2/import_collection.sh /srv/drugcentral/Drugcentral-RAG-EC2/collection-bundle
   mv environments/ec2/collection.env.new environments/ec2/collection.env
   unset VECTOR_COLLECTION STRUCTURED_COLLECTION            # a stale shell value would override the file
   docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env --profile graph up -d --force-recreate api
   ```
   Roll back by deleting `environments/ec2/collection.env` and recreating the API: the old collection is untouched.

Deploy the **code** with the new collection at the same time. New question cues route to new sections; against the old collection
a question like "what are the identifiers of X" would find no chunks and abstain.

## Alternative: run the whole build on the EC2 host

You can keep the source database on the server and build there, so only code moves from this machine and the new collection
never needs exporting. The embedding then uses the instance role (no AWS profile on this machine).

What this costs you: the source archive (1.3 GB) and, on an ARM instance, the source-database image (0.7 GB) must be uploaded once,
and the source database needs about 5 GB of disk and about 4-5 GB of RAM while it runs. On a `t4g.large` (8 GB) stop the API and
Neo4j for the duration, or resize the instance to 16 GB (`t4g.xlarge`) temporarily. The source database is not needed afterwards.

On this machine, once:
```bash
cd /path/to/Drugcentral-RAG-EC2      # this repository, on the machine with the source database
docker save drugcentral-source-db:rdkit-2026.03 | gzip > ~/source-bundle/source-db-image.tar.gz      # mkdir -p ~/source-bundle first
cp data/source/drugcen-2027-image.dump ~/source-bundle/source.dump && (cd ~/source-bundle && sha256sum source.dump > SHA256SUMS)
rsync -avP ~/source-bundle/ ec2-user@HOST:/srv/drugcentral/Drugcentral-RAG-EC2/source-bundle/
rsync -avP --exclude bundle/ --exclude data/ --exclude '.env' --exclude benchmarks/ \
      ./ ec2-user@HOST:/srv/drugcentral/Drugcentral-RAG-EC2/            # the new code
```
On the host (the code is already there from the first deployment):
```bash
cd /srv/drugcentral/Drugcentral-RAG-EC2
gunzip -c source-bundle/source-db-image.tar.gz | docker load
printf 'SOURCE_DB_IMAGE=drugcentral-source-db:rdkit-2026.03\nSOURCE_DB_PLATFORM=linux/arm64\nSOURCE_DB_PASSWORD=%s\n' "$(openssl rand -hex 24)" >> environments/ec2/.env
C="docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env"
$C stop api neo4j                                                     # free memory for the build (skip on a 16 GB instance)
$C build api                                                          # rebuild the tools image with the new code
environments/ec2/restore_source.sh source-bundle/source.dump          # expect: RDKit 4.8.0, 70 tables, 4 views, 5116 structures
mkdir -p benchmarks && environments/ec2/build_full_coverage.sh        # previews and the cost estimate only, nothing is written or spent
environments/ec2/build_full_coverage.sh --apply                       # publishes, then asks you to type YES before any Bedrock spend
```
Run the last command inside `tmux` or with `nohup` (embedding takes about 3 hours and resumes if interrupted).
When it finishes, evaluate the new collection (step 4 above), then switch: `printf 'VECTOR_COLLECTION=...\nSTRUCTURED_COLLECTION=...\n' >
environments/ec2/collection.env`, `unset VECTOR_COLLECTION STRUCTURED_COLLECTION`, and recreate the API. Stop the source database afterwards
(`$C --profile source stop source-db`) and restart the API and Neo4j. The tools container uses the ingestion role; the API stays read-only.

On an x86 instance you can skip the image upload and `restore_source.sh`: set `SOURCE_DB_IMAGE=unmtransinfo/drugcen-2027-db:latest`,
`SOURCE_DB_PLATFORM=linux/amd64`, `SOURCE_DB_NAME=drugcentral` and start it with `$C --profile source up -d source-db`; it initialises itself on first start.

## Alternative: x86 instance, pull the source image from Docker Hub, transfer only code

The published `unmtransinfo/drugcen-2027-db` image is **amd64-only and 18.3 GB**, so it cannot run on an ARM (`t4g`) instance. To build with
nothing but code transferred from this machine, use an x86 instance. It builds the whole new collection from the source and then serves it,
so it does not need the older bundle at all.

1. **Launch an x86 instance** (same region, key pair, security group and instance profile as before; metadata hop limit 2):
   `m6i.xlarge` or `t3.xlarge` (4 vCPU / 16 GB), Amazon Linux 2023 **x86_64** AMI, **100 GB gp3** (the image alone is 18 GB, the source database about 5 GB).
2. **Prepare the host** with the same Docker, compose and buildx steps as before, but with the x86 compose plugin and buildx binaries
   (`docker-compose-linux-x86_64`, `buildx-<version>.linux-amd64`).
3. **Transfer the code** from this machine (about 100 MB without data):
   ```bash
   rsync -avP --exclude bundle/ --exclude data/ --exclude benchmarks/ --exclude '.env' --exclude environments/ec2/.env \
         ./ ec2-user@X86_HOST:/srv/drugcentral/Drugcentral-RAG-EC2/
   ```
   The API image also needs `data/labels.sqlite` (2.6 GB, derived from the same source). Upload it once, or copy
   it between the instances inside AWS: `rsync -avP --rsync-path="sudo rsync" ec2-user@ARM_PRIVATE_IP:/srv/drugcentral/Drugcentral-RAG-EC2/data/ data/`
   from a session started with `ssh -A` (agent forwarding), so no key is stored on the instance.
4. **Configure and start**:
   ```bash
   cd /srv/drugcentral/Drugcentral-RAG-EC2 && mkdir -p data benchmarks
   cp environments/ec2/env.example environments/ec2/.env && chmod 600 environments/ec2/.env
   for k in POSTGRES_ADMIN_PASSWORD RAG_API_PASSWORD RAG_INGEST_PASSWORD API_AUTH_TOKEN NEO4J_PASSWORD SOURCE_DB_PASSWORD; do
     grep -q "^$k=" environments/ec2/.env && sed -i "s|^$k=.*|$k=$(openssl rand -hex 24)|" environments/ec2/.env || echo "$k=$(openssl rand -hex 24)" >> environments/ec2/.env; done
   printf 'SOURCE_DB_IMAGE=unmtransinfo/drugcen-2027-db:latest\nSOURCE_DB_PLATFORM=linux/amd64\nSOURCE_DB_NAME=drugcentral\n' >> environments/ec2/.env
   C="docker compose -f environments/ec2/docker-compose.yml --env-file environments/ec2/.env"
   $C up -d db && environments/ec2/init_rag_db.sh                       # empty drugcentral_rag with the two roles
   $C build api                                                         # also builds the tools image (about 10 minutes)
   $C --profile source up -d source-db                                  # pulls 18 GB, then restores the database on first start
   $C --profile source logs -f source-db                                # watch until "database system is ready to accept connections"
   ```
   The first start loads the bundled dump and rebuilds the RDKit columns, so it is slow (tens of minutes). Check it is complete:
   `$C --profile source exec -T source-db psql -U postgres -d drugcentral -Atc "select count(*) from structures"` should print 5116.
5. **Build, in `tmux`** (it needs the typed confirmation, and embedding takes about 3 hours):
   ```bash
   environments/ec2/build_full_coverage.sh            # previews and the cost estimate; nothing written or spent
   environments/ec2/build_full_coverage.sh --apply    # publish, then type YES to spend; embed, index, grant, build the graph
   ```
6. **Switch and serve**:
   ```bash
   $C --profile source stop source-db
   printf 'VECTOR_COLLECTION=drugcentral_full_cohere_v4_1024_YYYYMMDD\nSTRUCTURED_COLLECTION=drugcentral-sections-full-YYYYMMDD\n' > environments/ec2/collection.env
   sed -i 's/^GRAPH_ENABLED=.*/GRAPH_ENABLED=1/' environments/ec2/.env || echo GRAPH_ENABLED=1 >> environments/ec2/.env
   unset VECTOR_COLLECTION STRUCTURED_COLLECTION GRAPH_ENABLED
   $C --profile graph up -d --force-recreate api
   ```
   Use the actual dates the build script printed. Test with a graph question and a new-table question, then evaluate (step 4 above) before pointing users at it.
   If no IAM role is attached, add `-f environments/ec2/docker-compose.keys.yml` to `$C` and `COMPOSE_OVERRIDE=environments/ec2/docker-compose.keys.yml` before the build script.

The ARM instance can be stopped once the x86 one is verified; nothing on it is needed afterwards.

## Memory and speed

The collection roughly doubles (about 286k chunks, ~1.2 GB of vectors). Drug questions filter by drug first, entity questions by
section, using the partial indexes `ingestion/index_collection.py` creates. If the `t4g.large` (8 GB) feels tight with Neo4j running,
leave the graph off or move to a 16 GB instance before judging speed.
