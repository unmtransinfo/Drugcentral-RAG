import os
import psycopg2
import pandas as pd
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from chatbot.storage import backend_name, store_documents
from chatbot.embedding_config import create_embeddings, embedding_model_id
from chatbot.retrieval import section_chunks
from datetime import datetime, timezone
import json

# Embedding model must match the one used at query time in agent.py
load_dotenv()

# Source connection is independent from the destination RAG database.
from chatbot.database_config import connect_database
conn = connect_database(source=True)

print("Running queries against DrugCentral database...")

# Query 1: Base structures (with INN Stem descriptions)
structures_df = pd.read_sql("""
    SELECT s.id AS struct_id, s.name AS drug_name,
           s.cd_formula, s.cd_molweight, s.clogp, s.tpsa, s.lipinski, s.cas_reg_no,
           stem.stem AS inn_stem, stem.definition AS inn_stem_definition
    FROM structures s
    LEFT JOIN inn_stem stem ON s.stem = stem.stem
""", conn)
print(f"  [1/14] Loaded {len(structures_df)} structures")

# Query 2: Synonyms & Commercial Product Brand Names (Union)
synonyms_df = pd.read_sql("""
    SELECT struct_id, STRING_AGG(DISTINCT name, ' || ') AS synonyms,
           STRING_AGG(DISTINCT name, ' || ') FILTER (WHERE kind = 'ingredient') AS ingredient_synonyms,
           STRING_AGG(DISTINCT name, ' || ') FILTER (WHERE kind = 'product') AS product_names
    FROM (
        SELECT id AS struct_id, name, 'ingredient' AS kind FROM synonyms
        UNION
        SELECT ai.struct_id, p.product_name AS name, 'product' AS kind
        FROM product p
        JOIN active_ingredient ai ON p.ndc_product_code = ai.ndc_product_code
        WHERE ai.struct_id IS NOT NULL
    ) unioned_synonyms
    GROUP BY struct_id
""", conn)
print(f"  [2/14] Loaded combined synonyms & product names for {len(synonyms_df)} drugs")

# Query 3: Indications
indications_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(DISTINCT concept_name, ' || ') FILTER (WHERE relationship_name = 'indication') AS indications,
           STRING_AGG(DISTINCT concept_name, ' || ') FILTER (WHERE relationship_name = 'off-label use') AS off_label,
           STRING_AGG(DISTINCT concept_name, ' || ') FILTER (WHERE relationship_name = 'symptomatic treatment') AS symptomatic,
           STRING_AGG(DISTINCT concept_name, ' || ') FILTER (WHERE relationship_name = 'reduce risk') AS risk_reduction
    FROM omop_relationship
    WHERE relationship_name IN ('indication', 'off-label use', 'symptomatic treatment', 'reduce risk')
    GROUP BY struct_id
""", conn)
print(f"  [3/14] Loaded indications for {len(indications_df)} drugs")

# Query 4: Contraindications
contra_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(DISTINCT concept_name, ' || ') AS contraindications
    FROM omop_relationship
    WHERE relationship_name = 'contraindication'
    GROUP BY struct_id
""", conn)
print(f"  [4/14] Loaded contraindications for {len(contra_df)} drugs")

# Query 5: Biological Targets and Activity
targets_df = pd.read_sql("""
    SELECT ai.struct_id,
           STRING_AGG(DISTINCT CASE WHEN t.name IS NOT NULL THEN CONCAT(t.name, ' (', ai.gene, ')') ELSE NULL END, ' || ') AS targets,
           ROUND(AVG(ai.act_value)::numeric, 2) AS avg_activity_value,
           STRING_AGG(DISTINCT ai.action_type, ' || ') AS action_types,
           STRING_AGG(DISTINCT CASE WHEN ai.moa = 1 THEN CONCAT(t.name, ' (', ai.gene, ')') ELSE NULL END, ' || ') AS moa_targets
    FROM act_table_full ai
    LEFT JOIN target_dictionary t ON ai.target_id = t.id
    GROUP BY ai.struct_id
""", conn)
print(f"  [5/14] Loaded targets for {len(targets_df)} drugs")

# Query 6: Regulatory Approvals
approvals_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(DISTINCT CONCAT(type, ' (', COALESCE(TO_CHAR(approval, 'YYYY'), 'N/A'), ')'), ' || ') AS approvals,
           MIN(approval) AS first_approval_date
    FROM approval
    GROUP BY struct_id
""", conn)
print(f"  [6/14] Loaded approvals for {len(approvals_df)} drugs")

# Query 7: Pharmacological Classifications
pharma_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(DISTINCT CASE WHEN type = 'MoA' THEN name END, ' || ') AS mechanism_of_action,
           STRING_AGG(DISTINCT CASE WHEN type = 'EPC' THEN name END, ' || ') AS pharmacologic_class,
           STRING_AGG(DISTINCT CASE WHEN type = 'PA'  THEN name END, ' || ') AS pharmacologic_action,
           STRING_AGG(DISTINCT CASE WHEN type = 'CS'  THEN name END, ' || ') AS chemical_structure_class
    FROM pharma_class
    GROUP BY struct_id
""", conn)
print(f"  [7/14] Loaded pharma classes for {len(pharma_df)} drugs")

# Query 8: Drug-Drug Interactions
ddi_df = pd.read_sql("""
    SELECT s2d.struct_id,
           STRING_AGG(DISTINCT
               CONCAT(
                   CASE WHEN d.drug_class1 = dc.name THEN d.drug_class2 ELSE d.drug_class1 END,
                   ' [', d.ddi_risk, ']: ',
                   COALESCE(d.description, 'No description')
               ), ' || '
           ) AS drug_interactions
    FROM struct2drgclass s2d
    JOIN drug_class dc ON s2d.drug_class_id = dc.id
    JOIN ddi d ON dc.name = d.drug_class1 OR dc.name = d.drug_class2
    GROUP BY s2d.struct_id
""", conn)
print(f"  [8/14] Loaded drug interactions for {len(ddi_df)} drugs")

# Query 9: pKa Values
pka_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(DISTINCT CONCAT(pka_type, 'pKa: ', value, ' (level: ', COALESCE(pka_level, 'N/A'), ')'), ' || ') AS pka_values
    FROM pka
    GROUP BY struct_id
""", conn)
print(f"  [9/14] Loaded pKa values for {len(pka_df)} drugs")

# Query 10: ADME / Pharmacokinetics Properties
properties_df = pd.read_sql("""
    SELECT p.struct_id,
           STRING_AGG(DISTINCT CONCAT(pt.name, ' (', pt.symbol, '): ', p.value, ' ', COALESCE(pt.units, '')), ' || ') AS adme_properties
    FROM property p
    JOIN property_type pt ON p.property_type_id = pt.id
    GROUP BY p.struct_id
""", conn)
print(f"  [10/14] Loaded ADME properties for {len(properties_df)} drugs")

# Query 11: ATC Classifications
atc_df = pd.read_sql("""
    SELECT s2a.struct_id,
           STRING_AGG(DISTINCT CONCAT(a.code, ' - ', a.chemical_substance, ' [L1: ', a.l1_name, ', L2: ', a.l2_name, ', L3: ', a.l3_name, ', L4: ', a.l4_name, ']'), ' || ') AS atc_classifications
    FROM struct2atc s2a
    JOIN atc a ON s2a.atc_code = a.code
    GROUP BY s2a.struct_id
""", conn)
print(f"  [11/14] Loaded ATC codes for {len(atc_df)} drugs")

# Query 12: FDA Patents & Exclusivities
patents_df = pd.read_sql("""
    SELECT s2o.struct_id,
           STRING_AGG(DISTINCT CONCAT('Patent ', p.patent_no, ' (expires ', TO_CHAR(p.patent_expire_date, 'YYYY-MM-DD'), ')'), ' || ') AS patents,
           STRING_AGG(DISTINCT CONCAT('Exclusivity ', e.exclusivity_code, ' (expires ', TO_CHAR(e.exclusivity_date, 'YYYY-MM-DD'), ')'), ' || ') AS exclusivities
    FROM struct2obprod s2o
    JOIN ob_product op ON s2o.prod_id = op.id
    LEFT JOIN ob_patent p ON op.appl_no = p.appl_no AND op.product_no = p.product_no
    LEFT JOIN ob_exclusivity e ON op.appl_no = e.appl_no AND op.product_no = e.product_no
    GROUP BY s2o.struct_id
""", conn)
print(f"  [12/14] Loaded Orange Book patents and exclusivities for {len(patents_df)} drugs")

# Query 13: Protein Data Bank (PDB) Codes
pdb_df = pd.read_sql("""
    SELECT struct_id, STRING_AGG(DISTINCT pdb, ' || ') AS pdb_codes
    FROM pdb
    GROUP BY struct_id
""", conn)
print(f"  [13/14] Loaded PDB co-crystallized structures for {len(pdb_df)} drugs")

# Query 14: FAERS Statistics (All adverse events)
faers_df = pd.read_sql("""
    SELECT struct_id,
           STRING_AGG(CONCAT(meddra_name, ' (LLR: ', ROUND(llr::numeric, 2), ', ', drug_ae, ' cases)'), ' || ') AS top_adverse_events
    FROM faers
    GROUP BY struct_id
""", conn)
print(f"  [14/14] Loaded clinical FAERS adverse event stats for {len(faers_df)} drugs")

conn.close()

# Merge all data on struct_id
print("\nMerging all table data into unified drug dossiers...")
df = structures_df.copy()
df = df.merge(synonyms_df, on='struct_id', how='left')
df = df.merge(indications_df, on='struct_id', how='left')
df = df.merge(contra_df, on='struct_id', how='left')
df = df.merge(targets_df, on='struct_id', how='left')
df = df.merge(approvals_df, on='struct_id', how='left')
df = df.merge(pharma_df, on='struct_id', how='left')
df = df.merge(ddi_df, on='struct_id', how='left')
df = df.merge(pka_df, on='struct_id', how='left')
df = df.merge(properties_df, on='struct_id', how='left')
df = df.merge(atc_df, on='struct_id', how='left')
df = df.merge(patents_df, on='struct_id', how='left')
df = df.merge(pdb_df, on='struct_id', how='left')
df = df.merge(faers_df, on='struct_id', how='left')

print(f"Merged {len(df)} unique drug dossiers")

# Newline Bullet Formatters (No Truncation / Zero Data Loss)
def format_list_to_bullets(val, separator=' || '):
    if not val or pd.isna(val) or val == 'None documented' or val == 'N/A':
        return '  - None documented'
    items = val.split(separator)
    return '\n'.join(f"  - {item.strip()}" for item in items if item.strip())

def safe_val(val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return 'N/A'
    return val

dataset_version = os.getenv("DATASET_VERSION", datetime.now(timezone.utc).isoformat())
documents = []
for _, row in df.iterrows():
    synonyms_text = format_list_to_bullets(row.get('ingredient_synonyms'), ' || ')
    products_text = format_list_to_bullets(row.get('product_names'), ' || ')
    indications = format_list_to_bullets(row.get('indications'), ' || ')
    contraindications = format_list_to_bullets(row.get('contraindications'), ' || ')
    targets = format_list_to_bullets(row.get('targets'), ' || ')
    moa_targets = format_list_to_bullets(row.get('moa_targets'), ' || ')
    action_types = format_list_to_bullets(row.get('action_types'), ' || ')
    moa_text = format_list_to_bullets(row.get('mechanism_of_action'), ' || ')
    epc_text = format_list_to_bullets(row.get('pharmacologic_class'), ' || ')
    pa_text = format_list_to_bullets(row.get('pharmacologic_action'), ' || ')
    cs_text = format_list_to_bullets(row.get('chemical_structure_class'), ' || ')
    approvals_text = format_list_to_bullets(row.get('approvals'), ' || ')
    ddi_text = format_list_to_bullets(row.get('drug_interactions'), ' || ')
    faers_text = format_list_to_bullets(row.get('top_adverse_events'), ' || ')
    patents_text = format_list_to_bullets(row.get('patents'), ' || ')
    exclusivities_text = format_list_to_bullets(row.get('exclusivities'), ' || ')
    atc_text = format_list_to_bullets(row.get('atc_classifications'), ' || ')
    pka_text = format_list_to_bullets(row.get('pka_values'), ' || ')
    adme_text = format_list_to_bullets(row.get('adme_properties'), ' || ')
    pdb_text = format_list_to_bullets(row.get('pdb_codes'), ' || ')

    # Construct rich natural-language text dossier with 100% of data (no truncation)
    text = (
        f"Drug Profile: {row['drug_name']}.\n"
        f"Also known as:\n{synonyms_text}\n"
        f"Commercial product names:\n{products_text}\n"
        f"INN Stem: {safe_val(row.get('inn_stem'))} ({safe_val(row.get('inn_stem_definition'))}).\n"
        f"Molecular Formula: {safe_val(row.get('cd_formula'))}. "
        f"Molecular Weight: {safe_val(row.get('cd_molweight'))}. "
        f"cLogP: {safe_val(row.get('clogp'))}. "
        f"TPSA: {safe_val(row.get('tpsa'))}. "
        f"Lipinski Violations: {safe_val(row.get('lipinski'))}. "
        f"CAS: {safe_val(row.get('cas_reg_no'))}.\n"
        f"ATC Classifications:\n{atc_text}\n"
        f"pKa Values:\n{pka_text}\n"
        f"ADME / Pharmacokinetics Properties:\n{adme_text}\n"
        f"Indications:\n{indications}\n"
        f"Off-label Uses:\n{format_list_to_bullets(row.get('off_label'))}\n"
        f"Symptomatic Treatment:\n{format_list_to_bullets(row.get('symptomatic'))}\n"
        f"Risk Reduction:\n{format_list_to_bullets(row.get('risk_reduction'))}\n"
        f"Contraindications:\n{contraindications}\n"
        f"Biological Targets:\n{targets}\n"
        f"Mechanism of Action Targets:\n{moa_targets}\n"
        f"Action Types:\n{action_types}\n"
        f"Average Activity Value: {safe_val(row.get('avg_activity_value'))}.\n"
        f"Pharmacologic Class (EPC):\n{epc_text}\n"
        f"Mechanism of Action (MoA):\n{moa_text}\n"
        f"Pharmacologic Action:\n{pa_text}\n"
        f"Chemical Structure Class:\n{cs_text}\n"
        f"Regulatory Approvals:\n{approvals_text}\n"
        f"First Approval Date: {safe_val(row.get('first_approval_date'))}.\n"
        f"FDA Patents:\n{patents_text}\n"
        f"FDA Exclusivities:\n{exclusivities_text}\n"
        f"Protein Data Bank (PDB) Co-crystallized Codes:\n{pdb_text}\n"
        f"Clinical Adverse Events (FAERS Statistics):\n{faers_text}\n"
        f"Drug-Drug Interactions:\n{ddi_text}\n"
    )

    # Extract lowercased brand synonyms for high-precision retrieval
    syn_set = set()
    if row.get('synonyms') and not pd.isna(row['synonyms']):
        for s in row['synonyms'].split(' || '):
            s_clean = s.strip().lower()
            if s_clean:
                syn_set.add(s_clean)


    synonyms_lower = ", ".join(sorted(list(syn_set)))

    # Store drug name AND synonyms as metadata for the MetadataAwareRetriever
    metadata = {
        "source": "drugcentral",
        "schema_version": 2,
        "dataset_version": dataset_version,
        "synonyms_json": json.dumps(sorted(syn_set)),
        "ingredient_synonyms_json": json.dumps(sorted({n.strip().lower() for n in str(safe_val(row.get('ingredient_synonyms'))).split(' || ') if n.strip() and n != 'N/A'})),
        "product_names_json": json.dumps(sorted({n.strip().lower() for n in str(safe_val(row.get('product_names'))).split(' || ') if n.strip() and n != 'N/A'})),
        "drug_name": str(row['drug_name']).lower(),
        "synonyms": synonyms_lower,
        "struct_id": int(row['struct_id']),
    }

    # Split the dossier into zero-loss chunks and build LangChain Documents
    for section, header, sequence, chunk in section_chunks(text):
        documents.append(Document(page_content=chunk, metadata={
            **metadata, "section": section, "section_header": header, "sequence": sequence,
        }))

embeddings = create_embeddings()
EMBEDDING_MODEL_ID = embedding_model_id()
batch_size = 100
print(f"\nPrepared {len(documents)} section chunks for {backend_name()}.")

if os.getenv("INGEST_PREVIEW_ONLY") == "1":
    from pathlib import Path
    output = Path(os.getenv("INGEST_PREVIEW_PATH", "benchmarks/ingestion-preview.jsonl"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for doc in documents:
            handle.write(json.dumps({"text": doc.page_content, "metadata": doc.metadata}) + "\n")
    print(f"Preview written to {output}; no embeddings or storage writes performed.")
else:
    store_documents(documents, embeddings, EMBEDDING_MODEL_ID, batch_size)

print("\nIngestion preparation complete." if os.getenv("INGEST_PREVIEW_ONLY") == "1" else "\nVector database build committed successfully.")
print(f"Total documents prepared: {len(documents)}")
