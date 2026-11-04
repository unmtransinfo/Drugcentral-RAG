"""Build user-style questions for the sections added by ingestion/full_coverage.py.

The 300-question full-schema set asks for single rows by internal id ("record id=173309"), which a user would not know.
These questions are what a person would ask: by drug, target, disease or code, in plain words. Expected values come straight
from the source tables. Nothing here calls Bedrock. Output is UNREVIEWED: have someone check the questions and values.

For every question the builder also records which retrieval sections the routing picks, and flags questions the routing does
not send to the intended section, so phrasing gaps show up before any answer is generated.

Run: python -m evaluation.build_user_style_questions --output evaluation/user-style-new-tables
Env: SOURCE_DATABASE_URL (or SOURCE_PG*)
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from chatbot.question_retrieval import requested_sections

PER_TYPE = 6
ID_LABELS = {'CHEMBL_ID': 'ChEMBL ID', 'DRUGBANK_ID': 'DrugBank ID', 'PUBCHEM_CID': 'PubChem CID', 'UNII': 'UNII',
             'KEGG_DRUG': 'KEGG drug ID', 'CHEBI': 'ChEBI ID'}


def rows(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchall()


def pick(rng, items, n=PER_TYPE):
    items = list(items)
    rng.shuffle(items)
    return items[:n]


def q(section, table, question, facts, drug=None, kind='table'):
    facts = [str(f) for f in facts if f not in (None, '') and str(f).strip()]
    return {'section': section, 'source_table': table, 'source_object_type': kind, 'question': question,
            'reference_facts': facts, 'drug': drug, 'category': section}


def single(rows_, key=0):
    """Rows whose key (drug) occurs exactly once, so the expected answer is unambiguous."""
    counts = Counter(r[key] for r in rows_)
    return [r for r in rows_ if counts[r[key]] == 1]


def ddd(cur, rng):
    data = single(rows(cur, "SELECT s.name, d.ddd::text, d.unit_type FROM atc_ddd d JOIN structures s ON s.id = d.struct_id "
                            "WHERE d.ddd IS NOT NULL AND s.name ~ '^[a-z][a-z -]{3,30}$'"))
    return [q('ddd', 'atc_ddd', f'What is the defined daily dose (DDD) of {n}?', [v], n) for n, v, u in pick(rng, data)]


def identifiers(cur, rng):
    out = []
    for id_type, label in ID_LABELS.items():
        data = single(rows(cur, "SELECT s.name, i.identifier FROM identifier i JOIN structures s ON s.id = i.struct_id "
                                "WHERE i.id_type = %s AND s.name ~ '^[a-z][a-z -]{3,30}$'", (id_type,)))
        out += [q('identifiers', 'identifier', f'What is the {label} of {n}?', [v], n) for n, v in pick(rng, data, 1)]
    return out


def pk(cur, rng):
    data = rows(cur, "SELECT s.name, p.type, p.value::text, p.unit FROM pk_az p JOIN structures s ON s.id = p.struct_id "
                     "WHERE p.value IS NOT NULL AND s.name ~ '^[a-z][a-z -]{3,30}$'")
    return [q('pk_measurements', 'pk_az', f'What is the AstraZeneca {t} measurement of {n}?', [v], n) for n, t, v, u in pick(rng, data)]


def veterinary(cur, rng):
    data = single(rows(cur, "SELECT s.name, vp.trade_name FROM vetprod2struct vs JOIN vetprod vp ON vp.prodid = vs.prodid "
                            "JOIN structures s ON s.id = vs.struct_id WHERE s.name ~ '^[a-z][a-z -]{3,30}$'"))
    return [q('veterinary', 'vetprod', f'Which veterinary product contains {n}?', [t], n) for n, t in pick(rng, data)]


def faers_subgroups(cur, rng):
    out = []
    for table, section, words in [('faers_male', 'adverse_events_male', 'male patients'), ('faers_female', 'adverse_events_female', 'female patients'),
                                  ('faers_ped', 'adverse_events_pediatric', 'pediatric patients'),
                                  ('faers_ger', 'adverse_events_geriatric', 'geriatric patients')]:
        data = rows(cur, f"SELECT DISTINCT ON (f.struct_id) s.name, f.meddra_name FROM {table} f JOIN structures s ON s.id = f.struct_id "
                         f"WHERE s.name ~ '^[a-z][a-z -]{{3,30}}$' ORDER BY f.struct_id, f.llr DESC NULLS LAST, f.meddra_name")
        out += [q(section, table, f'What is the highest-LLR FAERS adverse event reported for {n} in {words}?', [e], n)
                for n, e in pick(rng, data, 4)]
    return out


def patents(cur, rng):
    data = single(rows(cur, "SELECT DISTINCT s.name, v.applicant FROM ob_patent_view v JOIN structures s ON s.id = v.struct_id "
                            "WHERE v.applicant IS NOT NULL AND s.name ~ '^[a-z][a-z -]{3,30}$'"))
    return [q('patent_details', 'ob_patent_view', f'Which applicant holds the Orange Book patents for {n}?', [a], n, 'view') for n, a in pick(rng, data)]


def doid_mapping(cur, rng):
    data = single(rows(cur, "SELECT s.name, split_part(v.doid, ',', 1) FROM omop_relationship_doid_view v JOIN structures s ON s.id = v.struct_id "
                            "WHERE v.doid IS NOT NULL AND v.relationship_name = 'indication' AND s.name ~ '^[a-z][a-z -]{3,30}$'"))
    return [q('doid_mapping', 'omop_relationship_doid_view', f'What is the Disease Ontology (DOID) mapping of the indication of {n}?', [d], n, 'view')
            for n, d in pick(rng, data)]


def lincs(cur, rng):
    ids = rows(cur, "SELECT DISTINCT struct_id1 FROM lincs_signature TABLESAMPLE SYSTEM (0.02)")
    chosen = [r[0] for r in pick(rng, ids, 40)]
    data = rows(cur, "SELECT DISTINCT ON (l.struct_id1) a.name, b.name FROM lincs_signature l JOIN structures a ON a.id = l.struct_id1 "
                     "JOIN structures b ON b.id = l.struct_id2 WHERE l.struct_id1 = ANY(%s) AND l.pearson IS NOT NULL "
                     "AND a.name ~ '^[a-z][a-z -]{3,30}$' ORDER BY l.struct_id1, l.pearson DESC", (chosen,))
    return [q('lincs', 'lincs_signature', f'Which compound has the most similar LINCS signature to {a}?', [b], a) for a, b in pick(rng, data)]


def targets(cur, rng):
    human = rows(cur, "SELECT t.id, t.name, c.gene, c.accession, t.target_class FROM target_dictionary t JOIN td2tc x ON x.target_id = t.id "
                      "JOIN target_component c ON c.id = x.component_id WHERE t.protein_components = 1 AND c.organism = 'Homo sapiens' "
                      "AND c.gene ~ '^[A-Z0-9]{3,8}$' AND c.accession IS NOT NULL")
    out = [q('target_profile', 'target_component', f'What is the UniProt accession of the {g} target?', [a]) for _i, _n, g, a, _c in pick(rng, human)]
    out += [q('target_profile', 'target_dictionary', f'What is the target class of {n}?', [c]) for _i, n, _g, _a, c in pick(rng, human) if c]
    for target_id, name, gene, _a, _c in pick(rng, human):
        go = [r[0] for r in rows(cur, "SELECT DISTINCT g.id FROM td2tc x JOIN tdgo2tc t ON t.component_id = x.component_id JOIN target_go g ON g.id = t.go_id "
                                      "WHERE x.target_id = %s ORDER BY g.id LIMIT 6", (target_id,))]
        if go:
            out.append(q('target_profile', 'tdgo2tc', f'What GO terms are recorded for the {gene} target?', go))
    return out


def ontology(cur, rng):
    out = [q('ontology', 'target_go', f'What is GO term {i}?', [t]) for i, t in pick(rng, rows(cur, "SELECT id, term FROM target_go"))]
    out += [q('ontology', 'target_keyword', f'What is the category of the target keyword {k}?', [c])
            for k, c in pick(rng, rows(cur, "SELECT keyword, category FROM target_keyword WHERE category IS NOT NULL"))]
    return out


def diseases(cur, rng):
    data = rows(cur, "SELECT label, doid FROM doid WHERE label ~ '^[a-z][a-z -]{4,40}$'")
    return [q('disease_profile', 'doid', f'What is the Disease Ontology ID of {label}?', [d]) for label, d in pick(rng, data)]


def references(cur, rng):
    data = rows(cur, "SELECT title, pmid::text FROM reference WHERE pmid IS NOT NULL AND length(title) BETWEEN 20 AND 120")
    return [q('references', 'reference', f'What is the PMID of the reference titled "{t}"?', [p]) for t, p in pick(rng, data)]


def dictionaries(cur, rng):
    out = [q('dictionary', 'ob_patent_use_code', f'What does the Orange Book patent use code {c} mean?', [d])
           for c, d in pick(rng, rows(cur, "SELECT code, description FROM ob_patent_use_code WHERE length(description) < 80"))]
    out += [q('dictionary', 'ob_exclusivity_code', f'What does the exclusivity code {c} mean?', [d])
            for c, d in pick(rng, rows(cur, "SELECT code, description FROM ob_exclusivity_code WHERE length(description) < 80"), 4)]
    out += [q('dictionary', 'data_source', f'Which data source has id {i}?', [n]) for i, n in pick(rng, rows(cur, "SELECT src_id, source_name FROM data_source"), 4)]
    out += [q('dictionary', 'action_type', f'What is the definition of the action type {a}?', [' '.join(d.split()[:5])])
            for a, d in pick(rng, rows(cur, "SELECT action_type, description FROM action_type WHERE description IS NOT NULL"), 4)]
    return out


GENERATORS = [ddd, identifiers, pk, veterinary, faers_subgroups, patents, doid_mapping, lincs, targets, ontology, diseases, references, dictionaries]


def routing_audit(question):
    found = requested_sections(question['question'])
    return {'routed_sections': found, 'routed_to_intended': question['section'] in found}


def build(conn, seed=20261008):
    rng = random.Random(seed)
    questions = []
    with conn.cursor() as cur:
        for generator in GENERATORS:
            questions += generator(cur, rng)
    seen, unique = set(), []
    for item in questions:
        if item['question'] not in seen and item['reference_facts']:
            seen.add(item['question'])
            unique.append(item)
    for n, item in enumerate(sorted(unique, key=lambda i: (i['section'], i['question'])), 1):
        item['id'] = f'us{n:03d}'
        item.update(routing_audit(item))
    return sorted(unique, key=lambda i: i['id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path('evaluation/user-style-new-tables'))
    parser.add_argument('--seed', type=int, default=20261008)
    args = parser.parse_args()
    from chatbot.database_config import connect_database
    conn = connect_database(source=True)
    try:
        with conn:
            questions = build(conn, args.seed)
    finally:
        conn.close()
    args.output.mkdir(parents=True, exist_ok=False)
    body = json.dumps(questions, indent=2, ensure_ascii=False)
    (args.output / 'questions.json').write_text(body)
    misrouted = [item for item in questions if not item['routed_to_intended']]
    manifest = {'questions': len(questions), 'by_section': dict(Counter(i['section'] for i in questions)), 'seed': args.seed,
                'questions_sha256': hashlib.sha256(body.encode()).hexdigest(),
                'routing_misses': {s: sum(1 for i in misrouted if i['section'] == s) for s in sorted({i['section'] for i in misrouted})},
                'status': 'UNREVIEWED: expected values come from source tables; a person should check them before scores are quoted'}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
