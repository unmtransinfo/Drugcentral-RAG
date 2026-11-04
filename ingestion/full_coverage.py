"""Evidence chunks for every DrugCentral source table and view that the base dossier does not cover.

The base ingestion (ingestion/vectors.py) turns 24 tables into per-drug dossier sections. This module adds the
rest so the whole database can be answered from retrieval:

* drug-linked tables are grouped per drug into NEW sections (never mixed into existing ones, so existing answers
  are unchanged) and keep every row;
* targets, diseases, references, ontology terms, keywords and code tables become entity documents that repeat
  their own title in every chunk, because they cannot be filtered by drug;
* the four views are built from their definitions;
* lincs_signature (8.7M pairwise similarity rows) is summarised per drug, not embedded row by row.

Every source object is listed in COVERAGE; a test fails if a table or view is missing from it.

Run (after `INGEST_PREVIEW_ONLY=1 python -m ingestion.vectors` for the base chunks, with the same DATASET_VERSION):
    python -m ingestion.full_coverage preview --output benchmarks/extra-preview.jsonl      # no Bedrock calls, prints cost
    python -m ingestion.full_coverage publish --base benchmarks/base-preview.jsonl \
        --extra benchmarks/extra-preview.jsonl --collection drugcentral-sections-full-YYYYMMDD
"""
import argparse
import re
from collections import defaultdict, Counter
import json
import os
from pathlib import Path

MAX_CHARS = 3000
EMBED_USD_PER_MILLION = float(os.getenv('COST_EMBED_INPUT_PER_MILLION', '0.12'))
VIEWS = ['faers_top', 'ob_exclusivity_view', 'ob_patent_view', 'omop_relationship_doid_view']
GENE_SPLIT = re.compile(r'[|;,]')
GO_ASPECT = {'F': 'molecular function', 'P': 'biological process', 'C': 'cellular component'}

# Header (as written in chunk text) -> section. Mirrored into chatbot.retrieval.SECTION_HEADERS; a test keeps them equal.
DRUG_HEADERS = {
    'Defined Daily Doses (ATC DDD)': 'ddd',
    'Chemical Identifiers (all identifier types)': 'identifiers',
    'Inxight Drugs Cross-reference': 'identifiers',
    'Structure Types': 'structure_info',
    'Parent Compound': 'structure_info',
    'AstraZeneca Pharmacokinetic Measurements': 'pk_measurements',
    'Human and Animal Use': 'veterinary',
    'Veterinary Indications and Contraindications': 'veterinary',
    'Veterinary Products': 'veterinary',
    'FAERS Reports in Male Patients': 'adverse_events_male',
    'FAERS Reports in Female Patients': 'adverse_events_female',
    'FAERS Reports in Pediatric Patients': 'adverse_events_pediatric',
    'FAERS Reports in Geriatric Patients': 'adverse_events_geriatric',
    'Highest-ranked FAERS Signals': 'adverse_events_top',
    'Orange Book Patents with Use Descriptions': 'patent_details',
    'Orange Book Exclusivities with Descriptions': 'patent_details',
    'SNOMED and Disease Ontology Mappings': 'doid_mapping',
    'LINCS Signature Similarity': 'lincs',
}
ENTITY_HEADERS = {
    'Target Profile': 'target_profile', 'Target Details': 'target_profile', 'Protein Components': 'target_profile',
    'Gene Ontology Terms': 'target_profile', 'Target Keywords': 'target_profile', 'Drugs with Recorded Activity': 'target_profile',
    'Disease Profile': 'disease_profile', 'Disease Details': 'disease_profile', 'Cross-references': 'disease_profile',
    'Drugs Mapped to this Disease': 'disease_profile',
    'Gene Ontology Term': 'ontology', 'Target Keyword': 'ontology', 'Term Details': 'ontology',
    'Reference': 'references', 'Citation': 'references',
    'Dictionary Code': 'dictionary', 'Code Details': 'dictionary', 'Dictionary Table': 'dictionary', 'Rows': 'dictionary',
}
SECTION_HEADERS_EXTRA = {**DRUG_HEADERS, **ENTITY_HEADERS}

# Every source object and how it is covered. 'base' = existing dossier ingestion.
COVERAGE = {
    # base dossier (ingestion/vectors.py)
    **{t: 'base' for t in ['act_table_full', 'active_ingredient', 'approval', 'atc', 'ddi', 'drug_class', 'faers', 'inn_stem',
                           'ob_exclusivity', 'ob_patent', 'ob_product', 'omop_relationship', 'pdb', 'pharma_class', 'pka',
                           'product', 'property', 'property_type', 'struct2atc', 'struct2drgclass', 'struct2obprod',
                           'structures', 'synonyms', 'target_dictionary']},
    # label database (data/labels.sqlite) is searched separately; these rows are also covered there
    **{t: 'base+labels' for t in ['label', 'section', 'prd2label']},
    # drug-linked, grouped per drug
    'atc_ddd': 'drug:ddd', 'identifier': 'drug:identifiers', 'inxight_drugs': 'drug:identifiers',
    'structure_type': 'drug:structure_info', 'struct2parent': 'drug:structure_info', 'parentmol': 'drug:structure_info',
    'pk_az': 'drug:pk_measurements', 'humanim': 'drug:veterinary', 'vetomop': 'drug:veterinary',
    'vetprod': 'drug:veterinary', 'vetprod2struct': 'drug:veterinary', 'vetprod_type': 'drug:veterinary',
    'vettype': 'drug:veterinary',
    'faers_male': 'drug:adverse_events_male', 'faers_female': 'drug:adverse_events_female',
    'faers_ped': 'drug:adverse_events_pediatric', 'faers_ger': 'drug:adverse_events_geriatric',
    'lincs_signature': 'drug:lincs (per-drug summary; pairwise rows are not embedded)',
    # entity documents
    'target_component': 'entity:target_profile', 'td2tc': 'entity:target_profile', 'tdgo2tc': 'entity:target_profile',
    'tdkey2tc': 'entity:target_profile', 'target_class': 'entity:dictionary',
    'target_go': 'entity:ontology', 'target_keyword': 'entity:ontology',
    'doid': 'entity:disease_profile', 'doid_xref': 'entity:disease_profile',
    'reference': 'entity:references',
    'ob_patent_use_code': 'entity:dictionary', 'ob_exclusivity_code': 'entity:dictionary',
    **{t: 'entity:dictionary' for t in ['action_type', 'approval_type', 'attr_type', 'data_source', 'dbversion', 'ddi_risk',
                                        'id_type', 'protein_type', 'ref_type', 'struct_type_def', 'rdkit_build_meta']},
    'ijc_connect_items': 'empty table (0 rows)', 'ijc_connect_structures': 'empty table (0 rows)',
    # views
    'faers_top': 'drug:adverse_events_top (view)', 'ob_exclusivity_view': 'drug:patent_details (view)',
    'ob_patent_view': 'drug:patent_details (view)', 'omop_relationship_doid_view': 'drug:doid_mapping (view)',
}
# Columns deliberately not embedded: binary or connection-table payloads with no useful text form.
EXCLUDED_COLUMNS = {'parentmol': ['molfile', 'molimg'], 'structures': ['molfile', 'molimg', 'm', 'mfp2', 'fp', 'torsionbv', 'ffp']}


def num(value):
    if value is None:
        return None
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value)


def join(*parts, sep='; '):
    return sep.join(str(p) for p in parts if p not in (None, ''))


def pack(header, lines, max_chars=MAX_CHARS, prefix=()):
    """Split a section into chunks without ever splitting a line; every chunk repeats its header (and legend)."""
    base = [*prefix]
    size0 = len(header) + 2 + sum(len(p) + 1 for p in base)
    current, size, sequence = list(base), size0, 0
    for line in lines:
        if len(current) > len(base) and size + len(line) + 1 > max_chars:
            yield sequence, header + ':\n' + '\n'.join(current)
            current, size, sequence = list(base), size0, sequence + 1
        current.append(line)
        size += len(line) + 1
    if len(current) > len(base):
        yield sequence, header + ':\n' + '\n'.join(current)


def bullet(text):
    return '  - ' + ' '.join(str(text).split())


def drug_chunks(struct_id, name, header, lines, version, tables):
    section = DRUG_HEADERS[header]
    legend = [bullet(LEGENDS[header])] if header in LEGENDS else []
    for sequence, text in pack(header, [bullet(l) for l in lines], prefix=legend):
        yield {'text': text, 'metadata': {
            'source': 'drugcentral', 'schema_version': 2, 'dataset_version': version, 'drug_name': name.lower(),
            'struct_id': int(struct_id), 'section': section, 'section_header': header, 'sequence': sequence,
            'source_tables': tables, 'coverage': 'full-database'}}


def entity_chunks(entity_type, entity_id, title_header, name, blocks, version, tables, url=None):
    """Chunks for one entity. The title line is repeated so each chunk is self-identifying without a drug filter."""
    section = ENTITY_HEADERS[title_header]
    sequence = 0
    for subheader, lines in blocks:
        if not lines:
            continue
        for _, body in pack(subheader, [bullet(l) for l in lines]):
            meta = {'source': 'drugcentral', 'schema_version': 2, 'dataset_version': version, 'section': section,
                    'section_header': subheader, 'sequence': sequence, 'entity_type': entity_type,
                    'entity_id': str(entity_id), 'entity_name': name, 'source_tables': tables, 'coverage': 'full-database'}
            if url:
                meta['source_url'] = url
            yield {'text': f'{title_header}: {" ".join(str(name).split())}\n{body}', 'metadata': meta}
            sequence += 1


def stream(conn, sql, name):
    with conn.cursor(name=name) as cur:
        cur.itersize = 20000
        cur.execute(sql)
        yield from cur


def grouped(conn, sql, name):
    """Yield (key, [rest-of-row...]) for rows ordered by their first column."""
    key, rows = None, []
    for row in stream(conn, sql, name):
        if rows and row[0] != key:
            yield key, rows
            rows = []
        key = row[0]
        rows.append(row[1:])
    if rows:
        yield key, rows


# Compact but lossless: the counts keep a fixed order explained once per chunk by FAERS_LEGEND.
FAERS_LINE = ("meddra_name || ' (MedDRA ' || coalesce(meddra_code::text,'?') || ', ' || coalesce(level,'?') || '): LLR ' || "
              "coalesce(round(llr::numeric,3)::text,'n/a') || ' (threshold ' || coalesce(round(llr_threshold::numeric,3)::text,'n/a') || "
              "'); counts ' || coalesce(drug_ae::text,'?') || '/' || coalesce(drug_no_ae::text,'?') || '/' || "
              "coalesce(no_drug_ae::text,'?') || '/' || coalesce(no_drug_no_ae::text,'?')")
FAERS_LEGEND = ('Counts are in the order: drug and event / drug without event / other drugs with event / other drugs without event; '
                'LLR is the log-likelihood ratio and the threshold is the signal cut-off')
LEGENDS = {h: FAERS_LEGEND for h in ['FAERS Reports in Male Patients', 'FAERS Reports in Female Patients',
                                    'FAERS Reports in Pediatric Patients', 'FAERS Reports in Geriatric Patients']}

# (name, header, sql returning struct_id then one text line per row, source tables)
DRUG_SECTIONS = [
    ('ddd', 'Defined Daily Doses (ATC DDD)',
     "SELECT struct_id, trim(atc_code) || ': ' || coalesce('DDD ' || ddd::text,'DDD n/a') || coalesce(' ' || unit_type,'') || "
     "coalesce('; route ' || route,'') || coalesce('; comment ' || comment,'') FROM atc_ddd ORDER BY struct_id, atc_code, id",
     ['atc_ddd']),
    ('identifiers', 'Chemical Identifiers (all identifier types)',
     "SELECT struct_id, replace(id_type, '_', ' ') || ': ' || identifier || CASE WHEN parent_match THEN ' (matches parent compound)' ELSE '' END "
     "FROM identifier ORDER BY struct_id, id_type, identifier",
     ['identifier', 'id_type']),
    ('inxight', 'Inxight Drugs Cross-reference',
     "SELECT dc_struct_id::int, 'Inxight UNII ' || coalesce(inx_unii,'n/a') || ': ' || coalesce(inx_drug_name,'n/a') || "
     "'; DrugCentral UNII ' || coalesce(dc_xref_unii,'n/a') FROM inxight_drugs WHERE dc_struct_id ~ '^[0-9]+$' ORDER BY 1, inx_unii",
     ['inxight_drugs']),
    ('structure_types', 'Structure Types',
     "SELECT s.struct_id, 'Structure type ' || s.type || coalesce(': ' || d.description,'') FROM structure_type s "
     "LEFT JOIN struct_type_def d ON d.type = s.type ORDER BY s.struct_id, s.type",
     ['structure_type', 'struct_type_def']),
    ('parent', 'Parent Compound',
     "SELECT sp.struct_id, p.name || '; CAS ' || coalesce(p.cas_reg_no,'n/a') || '; SMILES ' || coalesce(p.smiles,'n/a') || "
     "'; InChIKey ' || coalesce(p.inchikey,'n/a') || '; InChI ' || coalesce(p.inchi,'n/a') "
     "FROM struct2parent sp JOIN parentmol p ON p.cd_id = sp.parent_id ORDER BY sp.struct_id, p.name",
     ['struct2parent', 'parentmol']),
    ('pk_az', 'AstraZeneca Pharmacokinetic Measurements',
     "SELECT struct_id, type || ': ' || coalesce(value::text,'n/a') || coalesce(' ' || unit,'') || coalesce('; ' || description,'') || "
     "coalesce('; similarity ' || round(similarity::numeric,3)::text,'') || coalesce('; confidence ' || confidence,'') "
     "FROM pk_az ORDER BY struct_id, type",
     ['pk_az']),
    ('humanim', 'Human and Animal Use',
     "SELECT struct_id, 'Human use: ' || CASE WHEN human THEN 'yes' ELSE 'no' END || '; animal use: ' || "
     "CASE WHEN animal THEN 'yes' ELSE 'no' END FROM humanim ORDER BY struct_id",
     ['humanim']),
    ('vetomop', 'Veterinary Indications and Contraindications',
     "SELECT struct_id, coalesce(species,'species n/a') || ': ' || coalesce(relationship_type,'relationship n/a') || ' - ' || concept_name "
     "FROM vetomop ORDER BY struct_id, species, relationship_type, concept_name",
     ['vetomop']),
    ('vetprod', 'Veterinary Products',
     "SELECT vs.struct_id, st.name || ' is an active ingredient of the veterinary product ' || vp.trade_name || '; applicant ' || coalesce(vp.applicant,'n/a') || '; application ' || vp.appl_type || ' ' || "
     "vp.appl_no || coalesce(' (' || t.description || ')','') || '; active ingredients ' || coalesce(vp.active_ingredients_count::text,'?') || "
     "coalesce('; types ' || (SELECT string_agg(DISTINCT v.type, ', ') FROM vettype v WHERE v.prodid = vp.prodid),'') "
     "FROM vetprod2struct vs JOIN structures st ON st.id = vs.struct_id JOIN vetprod vp ON vp.prodid = vs.prodid LEFT JOIN vetprod_type t ON t.appl_type = vp.appl_type "
     "ORDER BY vs.struct_id, vp.trade_name",
     ['vetprod2struct', 'vetprod', 'vetprod_type', 'vettype']),
    ('faers_male', 'FAERS Reports in Male Patients',
     f"SELECT struct_id, {FAERS_LINE} FROM faers_male ORDER BY struct_id, llr DESC NULLS LAST, meddra_name", ['faers_male']),
    ('faers_female', 'FAERS Reports in Female Patients',
     f"SELECT struct_id, {FAERS_LINE} FROM faers_female ORDER BY struct_id, llr DESC NULLS LAST, meddra_name", ['faers_female']),
    ('faers_ped', 'FAERS Reports in Pediatric Patients',
     f"SELECT struct_id, {FAERS_LINE} FROM faers_ped ORDER BY struct_id, llr DESC NULLS LAST, meddra_name", ['faers_ped']),
    ('faers_ger', 'FAERS Reports in Geriatric Patients',
     f"SELECT struct_id, {FAERS_LINE} FROM faers_ger ORDER BY struct_id, llr DESC NULLS LAST, meddra_name", ['faers_ger']),
    ('faers_top', 'Highest-ranked FAERS Signals',
     "SELECT struct_id, meddra_name FROM faers_top ORDER BY struct_id, meddra_name", ['faers_top']),
    ('ob_patent_view', 'Orange Book Patents with Use Descriptions',
     "SELECT struct_id, 'Patent ' || patent_no || coalesce(' (' || description || ')','') || '; expires ' || "
     "coalesce(patent_expire_date::text,'n/a') || '; product ' || trade_name || coalesce(' ' || strength,'') || '; ' || "
     "coalesce(dose_form,'') || coalesce(', ' || route,'') || '; applicant ' || coalesce(applicant,'n/a') || '; ' || appl_type || ' ' || "
     "appl_no || '; approved ' || coalesce(approval_date::text,'n/a') FROM ob_patent_view ORDER BY struct_id, patent_no, trade_name",
     ['ob_patent_view', 'ob_patent', 'ob_patent_use_code', 'ob_product', 'struct2obprod']),
    ('ob_exclusivity_view', 'Orange Book Exclusivities with Descriptions',
     "SELECT struct_id, coalesce(description,'exclusivity') || '; expires ' || coalesce(exclusivity_date::text,'n/a') || '; product ' || "
     "trade_name || coalesce(' ' || strength,'') || '; ' || coalesce(dose_form,'') || coalesce(', ' || route,'') || '; applicant ' || "
     "coalesce(applicant,'n/a') || '; ' || appl_type || ' ' || appl_no || '; approved ' || coalesce(approval_date::text,'n/a') "
     "FROM ob_exclusivity_view ORDER BY struct_id, exclusivity_date, trade_name",
     ['ob_exclusivity_view', 'ob_exclusivity', 'ob_exclusivity_code', 'ob_product', 'struct2obprod']),
    ('omop_doid', 'SNOMED and Disease Ontology Mappings',
     "SELECT struct_id, relationship_name || ': ' || concept_name || '; DOID ' || doid || coalesce('; UMLS ' || umls_cui,'') || "
     "coalesce('; SNOMED ' || snomed_conceptid::text,'') || coalesce(' ' || snomed_full_name,'') || coalesce('; semantic type ' || cui_semantic_type,'') "
     "FROM omop_relationship_doid_view WHERE doid IS NOT NULL ORDER BY struct_id, relationship_name, concept_name",
     ['omop_relationship_doid_view', 'omop_relationship', 'doid_xref']),
]
LINCS_SQL = """
SELECT s.struct_id, n.signatures, n.cell_lines, top.similar FROM (SELECT DISTINCT struct_id1 AS struct_id FROM lincs_signature) s
JOIN (SELECT struct_id1, count(*) AS signatures, count(DISTINCT cell_id) AS cell_lines FROM lincs_signature GROUP BY struct_id1) n
  ON n.struct_id1 = s.struct_id
JOIN LATERAL (SELECT array_agg(x.line ORDER BY x.pearson DESC) AS similar FROM (
    SELECT l.pearson, st.name || ' (cell line ' || l.cell_id || '; Pearson ' || round(l.pearson::numeric,3)::text ||
           '; RMSD ' || coalesce(round(l.rmsd::numeric,3)::text,'n/a') || '; Euclidean ' || coalesce(round(l.euclid::numeric,3)::text,'n/a') || ')' AS line
    FROM lincs_signature l JOIN structures st ON st.id = l.struct_id2
    WHERE l.struct_id1 = s.struct_id AND l.pearson IS NOT NULL ORDER BY l.pearson DESC LIMIT 10) x) top ON true
ORDER BY s.struct_id
"""


def drug_names(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT id, name FROM structures')
        return {i: n for i, n in cur.fetchall()}


SUBGROUP_WORDS = {'FAERS Reports in Male Patients': 'male patients', 'FAERS Reports in Female Patients': 'female patients',
                  'FAERS Reports in Pediatric Patients': 'pediatric patients', 'FAERS Reports in Geriatric Patients': 'geriatric patients'}
EVENT_LLR = re.compile(r'^(.*?) \(MedDRA [^)]*\): LLR (\S+)')


def faers_summary(drug, header, lines, top=10):
    """One sentence naming the highest-LLR events of a subgroup.

    The detailed lines are long and ranked, so a question about the highest event lands on whichever chunk retrieval prefers.
    A small dedicated chunk states the ranking explicitly and is easy to retrieve. None when no event can be read.
    """
    events = [m.groups() for m in (EVENT_LLR.match(line) for line in lines[:top]) if m]
    if not events:
        return None
    ranked = '; '.join(f'{rank}. {name} (LLR {llr})' for rank, (name, llr) in enumerate(events, 1))
    return f'The highest-LLR FAERS adverse events reported for {drug} in {SUBGROUP_WORDS[header]}, ranked highest first: {ranked}'


def build_drug_sections(conn, version, names, only=None):
    for key, header, sql, tables in DRUG_SECTIONS:
        if only and DRUG_HEADERS[header] not in only:
            continue
        for struct_id, rows in grouped(conn, sql, 'cov_' + key):
            if struct_id in names:
                texts = [r[0] for r in rows]
                if header in SUBGROUP_WORDS:
                    summary = faers_summary(names[struct_id], header, texts)
                    if summary:
                        for chunk in drug_chunks(struct_id, names[struct_id], header, [summary], version, tables):
                            chunk['metadata']['summary'] = True
                            yield chunk
                yield from drug_chunks(struct_id, names[struct_id], header, texts, version, tables)


def build_lincs(conn, version, names):
    header = 'LINCS Signature Similarity'
    for struct_id, signatures, cell_lines, similar in stream(conn, LINCS_SQL, 'cov_lincs'):
        if struct_id not in names:
            continue
        drug = names[struct_id]
        lines = [f'{drug} has {signatures} LINCS signature comparisons across {cell_lines} cell lines '
                 f'(pairwise rows are not embedded individually; the {len(similar)} most similar compounds by Pearson correlation follow)']
        ordinals = ['most', '2nd most', '3rd most'] + [f'{n}th most' for n in range(4, 11)]
        lines += [f'The {ordinals[rank - 1]} similar compound to {drug} by LINCS signature correlation is {text}' for rank, text in enumerate(similar, 1)]
        yield from drug_chunks(struct_id, names[struct_id], header, lines, version, ['lincs_signature'])


def target_title(name, components, limit=6):
    """Target name plus its gene symbols, so every chunk of the profile can be found by symbol (EGFR) as well as by name.

    Human symbols come first; a large complex lists the first few and says how many more there are.
    """
    symbols, seen = [], set()
    for _n, gene, _a, _s, organism, _g, _t in sorted(components, key=lambda c: (c[4] != 'Homo sapiens', c[0] or '')):
        for symbol in GENE_SPLIT.split(gene or ''):
            symbol = symbol.strip()
            if symbol and symbol.casefold() not in seen:
                seen.add(symbol.casefold())
                symbols.append(symbol)
    if not symbols:
        return name
    shown = ', '.join(symbols[:limit]) + (f' and {len(symbols) - limit} more' if len(symbols) > limit else '')
    return f'{name} (gene symbols: {shown})'


def build_targets(conn, version):
    with conn.cursor() as cur:
        cur.execute('SELECT id, name, target_class, protein_type, protein_components, tdl FROM target_dictionary ORDER BY id')
        targets = cur.fetchall()
        cur.execute("""SELECT td.target_id, c.name, c.gene, c.accession, c.swissprot, c.organism, c.geneid, c.tdl
                       FROM td2tc td JOIN target_component c ON c.id = td.component_id ORDER BY td.target_id, c.name""")
        components = defaultdict(list)
        for row in cur.fetchall():
            components[row[0]].append(row[1:])
        cur.execute("""SELECT td.target_id, g.id, g.term, g.type FROM td2tc td JOIN tdgo2tc x ON x.component_id = td.component_id
                       JOIN target_go g ON g.id = x.go_id GROUP BY td.target_id, g.id, g.term, g.type ORDER BY td.target_id, g.type, g.id""")
        go = defaultdict(list)
        for target_id, go_id, term, kind in cur.fetchall():
            go[target_id].append(f'{go_id} {term} ({GO_ASPECT.get(kind, kind)})')
        cur.execute("""SELECT td.target_id, k.keyword, k.category FROM td2tc td JOIN tdkey2tc x ON x.component_id = td.component_id
                       JOIN target_keyword k ON k.id = x.tdkey_id GROUP BY td.target_id, k.keyword, k.category ORDER BY td.target_id, k.keyword""")
        keywords = defaultdict(list)
        for target_id, keyword, category in cur.fetchall():
            keywords[target_id].append(join(keyword, category and f'category {category}', sep=' - '))
        cur.execute("""SELECT a.target_id, s.name, a.action_type, a.organism FROM act_table_full a JOIN structures s ON s.id = a.struct_id
                       GROUP BY a.target_id, s.name, a.action_type, a.organism ORDER BY a.target_id, s.name""")
        drugs = defaultdict(list)
        for target_id, drug, action, organism in cur.fetchall():
            drugs[target_id].append(join(drug, action and f'action {action}', organism and f'organism {organism}'))
    tables = ['target_dictionary', 'td2tc', 'target_component', 'tdgo2tc', 'target_go', 'tdkey2tc', 'target_keyword', 'act_table_full']
    for target_id, name, target_class, protein_type, count, tdl in targets:
        blocks = [('Target Details', [join(f'Target class {target_class}' if target_class else None,
                                           f'protein type {protein_type}' if protein_type else None,
                                           f'protein components {count}' if count is not None else None,
                                           f'target development level {tdl}' if tdl else None, sep=', ') or 'no further details recorded']),
                  ('Protein Components', [join(n, g and f'gene {g}', a and f'UniProt {a}', s and f'Swiss-Prot {s}', o and f'organism {o}',
                                               gid is not None and f'GeneID {gid}', t and f'target development level {t}')
                                          for n, g, a, s, o, gid, t in components[target_id]]),
                  ('Gene Ontology Terms', go[target_id]), ('Target Keywords', keywords[target_id]),
                  ('Drugs with Recorded Activity', drugs[target_id])]
        yield from entity_chunks('target', target_id, 'Target Profile', target_title(name, components[target_id]), blocks, version, tables)


def build_diseases(conn, version):
    with conn.cursor() as cur:
        cur.execute('SELECT id, label, doid, url FROM doid ORDER BY id')
        diseases = cur.fetchall()
        cur.execute('SELECT doid, source, xref FROM doid_xref ORDER BY doid, source, xref')
        xrefs = defaultdict(list)
        for doid, source, xref in cur.fetchall():
            xrefs[doid].append(f"{source.replace('_', ' ')}: {xref}")   # readable: 'UMLS CUI', not 'UMLS_CUI'
        cur.execute("""SELECT d, s.name, v.relationship_name FROM omop_relationship_doid_view v
                       JOIN structures s ON s.id = v.struct_id, unnest(string_to_array(v.doid, ',')) d
                       WHERE v.doid IS NOT NULL GROUP BY d, s.name, v.relationship_name ORDER BY d, s.name""")
        mapped = defaultdict(list)
        for doid, drug, relationship in cur.fetchall():
            mapped[doid].append(f'{drug} ({relationship})')
    for row_id, label, doid, url in diseases:
        blocks = [('Disease Details', [f'Label {label}', f'Disease Ontology ID {doid}'] + ([f'URL {url}'] if url else [])),
                  ('Cross-references', xrefs[doid]), ('Drugs Mapped to this Disease', mapped[doid])]
        yield from entity_chunks('disease', doid, 'Disease Profile', label, blocks, version,
                                 ['doid', 'doid_xref', 'omop_relationship_doid_view'])


def build_references(conn, version):
    for row in stream(conn, "SELECT id, pmid, doi, document_id, type, authors, title, isbn10, url, journal, volume, issue, dp_year, pages "
                            "FROM reference ORDER BY id", 'cov_reference'):
        rid, pmid, doi, doc, kind, authors, title, isbn, url, journal, volume, issue, year, pages = row
        lines = [join(title and f'Title {title}', authors and f'authors {authors}', journal and f'journal {journal}',
                      volume and f'volume {volume}', issue and f'issue {issue}', pages and f'pages {pages}', year and f'year {year}'),
                 join(pmid and f'PMID {pmid}', doi and f'DOI {doi}', doc and f'document {doc}', isbn and f'ISBN-10 {isbn}',
                      url and f'URL {url}', kind and f'type {kind}', f'reference id {rid}')]
        yield from entity_chunks('reference', rid, 'Reference', title or f'reference {rid}', [('Citation', lines)], version, ['reference', 'ref_type'])


def build_ontology(conn, version):
    for term_id, term, kind in stream(conn, 'SELECT id, term, type FROM target_go ORDER BY id', 'cov_go'):
        yield from entity_chunks('go', term_id, 'Gene Ontology Term', f'{term_id} {term}', [(
            'Term Details', [f'GO identifier {term_id}', f'Term {term}', f'Aspect {GO_ASPECT.get(kind, kind)}'])], version, ['target_go'])
    for kid, descr, category, keyword in stream(conn, 'SELECT id, descr, category, keyword FROM target_keyword ORDER BY id', 'cov_keyword'):
        yield from entity_chunks('keyword', kid, 'Target Keyword', f'{kid} {keyword}', [(
            'Term Details', [f'Keyword {keyword}', f'Keyword identifier {kid}', f'Category {category}', f'Description {descr}'])], version,
            ['target_keyword'])


CODE_TABLES = [('ob_patent_use_code', 'FDA Orange Book patent use code'), ('ob_exclusivity_code', 'FDA Orange Book exclusivity code')]
DICTIONARY_TABLES = ['action_type', 'approval_type', 'attr_type', 'data_source', 'dbversion', 'ddi_risk', 'id_type', 'protein_type',
                     'ref_type', 'struct_type_def', 'rdkit_build_meta', 'target_class']


def build_dictionaries(conn, version):
    for table, label in CODE_TABLES:
        for code, description in stream(conn, f'SELECT code, description FROM {table} ORDER BY code', 'cov_' + table):
            yield from entity_chunks('code', f'{table}:{code}', 'Dictionary Code', f'{label} {code}',
                                     [('Code Details', [f'Table {table}', f'Code {code}', f'Description {description}'])], version, [table])
    with conn.cursor() as cur:
        for table in DICTIONARY_TABLES:
            cur.execute(f'SELECT * FROM {table} ORDER BY 1')
            columns = [d[0] for d in cur.description]
            rows = [' ; '.join(f'{c} = {num(v)}' for c, v in zip(columns, row) if v is not None) for row in cur.fetchall()]
            yield from entity_chunks('dictionary', table, 'Dictionary Table', table, [('Rows', rows)], version, [table])


def iter_extra_chunks(conn, version, only=None):
    names = drug_names(conn)
    yield from build_drug_sections(conn, version, names, only)
    if not only or 'lincs' in only:
        yield from build_lincs(conn, version, names)
    if not only or only & {'target_profile'}:
        yield from build_targets(conn, version)
    if not only or 'disease_profile' in only:
        yield from build_diseases(conn, version)
    if not only or 'references' in only:
        yield from build_references(conn, version)
    if not only or 'ontology' in only:
        yield from build_ontology(conn, version)
    if not only or 'dictionary' in only:
        yield from build_dictionaries(conn, version)


def cost_report(stats):
    total = {'chunks': sum(s['chunks'] for s in stats.values()), 'chars': sum(s['chars'] for s in stats.values())}
    total['estimated_tokens'] = total['chars'] // 4
    total['estimated_usd'] = round(total['estimated_tokens'] * EMBED_USD_PER_MILLION / 1e6, 2)
    return {'by_section': {k: {**v, 'estimated_tokens': v['chars'] // 4} for k, v in sorted(stats.items())}, 'total': total,
            'basis': f'chars/4 tokens at ${EMBED_USD_PER_MILLION}/M input tokens; the base dossier chunks are additional'}


def preview(conn, version, output, only=None):
    stats = defaultdict(lambda: {'chunks': 0, 'chars': 0})
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as handle:
        for record in iter_extra_chunks(conn, version, only):
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            s = stats[record['metadata']['section']]
            s['chunks'] += 1
            s['chars'] += len(record['text'])
    return cost_report(stats)


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge_records(base, extra):
    """One snapshot, one dataset version, no duplicate chunk text for the same drug and section."""
    versions = {r['metadata']['dataset_version'] for r in base + extra}
    if len(versions) != 1:
        raise ValueError(f'Base and extra chunks must share one dataset version, found {sorted(versions)}')
    if not base or not extra:
        raise ValueError('Both base and extra chunks are required')
    seen = set()
    for r in base + extra:
        key = (r['metadata'].get('drug_name'), r['metadata'].get('entity_id'), r['metadata'].get('section'), r['text'])
        if key in seen:
            raise ValueError('Duplicate chunk: ' + r['text'][:80])
        seen.add(key)
    return base + extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('preview', help='build the extra chunks to JSONL and print the embedding cost (no Bedrock calls)')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', help='comma-separated section names to build')
    p = sub.add_parser('publish', help='merge base and extra chunks into a new structured snapshot')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--extra', type=Path, required=True)
    p.add_argument('--collection', required=True)
    args = parser.parse_args()
    if args.command == 'preview':
        version = os.environ.get('DATASET_VERSION')
        if not version:
            parser.error('Set DATASET_VERSION to the same value used for the base ingestion')
        from chatbot.database_config import connect_database
        conn = connect_database(source=True)
        try:
            with conn:
                report = preview(conn, version, args.output, set(args.only.split(',')) if args.only else None)
        finally:
            conn.close()
        print(json.dumps(report, indent=2))
    else:
        from dotenv import load_dotenv
        load_dotenv()
        from chatbot.storage import publish_structured_snapshot
        records = merge_records(read_jsonl(args.base), read_jsonl(args.extra))
        publish_structured_snapshot(records, args.collection)
        print(json.dumps({'collection': args.collection, 'records': len(records),
                          'sections': dict(Counter(r['metadata']['section'] for r in records))}, indent=2))
        print(f'Published. Next: python -m ingestion.cohere_migration --source {args.collection} --target <new vector collection> '
              f'--report <file> --apply (this spends Bedrock tokens).')


if __name__ == '__main__':
    main()
