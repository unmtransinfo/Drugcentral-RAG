"""Inventory the complete DrugCentral source database and map it to retrieval mechanisms.

The inventory covers every table, not only those that feed the vector collection, so the
graph and structured-query layers can be designed from the real schema.

Run: python -m ingestion.schema_inventory --output-json docs/source-inventory.json \
        --output-markdown docs/SOURCE_RETRIEVAL_MAP.md
Connection: SOURCE_DATABASE_URL or SOURCE_PGHOST/SOURCE_PGDATABASE/... (see chatbot/database_config.py).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

# (category, table-name pattern); first match wins.
CATEGORY_RULES = [
    ('provenance', r'^(data_source|dbversion|reference|ref_type|attr_type|inxight_drugs)$|rdkit_build_meta'),
    ('adverse_events', r'^faers'),
    ('regulatory', r'^(approval|approval_type|ob_|struct2obprod)'),
    ('interactions', r'^(ddi|ddi_risk|drug_class|struct2drgclass)'),
    ('atc_ddd', r'^(atc|struct2atc|atc_ddd)'),
    ('veterinary', r'^vet'),
    ('targets', r'^(act_table_full|target_dictionary|target_component|target_class|target_go|target_keyword|td2tc|tdkey2tc|tdgo2tc|protein_type|action_type|humanim|lincs)'),
    ('indications', r'^(omop_relationship|doid|doid_xref|omop)'),
    ('products_labels', r'^(product|active_ingredient|prd2label|label|section|pharma_class|ijc_)'),
    ('structures_chemistry', r'^(structures|structure_type|struct_type_def|struct2parent|parentmol|property|property_type|pka|pk_az|identifier|id_type|inn_stem|synonyms|pdb)'),
]
CATEGORY_MECHANISM = {
    'structures_chemistry': 'Direct PostgreSQL query for exact values (formula, weight, identifiers, aliases)',
    'indications': 'PostgreSQL relationship query / hybrid RAG; Neo4j projection (Drug-INDICATED_FOR-Disease)',
    'targets': 'PostgreSQL relationship query; Neo4j projection (Drug-TARGETS-Target-ENCODED_BY-Gene)',
    'atc_ddd': 'Direct PostgreSQL lookup; Neo4j projection (Drug-HAS_ATC-ATC)',
    'regulatory': 'Direct PostgreSQL lookup and evidence binding; Neo4j projection (Drug-HAS_APPROVAL)',
    'interactions': 'PostgreSQL relationship query; Neo4j projection (DrugClass-INTERACTS_WITH-DrugClass)',
    'products_labels': 'Section-aware hybrid RAG for long label text; direct lookup for product identity',
    'adverse_events': 'Hybrid RAG with FAERS caveat; direct PostgreSQL for exact counts',
    'veterinary': 'Direct PostgreSQL lookup; keep separate from human indications and interactions',
    'provenance': 'Direct PostgreSQL lookup and evidence binding',
    'other': 'Review manually; no mechanism assigned',
}
# Source tables read by the existing pipelines, so unrepresented ones are visible.
VECTOR_SQL = ROOT / 'ingestion' / 'vectors.py'
LABEL_TABLES = {'product', 'active_ingredient', 'prd2label', 'label', 'section'}
RELATION_TABLES = {'structures', 'target_dictionary', 'act_table_full'}


def classify(table):
    for category, pattern in CATEGORY_RULES:
        if re.search(pattern, table):
            return category
    return 'other'


def tables_in_sql(text):
    """Tables named after FROM/JOIN; CTE and derived-table aliases are not present in the schema anyway."""
    return set(re.findall(r'\b(?:FROM|JOIN)\s+(?:public\.)?([a-z_][a-z0-9_]*)\b', text, re.I))


def representation(table, vector_tables):
    from ingestion.full_coverage import COVERAGE
    where = []
    if COVERAGE.get(table, '').startswith(('drug:', 'entity:')):
        where.append('full_coverage')
    if table in vector_tables:
        where.append('vector_evidence')
    if table in LABEL_TABLES:
        where.append('label_database')
    if table in RELATION_TABLES:
        where.append('relations_sqlite')
    return where or ['unrepresented']


def collect(conn, exact_counts=True):
    """Read tables, columns, keys and row counts from the live source database."""
    with conn.cursor() as cur:
        cur.execute("""SELECT c.relname, c.reltuples::bigint, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
                       WHERE n.nspname='public' AND c.relkind IN ('r','p','v') ORDER BY 1""")
        tables = {name: {'estimated_rows': max(0, int(est)), 'columns': [], 'primary_key': [], 'foreign_keys': [],
                         'kind': 'view' if kind == 'v' else 'table'}
                  for name, est, kind in cur.fetchall()}
        cur.execute("""SELECT table_name, column_name, data_type, udt_name, is_nullable FROM information_schema.columns
                       WHERE table_schema='public' ORDER BY table_name, ordinal_position""")
        for table, column, data_type, udt, nullable in cur.fetchall():
            if table in tables:
                kind = udt if data_type in ('USER-DEFINED', 'ARRAY') else data_type
                tables[table]['columns'].append({'name': column, 'type': kind, 'nullable': nullable == 'YES'})
        cur.execute("""SELECT cl.relname, con.contype, con.conname,
                              (SELECT array_agg(a.attname ORDER BY k.ord) FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
                                 JOIN pg_attribute a ON a.attrelid=con.conrelid AND a.attnum=k.attnum),
                              fcl.relname,
                              (SELECT array_agg(a.attname ORDER BY k.ord) FROM unnest(con.confkey) WITH ORDINALITY k(attnum, ord)
                                 JOIN pg_attribute a ON a.attrelid=con.confrelid AND a.attnum=k.attnum)
                       FROM pg_constraint con JOIN pg_class cl ON cl.oid=con.conrelid
                       JOIN pg_namespace n ON n.oid=cl.relnamespace LEFT JOIN pg_class fcl ON fcl.oid=con.confrelid
                       WHERE n.nspname='public' AND con.contype IN ('p','f')""")
        for table, kind, name, columns, ftable, fcolumns in cur.fetchall():
            if table not in tables:
                continue
            if kind == 'p':
                tables[table]['primary_key'] = columns
            else:
                tables[table]['foreign_keys'].append({'columns': columns, 'references': ftable, 'referenced_columns': fcolumns})
        if exact_counts:
            for table in tables:
                cur.execute(f'SELECT count(*) FROM public."{table}"')
                tables[table]['rows'] = cur.fetchone()[0]
    return tables


def build_inventory(tables, vector_tables=None):
    vector_tables = set(vector_tables if vector_tables is not None else tables_in_sql(VECTOR_SQL.read_text()))
    inventory = {}
    for name, info in sorted(tables.items()):
        category = classify(name)
        inventory[name] = {**info, 'category': category, 'retrieval': CATEGORY_MECHANISM[category],
                           'derived_into': representation(name, vector_tables),
                           'links_to_drug': name == 'structures' or any(c['name'] == 'struct_id' for c in info['columns'])}
    return inventory


def render_markdown(inventory):
    by_category = defaultdict(list)
    for name, info in inventory.items():
        by_category[info['category']].append((name, info))
    unrepresented = [n for n, i in inventory.items() if i['derived_into'] == ['unrepresented']]
    lines = ['# DrugCentral source-to-retrieval map', '',
             'Generated by `ingestion/schema_inventory.py` from the complete source database. '
             'PostgreSQL is authoritative; vectors, FTS indexes, graph edges and caches are derived.', '',
             f"Source objects: {len(inventory)} ({sum(i.get('kind') != 'view' for i in inventory.values())} tables, "
             f"{sum(i.get('kind') == 'view' for i in inventory.values())} views). Not represented in any retrieval store: {len(unrepresented)}.", '']
    for category in sorted(by_category):
        lines += [f'## {category}', '', f'Retrieval: {CATEGORY_MECHANISM[category]}', '',
                  '| Table | Rows | Primary key | Foreign keys | Derived into |', '|---|---:|---|---|---|']
        for name, info in by_category[category]:
            fks = '; '.join(f"{','.join(f['columns'])}→{f['references']}" for f in info['foreign_keys']) or '—'
            rows = info.get('rows', info['estimated_rows'])
            lines.append(f"| {name}{' (view)' if info.get('kind') == 'view' else ''} | {rows} | {','.join(info['primary_key'] or []) or '—'} | {fks} | {', '.join(info['derived_into'])} |")
        lines.append('')
    lines += ['## Tables not yet represented', '',
              'These hold facts the vector collection cannot answer. Route exact questions to direct PostgreSQL '
              'queries, and consider them for the relationship graph:', '', ', '.join(unrepresented) or 'none', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-json', type=Path, default=ROOT / 'docs' / 'source-inventory.json')
    parser.add_argument('--output-markdown', type=Path, default=ROOT / 'docs' / 'SOURCE_RETRIEVAL_MAP.md')
    parser.add_argument('--estimate-only', action='store_true', help='skip exact row counts')
    args = parser.parse_args()
    from chatbot.database_config import connect_database
    conn = connect_database(source=True)
    try:
        with conn:
            inventory = build_inventory(collect(conn, exact_counts=not args.estimate_only))
    finally:
        conn.close()
    args.output_json.write_text(json.dumps(inventory, indent=2, sort_keys=True))
    args.output_markdown.write_text(render_markdown(inventory))
    print(f'Wrote {len(inventory)} tables to {args.output_json} and {args.output_markdown}')


if __name__ == '__main__':
    main()
