"""Build a product-scoped label search database from a DrugCentral PostgreSQL dump.

Usage: python -m ingestion.labels SOURCE.sql.gz --output data/labels.sqlite
A temporary database is validated and atomically published; existing readers
keep the previous file until their connection closes. No embeddings are changed.
"""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import re
import sqlite3

SECTIONS = {
    'ADVERSE REACTIONS SECTION': 'adverse_reactions',
    'WARNINGS SECTION': 'warnings',
    'WARNINGS AND PRECAUTIONS SECTION': 'warnings',
    'BOXED WARNING SECTION': 'warnings',
    'CONTRAINDICATIONS SECTION': 'contraindications',
    'DRUG INTERACTIONS SECTION': 'interactions',
    'DOSAGE & ADMINISTRATION SECTION': 'dosage',
    'INDICATIONS & USAGE SECTION': 'indications',
    'OVERDOSAGE SECTION': 'overdosage',
    'USE IN SPECIFIC POPULATIONS SECTION': 'populations',
    'PREGNANCY SECTION': 'populations',
    'PEDIATRIC USE SECTION': 'populations',
    'GERIATRIC USE SECTION': 'populations',
    'OTC - STOP USE SECTION': 'warnings',
    'OTC - DO NOT USE SECTION': 'contraindications',
    'OTC - ASK DOCTOR SECTION': 'warnings',
}


def unescape(value):
    if value == r'\N':
        return None
    mapping = {'n': '\n', 'r': '\r', 't': '\t', 'b': '\b', 'f': '\f', 'v': '\v', '\\': '\\'}
    return re.sub(r'\\([0-7]{1,3}|x[0-9a-fA-F]{1,2}|.)',
                  lambda m: chr(int(m[1], 8)) if re.fullmatch('[0-7]{1,3}', m[1]) else
                  chr(int(m[1][1:], 16)) if m[1].startswith('x') and len(m[1]) > 1 else mapping.get(m[1], m[1]), value)


def dump_rows(path, wanted, counts):
    table = None
    with gzip.open(path, 'rt', encoding='utf-8') as source:
        for line in source:
            if line.startswith('COPY public.'):
                table = line.split()[1].split('.', 1)[1]
                columns = line.split('(', 1)[1].split(')', 1)[0].split(', ')
                continue
            if line.rstrip('\n') == r'\.':
                table = None
                continue
            if table is None:
                continue
            counts[table] += 1
            if table in wanted:
                values = line.rstrip('\n').split('\t')
                if len(values) != len(columns):
                    raise ValueError(f'Invalid COPY record in {table}')
                yield table, dict(zip(columns, map(unescape, values)))


def build(source, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.building.sqlite')
    if temp.exists():
        raise ValueError(f'Incomplete build exists: {temp}; inspect/remove it before retrying')
    counts = Counter()
    conn = sqlite3.connect(temp)
    try:
        conn.executescript('''
        PRAGMA journal_mode=OFF;
        CREATE TABLE products(ndc TEXT PRIMARY KEY, name TEXT, generic_name TEXT, route TEXT, form TEXT, ingredient_count INTEGER);
        CREATE TABLE ingredients(ndc TEXT, struct_id INTEGER, name TEXT);
        CREATE TABLE links(ndc TEXT, label_id TEXT);
        CREATE TABLE labels(id TEXT PRIMARY KEY, title TEXT, effective_date TEXT, pdf_url TEXT);
        CREATE TABLE sections(id INTEGER PRIMARY KEY, label_id TEXT, section TEXT, title TEXT, text TEXT);
        ''')
        buffers = {key: [] for key in ('products', 'ingredients', 'links', 'labels', 'sections')}
        def insert(table, row):
            buffers[table].append(row)
            if len(buffers[table]) >= 2000:
                flush(table)
        def flush(table):
            rows = buffers[table]
            if rows:
                conn.executemany(f"INSERT OR IGNORE INTO {table} VALUES ({','.join('?' for _ in rows[0])})", rows)
                rows.clear()
                conn.commit()
        for table, r in dump_rows(source, {'product', 'active_ingredient', 'prd2label', 'label', 'section'}, counts):
            if table == 'product':
                insert('products', (r['ndc_product_code'], r['product_name'], r['generic_name'], r['route'], r['form'], r['active_ingredient_count']))
            elif table == 'active_ingredient' and r['struct_id']:
                insert('ingredients', (r['ndc_product_code'], r['struct_id'], r['substance_name']))
            elif table == 'prd2label':
                insert('links', (r['ndc_product_code'], r['label_id']))
            elif table == 'label':
                insert('labels', (r['id'], r['title'], r['effective_date'], r['pdf_url']))
            elif table == 'section' and r['title'] in SECTIONS and r['text']:
                insert('sections', (r['id'], r['label_id'], SECTIONS[r['title']], r['title'], r['text']))
        for table in buffers:
            flush(table)
        conn.executescript('''
        CREATE INDEX ingredient_drug ON ingredients(struct_id, ndc);
        CREATE INDEX ingredient_product ON ingredients(ndc, struct_id);
        CREATE INDEX product_labels ON links(ndc, label_id);
        CREATE INDEX label_sections ON sections(label_id, section);
        CREATE VIRTUAL TABLE section_search USING fts5(text, content='sections', content_rowid='id');
        INSERT INTO section_search(section_search) VALUES('rebuild');
        ANALYZE;
        ''')
        report = {'source': Path(source).name, 'source_tables': dict(counts),
                  'imported': {t: conn.execute(f'SELECT count(*) FROM {t}').fetchone()[0] for t in buffers}}
        if conn.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or not report['imported']['sections']:
            raise ValueError('Label database failed validation')
        conn.close()
        os.replace(temp, output)
        output.with_suffix('.audit.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    finally:
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('--output', default='data/labels.sqlite')
    args = parser.parse_args()
    build(args.source, args.output)
