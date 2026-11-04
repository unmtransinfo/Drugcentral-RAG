"""Build the Neo4j relationship graph from the authoritative DrugCentral PostgreSQL database.

The graph is a derived index: it can be dropped and rebuilt at any time and never becomes a source
of truth. Every relationship records source_table, source_pk and dataset_version so an answer can
be traced to the PostgreSQL row it came from.

Run: python -m ingestion.graph_build [--rebuild] [--with-products]
Env: SOURCE_DATABASE_URL (or SOURCE_PG*), NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
"""
import argparse
import json
import os
import re

BATCH = 5000

CONSTRAINTS = [
    'CREATE CONSTRAINT drug_key IF NOT EXISTS FOR (n:Drug) REQUIRE n.struct_id IS UNIQUE',
    'CREATE CONSTRAINT target_key IF NOT EXISTS FOR (n:Target) REQUIRE n.target_id IS UNIQUE',
    'CREATE CONSTRAINT gene_key IF NOT EXISTS FOR (n:Gene) REQUIRE n.key IS UNIQUE',
    'CREATE CONSTRAINT disease_key IF NOT EXISTS FOR (n:Disease) REQUIRE n.concept_id IS UNIQUE',
    'CREATE CONSTRAINT atc_key IF NOT EXISTS FOR (n:ATC) REQUIRE n.code IS UNIQUE',
    'CREATE CONSTRAINT approval_key IF NOT EXISTS FOR (n:Approval) REQUIRE n.id IS UNIQUE',
    'CREATE CONSTRAINT class_key IF NOT EXISTS FOR (n:DrugClass) REQUIRE n.name IS UNIQUE',
    'CREATE CONSTRAINT product_key IF NOT EXISTS FOR (n:Product) REQUIRE n.id IS UNIQUE',
    'CREATE INDEX drug_name IF NOT EXISTS FOR (n:Drug) ON (n.name_lower)',
    'CREATE INDEX target_name IF NOT EXISTS FOR (n:Target) ON (n.name_lower)',
    'CREATE INDEX gene_symbol IF NOT EXISTS FOR (n:Gene) ON (n.symbol_lower)',
    'CREATE INDEX disease_name IF NOT EXISTS FOR (n:Disease) ON (n.name_lower)',
]

# (name, source SQL, Cypher). `$version` is the source dataset version; rows are `r`.
LOADERS = [
    ('drugs', 'SELECT id AS struct_id, name FROM structures',
     '''UNWIND $rows AS r MERGE (d:Drug {struct_id: r.struct_id})
        SET d.name = r.name, d.name_lower = toLower(r.name), d.source_table = 'structures',
            d.source_pk = r.struct_id, d.dataset_version = $version'''),
    ('targets', 'SELECT id AS target_id, name, target_class FROM target_dictionary',
     '''UNWIND $rows AS r MERGE (t:Target {target_id: r.target_id})
        SET t.name = r.name, t.name_lower = toLower(r.name), t.target_class = r.target_class,
            t.source_table = 'target_dictionary', t.source_pk = r.target_id, t.dataset_version = $version'''),
    ('drug_targets',
     'SELECT act_id, struct_id, target_id, organism, action_type, moa FROM act_table_full',
     '''UNWIND $rows AS r MATCH (d:Drug {struct_id: r.struct_id}), (t:Target {target_id: r.target_id})
        MERGE (d)-[e:TARGETS {act_id: r.act_id}]->(t)
        SET e.organism = r.organism, e.action_type = r.action_type, e.moa = r.moa,
            e.source_table = 'act_table_full', e.source_pk = r.act_id, e.dataset_version = $version'''),
    ('atc', '''SELECT trim(code) AS code, chemical_substance, l1_name, l2_name, l3_name, l4_name FROM atc''',
     '''UNWIND $rows AS r MERGE (a:ATC {code: r.code})
        SET a.name = r.chemical_substance, a.l1_name = r.l1_name, a.l2_name = r.l2_name, a.l3_name = r.l3_name,
            a.l4_name = r.l4_name, a.source_table = 'atc', a.dataset_version = $version'''),
    ('drug_atc', "SELECT struct_id, trim(atc_code) AS atc_code, struct_id || ':' || trim(atc_code) AS pk FROM struct2atc",
     '''UNWIND $rows AS r MATCH (d:Drug {struct_id: r.struct_id}), (a:ATC {code: r.atc_code})
        MERGE (d)-[e:HAS_ATC]->(a)
        SET e.source_table = 'struct2atc', e.source_pk = r.pk, e.dataset_version = $version'''),
    ('approvals', "SELECT id, struct_id, approval::text AS date, type, applicant, orphan FROM approval",
     '''UNWIND $rows AS r MATCH (d:Drug {struct_id: r.struct_id})
        MERGE (p:Approval {id: r.id}) SET p.agency = r.type, p.date = r.date, p.applicant = r.applicant,
            p.orphan = r.orphan, p.source_table = 'approval', p.dataset_version = $version
        MERGE (d)-[e:HAS_APPROVAL]->(p)
        SET e.source_table = 'approval', e.source_pk = r.id, e.dataset_version = $version'''),
    ('drug_classes', 'SELECT DISTINCT name FROM drug_class',
     '''UNWIND $rows AS r MERGE (c:DrugClass {name: r.name}) SET c.source_table = 'drug_class', c.dataset_version = $version'''),
    ('drug_class_members', 'SELECT s.id, s.struct_id, c.name AS class_name FROM struct2drgclass s JOIN drug_class c ON c.id = s.drug_class_id',
     '''UNWIND $rows AS r MATCH (d:Drug {struct_id: r.struct_id}), (c:DrugClass {name: r.class_name})
        MERGE (d)-[e:IN_CLASS {pk: r.id}]->(c)
        SET e.source_table = 'struct2drgclass', e.source_pk = r.id, e.dataset_version = $version'''),
    ('class_interactions', 'SELECT id, drug_class1, drug_class2, ddi_risk, description FROM ddi',
     '''UNWIND $rows AS r MATCH (a:DrugClass {name: r.drug_class1}), (b:DrugClass {name: r.drug_class2})
        MERGE (a)-[x:INTERACTS_WITH {pk: r.id}]->(b)
        SET x.risk = r.ddi_risk, x.description = r.description,
            x.source_table = 'ddi', x.source_pk = r.id, x.dataset_version = $version'''),
]
# One loader per relationship type: Cypher cannot parameterise a relationship type.
CONCEPT_RELATIONS = [('indication', 'INDICATED_FOR'), ('contraindication', 'CONTRAINDICATED_FOR'),
                     ('off-label use', 'OFF_LABEL_FOR'), ('symptomatic treatment', 'TREATS_SYMPTOMS_OF'),
                     ('reduce risk', 'REDUCES_RISK_OF')]
for _name, _type in CONCEPT_RELATIONS:
    LOADERS.append((f'disease_{_type.lower()}',
        "SELECT id, struct_id, concept_id, concept_name FROM omop_relationship WHERE relationship_name = '" + _name + "'",
        f'''UNWIND $rows AS r MATCH (d:Drug {{struct_id: r.struct_id}})
            MERGE (x:Disease {{concept_id: r.concept_id}})
              SET x.name = r.concept_name, x.name_lower = toLower(r.concept_name),
                  x.source_table = 'omop_relationship', x.dataset_version = $version
            MERGE (d)-[e:{_type} {{pk: r.id}}]->(x)
            SET e.source_table = 'omop_relationship', e.source_pk = r.id, e.dataset_version = $version'''))

PRODUCT_LOADERS = [
    ('products', 'SELECT id, ndc_product_code, product_name FROM product',
     '''UNWIND $rows AS r MERGE (p:Product {id: r.id}) SET p.ndc = r.ndc_product_code, p.name = r.product_name,
        p.source_table = 'product', p.dataset_version = $version'''),
    ('product_ingredients',
     '''SELECT p.id AS product_id, ai.struct_id, ai.id AS pk FROM product p
        JOIN active_ingredient ai ON ai.ndc_product_code = p.ndc_product_code WHERE ai.struct_id IS NOT NULL''',
     '''UNWIND $rows AS r MATCH (p:Product {id: r.product_id}), (d:Drug {struct_id: r.struct_id})
        MERGE (p)-[e:CONTAINS]->(d)
        SET e.source_table = 'active_ingredient', e.source_pk = r.pk, e.dataset_version = $version'''),
]
GENE_SPLIT = re.compile(r'[|;,]')


def genes(rows):
    """Unique (target, symbol, organism) triples from the multi-symbol gene column."""
    seen = {}
    for target_id, gene, organism in rows:
        for symbol in GENE_SPLIT.split(gene or ''):
            symbol = symbol.strip()
            if symbol:
                key = f'{symbol.casefold()}|{(organism or "").casefold()}'
                seen[(target_id, key)] = {'target_id': target_id, 'key': key, 'symbol': symbol,
                                          'symbol_lower': symbol.casefold(), 'organism': organism}
    return list(seen.values())


GENE_CYPHER = '''UNWIND $rows AS r MATCH (t:Target {target_id: r.target_id})
    MERGE (g:Gene {key: r.key}) SET g.symbol = r.symbol, g.symbol_lower = r.symbol_lower, g.organism = r.organism,
        g.source_table = 'act_table_full', g.dataset_version = $version
    MERGE (t)-[e:ENCODED_BY]->(g)
    SET e.source_table = 'act_table_full', e.source_pk = r.target_id, e.dataset_version = $version'''


def batches(cursor, size=BATCH):
    while True:
        chunk = cursor.fetchmany(size)
        if not chunk:
            return
        # A server-side cursor reports its columns only after the first fetch.
        columns = [d[0] for d in cursor.description]
        yield [dict(zip(columns, row)) for row in chunk]


def dataset_version(conn):
    with conn.cursor() as cur:
        cur.execute('SELECT version, dtime::text FROM dbversion ORDER BY dtime DESC LIMIT 1')
        version, loaded = cur.fetchone()
    return f'drugcentral-{version}-{loaded[:10]}'


def build(conn, driver, rebuild=False, with_products=False, database='neo4j'):
    version = dataset_version(conn)
    report = {'dataset_version': version, 'loaded': {}}
    with driver.session(database=database) as session:
        existing = session.run('MATCH (n) RETURN count(n) AS c').single()['c']
        if existing and not rebuild:
            raise ValueError('The graph is not empty; use --rebuild to replace this derived index')
        if existing:
            while session.run('MATCH (n) WITH n LIMIT 10000 DETACH DELETE n RETURN count(*) AS c').single()['c']:
                pass
        for statement in CONSTRAINTS:
            session.run(statement)
        loaders = LOADERS + (PRODUCT_LOADERS if with_products else [])
        for name, sql, cypher in loaders:
            total = 0
            with conn.cursor(name='graph_' + name) as cur:
                cur.execute(sql)
                for rows in batches(cur):
                    session.execute_write(lambda tx, rows=rows: tx.run(cypher, rows=rows, version=version).consume())
                    total += len(rows)
            report['loaded'][name] = total
            if name == 'drug_targets':
                with conn.cursor() as cur:
                    cur.execute('SELECT target_id, gene, organism FROM act_table_full')
                    pairs = genes(cur.fetchall())
                for start in range(0, len(pairs), BATCH):
                    session.execute_write(lambda tx, rows=pairs[start:start + BATCH]: tx.run(GENE_CYPHER, rows=rows, version=version).consume())
                report['loaded']['target_genes'] = len(pairs)
        session.run('MERGE (m:GraphMeta {id: 1}) SET m.dataset_version = $version, m.built_at = datetime()', version=version).consume()
        report['nodes'] = {r['label']: r['c'] for r in session.run(
            'MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label')}
        report['relationships'] = {r['t']: r['c'] for r in session.run(
            'MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS c ORDER BY t')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', action='store_true', help='drop the existing derived graph first')
    parser.add_argument('--with-products', action='store_true', help='also load products and ingredients (large)')
    args = parser.parse_args()
    from neo4j import GraphDatabase
    from chatbot.database_config import connect_database
    conn = connect_database(source=True)
    driver = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD']))
    try:
        print(json.dumps(build(conn, driver, args.rebuild, args.with_products, os.getenv('NEO4J_DATABASE', 'neo4j')), indent=2))
    finally:
        conn.close()
        driver.close()


if __name__ == '__main__':
    main()
