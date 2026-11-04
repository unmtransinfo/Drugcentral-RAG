"""Build a reproducible, complete public-table coverage benchmark from a SQL dump.

This is a database coverage/gap benchmark, not a claim that the serving RAG exposes
all source tables. No model answers or benchmark scores affect selection.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
from ingestion.labels import unescape


def scan(source, seed):
    rng=random.Random(seed); tables={}; names={}; current=None; altered=None
    def entry(name):
        return tables.setdefault(name,dict(columns=[],row_count=0,eligible_rows=0,samples=[],primary_key=[]))
    with gzip.open(source,'rt',encoding='utf-8') as handle:
        for line in handle:
            if current is not None:
                if line.rstrip('\n')==r'\.':current=None;continue
                t=entry(current);t['row_count']+=1
                if current=='structures':
                    values=line.rstrip('\n').split('\t');r=dict(zip(t['columns'],values))
                    names[r['id']]=unescape(r['name'])
                if len(line)<=100000:
                    t['eligible_rows']+=1; n=t['eligible_rows'];slot=rng.randrange(n)
                    sample={'copy_row_number':t['row_count'],'raw':line.rstrip('\n')}
                    if len(t['samples'])<12:t['samples'].append(sample)
                    elif slot<12:t['samples'][slot]=sample
                continue
            m=re.match(r'CREATE TABLE public\.([^\s(]+)',line)
            if m:entry(m[1])
            m=re.match(r'COPY public\.([^\s]+) \((.*?)\) FROM stdin;',line)
            if m:
                current=m[1];entry(current)['columns']=m[2].split(', ');continue
            m=re.match(r'ALTER TABLE ONLY public\.([^\s;]+)',line)
            if m:altered=m[1]
            m=re.search(r'PRIMARY KEY \(([^)]+)\)',line)
            if m and altered in tables:tables[altered]['primary_key']=m[1].split(', ')
    return tables,names


def ingestion_paths():
    sql=Path('ingestion/vectors.py').read_text()
    vector_tables=set(re.findall(r'\b(?:FROM|JOIN)\s+([a-z][a-z0-9_]*)',sql))
    labels={'product','active_ingredient','prd2label','label','section'}
    relations={'structures','target_dictionary','act_table_full'}
    return vector_tables,labels,relations


def candidates(table,info,names):
    result=[]
    for sample in info['samples']:
        values=sample['raw'].split('\t')
        if len(values)!=len(info['columns']):raise ValueError('Malformed COPY row: '+table)
        row=dict(zip(info['columns'],map(unescape,values)))
        primary=info['primary_key']
        if not primary:
            # Explicit dump row numbers provide unambiguous identity when no PK exists.
            selector={'copy_row_number':sample['copy_row_number']}
        else:selector={k:row[k] for k in primary}
        drug=names.get(row.get('struct_id')) or (row.get('name') if table=='structures' else None)
        for column,value in row.items():
            if column in primary or value is None or not value.strip() or len(value)>240 or '\n' in value:continue
            if column in {'molfile','molimg','rgb'}:continue
            key=', '.join(f'{k}={json.dumps(v,ensure_ascii=False)}' for k,v in selector.items())
            scope=f' for {drug}' if drug else ''
            question=f'In the DrugCentral source snapshot, what is the recorded {column} value{scope} in table {table} for record {key}?'
            if len(question)>900:continue
            result.append(dict(question=question,drug=drug,reference_facts=[value],reference_answer=value,
                source_row_key=selector,requested_columns=[column],reference_record=row,
                source_row_sha256=hashlib.sha256(sample['raw'].encode()).hexdigest(),
                copy_row_number=sample['copy_row_number'],question_kind='record_field'))
    result.append(dict(question=f'How many records does table {table} contain in the supplied DrugCentral source snapshot?',
        drug=None,reference_facts=[str(info['row_count'])],reference_answer=str(info['row_count']),
        source_row_key=None,requested_columns=[],reference_record={'row_count':info['row_count']},question_kind='snapshot_count'))
    return result


def build(source,output,count=300,seed=20261003,provenance=None):
    if output.exists():raise ValueError('Output already exists; preserve prior benchmarks')
    tables,names=scan(source,seed)
    if provenance and 'archive_tables' in provenance:
        expected=set(provenance['archive_tables'])
        actual={'public.'+name for name in tables}
        if actual!=expected:raise ValueError(f'Archive/schema inventory mismatch: {sorted(actual ^ expected)}')
    if count<len(tables):raise ValueError('Question budget must cover every source table')
    vector,label,relations=ingestion_paths();rng=random.Random(seed);pools={}
    for table,info in sorted(tables.items()):
        pools[table]=candidates(table,info,names);rng.shuffle(pools[table])
    selected=[];counts=Counter()
    while len(selected)<count:
        before=len(selected)
        for table in sorted(tables):
            if not pools[table]:continue
            q=pools[table].pop();counts[table]+=1
            q.update(id=f'full-{seed}-{table}-{counts[table]:03d}',category=table,source_table=table,
                source_tables=[table],benchmark_scope='full_source_table_coverage',
                source_url=f'{source.name}#public.{table}',section_header=f'public.{table}',
                answer_rule='Answer the requested column for the exact identified source record; do not substitute another drug, subgroup, record, or table. Counts refer only to this frozen dump.',
                reference_sha256=hashlib.sha256(json.dumps(q['reference_facts'],ensure_ascii=False).encode()).hexdigest(),
                review={'status':'pending','reviewer':None})
            selected.append(q)
            if len(selected)==count:break
        if len(selected)==before:raise ValueError('Insufficient distinct source questions')
    assert set(counts)==set(tables)
    assert len({q['question'] for q in selected})==count
    digest=hashlib.sha256()
    with source.open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    source_hash=digest.hexdigest()
    for q in selected:q['dataset_version']='sha256:'+source_hash
    coverage={}
    for table,info in sorted(tables.items()):
        paths=[]
        if table in vector:paths.append('structured_vector_ingestion')
        if table in label:paths.append('product_label_ingestion')
        if table in relations:paths.append('relation_lookup_ingestion')
        coverage[table]={k:v for k,v in info.items() if k not in {'samples','eligible_rows'}}
        coverage[table].update(question_count=counts[table],question_ids=[q['id'] for q in selected if q['source_table']==table],
            referenced_by_ingestion=paths,serving_status='partial table/column use; exact-record answerability not established' if paths else 'not referenced by current ingestion paths',
            sampled_rows=info['eligible_rows'],oversized_rows_excluded_from_sampling=info['row_count']-info['eligible_rows'])
    output.mkdir(parents=True)
    encoded=json.dumps(selected,ensure_ascii=False,indent=2)+'\n'
    (output/'questions.json').write_text(encoded)
    (output/'table_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    manifest=dict(questions=count,source_tables=len(tables),covered_tables=len(counts),table_coverage_percent=100,
        source_file=str(source),source_sha256=source_hash,seed=seed,source_provenance=provenance,questions_by_table=dict(sorted(counts.items())),
        questions_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        coverage_definition='Every public source table discovered in CREATE TABLE/COPY statements has at least one exact-record or snapshot-count question. Not exhaustive column/row coverage or proof of serving ingestion.',
        sampling='Seeded reservoir of 12 COPY rows per table, excluding rows longer than 100000 characters; literal nonempty fields <=240 characters; round-robin question allocation.',
        serving_gap_tables=[t for t,v in coverage.items() if not v['referenced_by_ingestion']],
        review_status='Pending independent source/answer review')
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text('# Full source-table benchmark\n\n'+f'{count} questions cover all {len(tables)} public tables in the frozen SQL dump.\n\nThese are explicit database record/field questions, including lookup dictionaries, link tables, veterinary data, and subgroup FAERS tables. They intentionally test ingestion gaps. This is not a claim that the current chatbot can answer every question. Table coverage does not mean every column or row is tested.\n\n`table_coverage.json` lists row counts, primary keys, question IDs and ingestion references. `questions.json` retains exact reference records for review; the runner sends only the question text to the API. Automatic retrieval scoring remains unmeasured because current traces do not establish exact source-table/row provenance. Review answers against the saved reference records.\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='questions_by_table'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('data/source/drugcentral.dump.09022026.sql.gz'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=300);p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--source-provenance',type=Path,help='JSON with immutable image identity and extraction provenance')
    a=p.parse_args();build(a.source,a.output,a.count,a.seed,json.loads(a.source_provenance.read_text()) if a.source_provenance else None)
