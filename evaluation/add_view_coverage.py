"""Add real view-result questions to a frozen table benchmark, retaining its size."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

VIEWS={'faers_top':'meddra_name','ob_exclusivity_view':'description',
       'ob_patent_view':'patent_expire_date','omop_relationship_doid_view':'doid'}


def literal(value):
    return "'"+value.replace("'","''")+"'"


def query(container,sql):
    result=subprocess.run(['docker','exec',container,'psql','-X','-qAt','-U','postgres','-d','postgres',
        '-v','ON_ERROR_STOP=1','-c',"SET statement_timeout='60s'; "+sql],check=True,capture_output=True,text=True)
    return json.loads(result.stdout)


def rebalance(original,additions):
    rows=list(original);counts=Counter(q['source_table'] for q in rows)
    for _ in additions:
        eligible=[t for t,n in counts.items() if n>1]
        if not eligible:raise ValueError('Cannot retain all tables within question budget')
        table=max(eligible,key=lambda t:(counts[t],t))
        i=max(i for i,q in enumerate(rows) if q['source_table']==table)
        rows.pop(i);counts[table]-=1
    return rows+additions


def build(base,output,container):
    if output.exists():raise ValueError('Preserve existing benchmark outputs')
    manifest=json.loads((base/'manifest.json').read_text())
    original=json.loads((base/'questions.json').read_text())
    live=query(container,"SELECT coalesce(json_agg(table_name ORDER BY table_name),'[]') FROM information_schema.views WHERE table_schema='public'")
    if set(live)!=set(VIEWS):raise ValueError('Unexpected view inventory: '+str(live))
    coverage={};new=[]
    for view,column in VIEWS.items():
        count=query(container,f'SELECT to_json(count(*)) FROM public.{view}')
        definition=query(container,f"SELECT to_json(pg_get_viewdef('public.{view}'::regclass,true))")
        samples=query(container,f"SELECT coalesce(json_agg(r),'[]') FROM (SELECT * FROM public.{view} v WHERE to_jsonb(v)->{literal(column)} <> 'null'::jsonb ORDER BY to_jsonb(v)::text LIMIT 4) r")
        entries=[]
        for row in samples:
            selector={k:v for k,v in row.items() if k!=column}
            selector_json=json.dumps(selector,sort_keys=True,ensure_ascii=False)
            facts=query(container,f"SELECT coalesce(jsonb_agg(DISTINCT to_jsonb(v)->{literal(column)}),'[]'::jsonb) FROM public.{view} v WHERE to_jsonb(v)-{literal(column)}={literal(selector_json)}::jsonb AND to_jsonb(v)->{literal(column)} <> 'null'::jsonb")
            facts=[str(v) for v in facts]
            entries.append(dict(question=f'In the DrugCentral snapshot view {view}, give one recorded {column} value for a row matching {selector_json}.',
                reference_facts=facts,reference_answer=facts[0],source_row_key=selector,
                requested_columns=[column],reference_record=row,question_kind='view_record_field'))
        entries.append(dict(question=f'How many rows does view {view} return in the supplied DrugCentral source snapshot?',
            reference_facts=[str(count)],reference_answer=str(count),source_row_key=None,requested_columns=[],
            reference_record={'row_count':count},question_kind='view_snapshot_count'))
        seen=set();unique=[]
        for q in entries:
            if q['question'] in seen:continue
            seen.add(q['question']);unique.append(q)
        for n,q in enumerate(unique,1):
            q.update(id=f'full-{manifest["seed"]}-view-{view}-{n:03d}',category=view,drug=None,
                source_table=view,source_tables=[],source_view=view,source_views=[view],source_object_type='view',
                benchmark_scope='full_source_table_coverage',dataset_version=original[0]['dataset_version'],
                source_url=f'{Path(manifest["source_file"]).name}#public.{view}',section_header=f'public.{view}',
                answer_rule='One listed value suffices for the exact view row filter. Do not substitute the base table: preserve view filtering, joins, grouping and ranking. Counts refer only to the frozen snapshot.',
                reference_sha256=hashlib.sha256(json.dumps(q['reference_facts'],ensure_ascii=False).encode()).hexdigest(),
                review={'status':'pending','reviewer':None})
        new.extend(unique)
        coverage[view]={'object_type':'view','row_count':count,'definition_sql':definition,
            'definition_sha256':hashlib.sha256(definition.encode()).hexdigest(),
            'question_count':len(unique),'question_ids':[q['id'] for q in unique],
            'reference_method':'PostgreSQL evaluation of exact archive view definitions over restored source dependency tables',
            'serving_status':'not directly queried by current serving pipeline; view-result answerability not established'}
    questions=rebalance(original,new)
    for q in questions:q.setdefault('source_object_type','table')
    table_coverage=json.loads((base/'table_coverage.json').read_text())
    for table,row in table_coverage.items():
        ids=[q['id'] for q in questions if q['source_object_type']=='table' and q['source_table']==table]
        assert ids;row.update(question_count=len(ids),question_ids=ids)
    assert len(questions)==len(original)
    assert len({q['id'] for q in questions})==len(questions)
    assert len({q['question'] for q in questions})==len(questions)
    output.mkdir(parents=True)
    encoded=json.dumps(questions,ensure_ascii=False,indent=2)+'\n'
    (output/'questions.json').write_text(encoded)
    (output/'table_coverage.json').write_text(json.dumps(table_coverage,indent=2)+'\n')
    (output/'view_coverage.json').write_text(json.dumps(coverage,indent=2)+'\n')
    (output/'object_coverage.json').write_text(json.dumps({**{k:dict(v,object_type='table') for k,v in table_coverage.items()},**coverage},indent=2)+'\n')
    manifest.update(source_views=len(VIEWS),covered_views=len(coverage),source_objects=len(table_coverage)+len(VIEWS),
        covered_objects=len(table_coverage)+len(coverage),object_coverage_percent=100,
        questions_by_table={k:v['question_count'] for k,v in table_coverage.items()},
        questions_by_view={k:v['question_count'] for k,v in coverage.items()},
        questions_sha256=hashlib.sha256(encoded.encode()).hexdigest(),
        coverage_definition='At least one exact-record/result or frozen-count question per public source table AND view. Not exhaustive row/column coverage or proof of serving ingestion.',
        view_sampling='Up to four deterministic result rows with a non-null tested field, plus a view row count; all matching answers retained for each row filter.',
        parent_questions_sha256=json.loads((base/'manifest.json').read_text())['questions_sha256'])
    manifest['source_provenance']['archive_views']=['public.'+v for v in VIEWS]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'questions':len(questions),'tables':len(table_coverage),'views':len(coverage),'view_questions':len(new),'view_rows':{v:r['row_count'] for v,r in coverage.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--container',required=True)
    a=p.parse_args();build(a.base,a.output,a.container)
