"""Import exact drug-target associations, retaining organism and source identity.

Run: python -m ingestion.relations SOURCE.sql.gz --output data/relations.sqlite
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
from ingestion.labels import dump_rows


def build(source, output):
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    temp=output.with_suffix('.building.sqlite')
    if temp.exists():raise ValueError('Incomplete build already exists: '+str(temp))
    counts=Counter()
    with sqlite3.connect(temp) as c:
        c.executescript('''CREATE TABLE drugs(id INTEGER PRIMARY KEY,name TEXT);
        CREATE TABLE targets(id INTEGER PRIMARY KEY,name TEXT);
        CREATE TABLE associations(id INTEGER PRIMARY KEY,drug_id INTEGER,target_id INTEGER,
            gene TEXT,organism TEXT,action_type TEXT,moa INTEGER,source_url TEXT);
        ''')
        for table,r in dump_rows(source,{'structures','target_dictionary','act_table_full'},counts):
            if table=='structures':c.execute('INSERT INTO drugs VALUES(?,?)',(r['id'],r['name']))
            elif table=='target_dictionary':c.execute('INSERT INTO targets VALUES(?,?)',(r['id'],r['name']))
            else:c.execute('INSERT INTO associations VALUES(?,?,?,?,?,?,?,?)',
                (r['act_id'],r['struct_id'],r['target_id'],r['gene'],r['organism'],r['action_type'],r['moa'],r['act_source_url']))
        c.executescript('CREATE INDEX association_target ON associations(target_id); CREATE INDEX association_gene ON associations(gene,organism);')
        if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Integrity check failed')
        missing=c.execute('SELECT count(*) FROM associations a LEFT JOIN drugs d ON d.id=a.drug_id LEFT JOIN targets t ON t.id=a.target_id WHERE d.id IS NULL OR t.id IS NULL').fetchone()[0]
        if missing:raise ValueError('Unresolved drug/target references: '+str(missing))
        report={'source':Path(source).name,'drugs':c.execute('SELECT count(*) FROM drugs').fetchone()[0],
            'associations':c.execute('SELECT count(*) FROM associations').fetchone()[0],
            'organisms':dict(c.execute('SELECT coalesce(organism,\'unknown\'),count(*) FROM associations GROUP BY organism'))}
    os.replace(temp,output)
    output.with_suffix('.audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source');p.add_argument('--output',default='data/relations.sqlite')
    args=p.parse_args();build(args.source,args.output)
