"""Backfill ingredient alias provenance from a DrugCentral SQL dump.

Run with --apply to update metadata only; embeddings and documents are retained.
"""
import argparse
import gzip
import json
from collections import defaultdict
from contextlib import closing


def read_aliases(path):
    aliases = defaultdict(set)
    names = {}
    table = None
    with gzip.open(path, 'rt') as source:
        for line in source:
            if line.startswith('COPY '):
                table = line.split()[1]
                columns = line.split('(', 1)[1].split(')', 1)[0].split(', ')
                continue
            if line.strip() == '\\.':
                table = None
                continue
            if table not in {'public.synonyms', 'public.structures'}:
                continue
            row = dict(zip(columns, line.rstrip('\n').split('\t')))
            name = row['name'].replace('\\t', '\t').replace('\\n', '\n').replace('\\\\', '\\').strip().lower()
            if name == '\\n' or not name:
                continue
            if not row['id'].isdigit():
                continue
            ident = int(row['id'])
            if table == 'public.structures':
                names[ident] = name
            else:
                aliases[ident].add(name)
    return names, aliases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dump')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    import os
    from chatbot.database_config import connect_database
    from psycopg2.extras import execute_values
    names, aliases = read_aliases(args.dump)
    print('Source acetaminophen mappings:', [names.get(i) for i, values in aliases.items() if 'acetaminophen' in values])
    collections = [('drugcentral_chunks', os.getenv('VECTOR_COLLECTION', 'drugcentral'))]
    if os.getenv('STRUCTURED_COLLECTION'):
        collections.append(('drugcentral_evidence', os.environ['STRUCTURED_COLLECTION']))
    with closing(connect_database()) as conn, conn, conn.cursor() as cur:
        for table, collection in collections:
            cur.execute(f'SELECT DISTINCT metadata->>\'struct_id\', metadata->>\'drug_name\' FROM public.{table} WHERE collection=%s', (collection,))
            updates = []
            for ident, drug in cur.fetchall():
                ident = int(ident)
                if names.get(ident) != drug.strip().lower():
                    raise ValueError(f'Source identity mismatch for struct_id {ident}; refusing migration')
                updates.append((json.dumps(sorted(aliases[ident])), collection, str(ident)))
            print(table, collection, 'drug identities verified:', len(updates))
            if args.apply:
                cur.execute("CREATE TEMP TABLE alias_repair (aliases text, collection text, struct_id text) ON COMMIT DROP")
                execute_values(cur, "INSERT INTO alias_repair VALUES %s", updates)
                cur.execute(f"UPDATE public.{table} AS target SET metadata=jsonb_set(target.metadata, '{{ingredient_synonyms_json}}', to_jsonb(repair.aliases)) FROM alias_repair AS repair WHERE target.collection=repair.collection AND target.metadata->>'struct_id'=repair.struct_id")
                cur.execute("DROP TABLE alias_repair")
    print('Metadata updated.' if args.apply else 'Dry run; pass --apply to update metadata.')


if __name__ == '__main__':
    main()
