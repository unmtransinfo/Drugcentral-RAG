"""Validate an ingestion preview and optionally publish a new evidence snapshot."""
import argparse
import json
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from chatbot.retrieval import EvidenceIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='benchmarks/ingestion-preview.jsonl')
    parser.add_argument('--collection')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    records = [json.loads(line) for line in Path(args.input).open()]
    data = {'documents': [r['text'] for r in records], 'metadatas': [r['metadata'] for r in records]}
    versions = {meta['dataset_version'] for meta in data['metadatas']}
    if len(versions) != 1 or not records:
        raise ValueError('Expected one nonempty dataset version')
    index = EvidenceIndex(data)
    headers = Counter(r['metadata']['section_header'] for r in records)
    if 'Approved Indications' in headers:
        raise ValueError('Ambiguous legacy indication labels found')
    if not headers['Indications'] or not headers['Off-label Uses']:
        raise ValueError('Missing indication relationship sections')
    print(json.dumps({'records': len(records), 'drugs': len(index.ids), 'versions': sorted(versions),
                      'sections': dict(headers)}, indent=2), flush=True)
    if args.publish:
        if not args.collection:
            parser.error('--collection is required for publishing')
        load_dotenv()
        from chatbot.storage import publish_structured_snapshot
        publish_structured_snapshot(records, args.collection)
        print(f'Published {args.collection}; activation requires STRUCTURED_COLLECTION={args.collection}.')


if __name__ == '__main__':
    main()
