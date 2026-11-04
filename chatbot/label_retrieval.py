"""Product-scoped label retrieval, separate from FAERS association records."""
import os
from pathlib import Path
import re
import sqlite3


def label_sections(question):
    if re.search(r'\bfaers\b|\breported adverse events\b', question, re.I):
        return []
    sections = []
    for section, cue in [
        ('adverse_reactions', r'side effects?|adverse reactions?|adverse events?'),
        ('warnings', r'warnings?|precautions?|safe|safety|risk'),
        ('contraindications', r'contraindicat|should not take'),
        ('dosage', r'\bdos(?:e|age|ing)\b'),
        ('populations', r'pregnan|breast.?feed|children|pediatric|elderly'),
        ('interactions', r'interact|taken together'),
    ]:
        if re.search(cue, question, re.I):
            sections.append(section)
    if 'adverse_reactions' in sections:
        sections.append('warnings')
    return list(dict.fromkeys(sections))


def database_path():
    return Path(os.getenv('LABEL_DATABASE', str(Path(__file__).resolve().parents[1] / 'data' / 'labels.sqlite')))


def search_labels(index, question, drug, path=None, *, exact_text=False):
    """Return (records, clarification). Never attribute combination labels to one ingredient."""
    sections = label_sections(question)
    path = Path(path) if path else database_path()
    if not sections or not drug or drug not in index.ids or not path.exists():
        return [], None
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        products = conn.execute('''SELECT DISTINCT p.*, l.label_id FROM products p
            JOIN ingredients i ON i.ndc=p.ndc JOIN links l ON l.ndc=p.ndc
            WHERE i.struct_id=? AND p.ingredient_count=1
            AND NOT EXISTS (SELECT 1 FROM ingredients other WHERE other.ndc=p.ndc AND other.struct_id<>i.struct_id)''', (index.ids[drug],)).fetchall()
        requested_routes = []
        for route, cue in [('ORAL', r'\boral\b|\btablets?\b|\bcapsules?\b'),
                           ('INTRAVENOUS', r'\bintravenous\b|\biv\b|\binfusion\b'),
                           ('RECTAL', r'\brectal\b|\bsuppositor'),
                           ('TOPICAL', r'\btopical\b|\bcream\b|\bointment\b'),
                           ('TRANSDERMAL', r'\btransdermal\b|\bpatch\b'),
                           ('OPHTHALMIC', r'\bophthalmic\b|\beye drops\b'),
                           ('INHALATION', r'\binhal(?:ed|ation|er)\b')]:
            if re.search(cue, question, re.I):
                requested_routes.append(route)
        if requested_routes:
            products = [p for p in products if any(r in (p['route'] or '').upper() for r in requested_routes)]
        forms = [f for f in ('tablet', 'capsule', 'solution', 'suspension', 'suppository')
                 if re.search(r'\b' + f + r'(?:s|ies)?\b', question, re.I)]
        if forms:
            products = [p for p in products if any(f in (p['form'] or '').lower() for f in forms)]
        if not products:
            if requested_routes or forms:
                return [], f'No matching single-ingredient label was found for {drug} in the requested formulation. Please provide the exact product name or a different route.'
            return [], None
        routes = sorted({p['route'] or 'unspecified' for p in products})
        if len(routes) > 1 and not requested_routes:
            return [], (f'Which formulation of {drug} do you mean? The available single-ingredient '
                        'labels cover these routes: ' + ', '.join(routes) +
                        '. Please specify the route or dosage form so I can use the matching label.')
        # Avoid arbitrary lexical ranking of product names. Newest label first;
        # retain a single product label so populations/routes cannot be blended.
        product_by_label = {}
        for p in sorted(products, key=lambda row: row['ndc']):
            product_by_label.setdefault(p['label_id'], p)
        candidates = []
        for offset in range(0, len(product_by_label), 400):
            ids = list(product_by_label)[offset:offset + 400]
            marks = ','.join('?' for _ in ids)
            smarks = ','.join('?' for _ in sections)
            candidates.extend(conn.execute(f'''SELECT s.*, l.effective_date, l.title AS label_title
                FROM sections s JOIN labels l ON l.id=s.label_id
                WHERE s.label_id IN ({marks}) AND s.section IN ({smarks})''', ids + sections).fetchall())
        if not candidates:
            return [], None
        # Prefer the primary requested section, then newest supporting label.
        primary = sections[0]
        candidates.sort(key=lambda r: (r['section'] == primary, r['effective_date'] or '', r['label_id']), reverse=True)
        chosen_label = candidates[0]['label_id']
        chosen = [r for r in candidates if r['label_id'] == chosen_label]
        chosen.sort(key=lambda r: (sections.index(r['section']), r['id']))
        records = []
        p = product_by_label[chosen_label]
        for row in chosen[:6]:
            scope = (f"Product label: {p['name']}. Ingredient: {drug}. Route: {p['route']}. "
                     f"Form: {p['form']}. Label effective date: {row['effective_date'] or 'unknown'}.")
            body = re.sub(r'(?<=[.!?])\s+(?=[A-Z0-9])', '\n', row['text'])
            records.append({'text': row['text'] if exact_text else scope + '\n' + row['title'] + ':\n' + body,
                'metadata': {'drug_name': drug, 'struct_id': index.ids[drug], 'section': row['section'],
                    'source_type': 'product_label', 'product_name': p['name'], 'route': p['route'],
                    'form': p['form'], 'label_id': chosen_label,
                    'source_id': f"drugcentral-label:{chosen_label}:section:{row['id']}",
                    'section_title': row['title'],
                    'source_url': 'https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=' + chosen_label,
                    'dataset_version': 'DrugCentral label snapshot; label date ' + (row['effective_date'] or 'unknown')}})
        return records, None
