"""Relationship and multi-hop questions answered from the derived Neo4j graph.

PostgreSQL stays authoritative: the graph is rebuilt from it (ingestion/graph_build.py) and every
edge keeps its source table and primary key. A question is first reduced to a small constrained
intent; only fixed, parameterised Cypher templates run, never model-written Cypher. Answer text is
composed deterministically from graph rows, so no unsupported facts can be introduced.
"""
import os
import re

MAX_ROWS = 500
SHOW_ROWS = 20
INTENTS = {'drugs_by_target', 'shared_target_drugs', 'same_atc_drugs', 'atc_for_target', 'interactions_of'}
HOPS = {'drugs_by_target': 1, 'shared_target_drugs': 2, 'same_atc_drugs': 2, 'atc_for_target': 3, 'interactions_of': 2}
ORGANISMS = {'human': 'homo sapiens', 'mouse': 'mus musculus', 'rat': 'rattus norvegicus'}
CAVEAT = 'These are recorded database relationships, not treatment recommendations or an exhaustive clinical comparison.'

# Each template is fixed text; user values only ever travel as parameters.
TARGET_MATCH = ('(toLower(t.name) = $target OR EXISTS { (t)-[:ENCODED_BY]->(g:Gene) WHERE g.symbol_lower = $target })')
TEMPLATES = {
    'drugs_by_target': f'''
        MATCH (d:Drug)-[r:TARGETS]->(t:Target)
        WHERE {TARGET_MATCH} AND ($organism IS NULL OR toLower(r.organism) = $organism)
          AND ($moa IS NULL OR r.moa = $moa)
        OPTIONAL MATCH (d)-[i:INDICATED_FOR]->(dis:Disease)
          WHERE $disease IS NOT NULL AND dis.name_lower CONTAINS $disease
        WITH d, t, r, collect(DISTINCT dis.name)[..3] AS diseases, count(dis) AS matched
        WHERE $disease IS NULL OR matched > 0
        RETURN d.name AS drug, d.struct_id AS struct_id, t.name AS target, r.organism AS organism,
               r.action_type AS action, r.source_table AS source_table, r.source_pk AS source_pk, diseases
        ORDER BY drug, target LIMIT $limit''',
    'shared_target_drugs': '''
        MATCH (d:Drug {struct_id: $struct_id})-[:TARGETS]->(t:Target)<-[r:TARGETS]-(o:Drug)
        WHERE o.struct_id <> d.struct_id
        WITH o, collect(DISTINCT t.name) AS shared, collect(DISTINCT r.source_pk)[..3] AS pks
        RETURN o.name AS drug, o.struct_id AS struct_id, shared AS targets, 'act_table_full' AS source_table, pks AS source_pk
        ORDER BY size(shared) DESC, drug LIMIT $limit''',
    'same_atc_drugs': '''
        MATCH (d:Drug {struct_id: $struct_id})-[:HAS_ATC]->(a:ATC)
        MATCH (o:Drug)-[:HAS_ATC]->(b:ATC)
        WHERE o.struct_id <> d.struct_id AND substring(b.code, 0, $length) = substring(a.code, 0, $length)
        RETURN o.name AS drug, o.struct_id AS struct_id, collect(DISTINCT b.code)[..3] AS codes,
               collect(DISTINCT a.code)[..3] AS reference_codes, 'struct2atc' AS source_table
        ORDER BY drug LIMIT $limit''',
    'atc_for_target': f'''
        MATCH (d:Drug)-[:TARGETS]->(t:Target), (d)-[:HAS_ATC]->(a:ATC)
        WHERE {TARGET_MATCH}
        WITH substring(a.code, 0, 5) AS subgroup, a.l4_name AS name, collect(DISTINCT d.name) AS drugs
        RETURN subgroup, name, size(drugs) AS drug_count, drugs[..5] AS examples
        ORDER BY drug_count DESC, subgroup LIMIT $limit''',
    'interactions_of': '''
        MATCH (d:Drug {struct_id: $struct_id})-[:IN_CLASS]->(c:DrugClass)-[x:INTERACTS_WITH]-(o:DrugClass)
        RETURN c.name AS drug_class, o.name AS interacting_class, x.risk AS risk,
               x.description AS description, x.source_pk AS source_pk, 'ddi' AS source_table
        ORDER BY CASE x.risk WHEN 'Contraindicated' THEN 0 WHEN 'Critical' THEN 1 WHEN 'Avoid combination' THEN 2 WHEN 'Significant' THEN 3 ELSE 4 END,
                 interacting_class LIMIT $limit''',
}

_WORD = r"[A-Za-z0-9][A-Za-z0-9 \-/+(),']*?"
PATTERNS = [
    ('drugs_by_target', re.compile(
        rf'(?:which|what) drugs (?:are indicated for (?P<disease2>{_WORD}) and )?(?:target|act on|bind(?: to)?|inhibit)s? '
        rf'(?:the )?(?:(?P<species>human|mouse|rat) )?(?P<target>{_WORD})(?: (?:target|receptor protein))?'
        rf'(?: (?:and|that|which) (?:are )?(?:indicated for|used (?:for|to treat)|treat) (?P<disease>{_WORD}))?', re.I)),
    # "association" wordings. A bare "associated with X" may name a disease instead of a target, so those are marked loose
    # (no "target" noun) and fall back to normal retrieval when the graph finds nothing.
    ('drugs_by_target', re.compile(
        rf'(?:which|what) drugs (?:are|is) (?:known to be )?associated with (?:the )?(?:(?P<species>human|mouse|rat) )?(?P<target>{_WORD})(?P<noun> targets?)?'
        rf'(?: (?:and|that|which) (?:are )?(?:indicated for|used (?:for|to treat)|treat) (?P<disease>{_WORD}))?', re.I)),
    ('drugs_by_target', re.compile(
        rf'(?:which|what) drugs have (?:the )?(?:(?P<species>human|mouse|rat) )?(?P<target>{_WORD}) as (?:an? |their )?(?P<moa>mechanism[- ]of[- ]action |moa )?(?:biological )?targets?', re.I)),
    ('drugs_by_target', re.compile(
        rf'(?:list|show|name|give me)(?: me)?(?: all)?(?: the)? drugs (?:that are )?(?:associated with|that target|targeting) (?:the )?(?:(?P<species>human|mouse|rat) )?(?P<target>{_WORD})(?P<noun> targets?)?', re.I)),
    ('drugs_by_target', re.compile(
        rf'(?:what|which) are the drug[- ]target associations (?:for|of) (?:the )?(?:(?P<species>human|mouse|rat) )?(?P<target>{_WORD})', re.I)),
    ('shared_target_drugs', re.compile(
        rf'(?:which|what) (?:other )?drugs (?:share|have)(?: (?:a|the same|any))? targets? (?:in common )?(?:with|as) (?P<drug>{_WORD})', re.I)),
    ('same_atc_drugs', re.compile(
        rf'(?:which|what) (?:other )?drugs (?:are in|share|belong to|have) the same ATC (?:class|group|subgroup|level ?(?P<level>[1-4]))'
        rf'(?: as| with)? (?P<drug>{_WORD})', re.I)),
    ('atc_for_target', re.compile(
        rf'(?:which|what) ATC (?:classes|classifications|codes|groups) (?:do )?drugs (?:that )?(?:target|targeting|act on) '
        rf'(?:the )?(?P<target>{_WORD}) (?:have|belong to|fall into)', re.I)),
    ('interactions_of', re.compile(
        rf'(?:which|what) (?:drug )?(?:classes|drugs) (?:interact|have interactions) with (?P<drug>{_WORD})', re.I)),
]
ATC_LENGTH = {None: 5, '1': 1, '2': 3, '3': 4, '4': 5}


def _clean(value):
    value = re.sub(r'\s+in drugcentral\s*$', '', (value or '').strip(' ?.!'), flags=re.I)
    return re.sub(r'\s+', ' ', value).strip()


def parse_graph_query(question):
    """Reduce a question to a constrained intent, or None when it is not a graph question."""
    text = re.sub(r'\s+in drugcentral\s*[?.!]*$', '', question.strip(), flags=re.I).rstrip('?.! ')
    for intent, pattern in PATTERNS:
        match = pattern.fullmatch(text)
        if not match:
            continue
        groups = match.groupdict()
        query = {'intent': intent, 'hops': HOPS[intent]}
        if 'target' in groups:
            query['target_text'] = _clean(groups['target'])
            query['target'] = query['target_text'].casefold()
            # Dependent chains ("proteins encoded by genes associated with ...") are not templates.
            if not query['target'] or re.search(r'\b(?:encoded|associated|pathways?|proteins|genes)\b', query['target']):
                continue
            query['organism'] = ORGANISMS.get((groups.get('species') or '').casefold())
            query['moa'] = 1 if groups.get('moa') or re.search(r'mechanism[- ]of[- ]action|\bmoa\b', text, re.I) else None
            # Loose: an "associated with X" wording that never says "target", so X may be a disease or something else.
            query['loose'] = 'noun' in groups and not groups.get('noun')
            disease = _clean(groups.get('disease') or groups.get('disease2'))
            query['disease'] = disease.casefold() or None
        if 'drug' in groups:
            query['drug'] = _clean(groups['drug'])
            if not query['drug']:
                continue
        if intent == 'same_atc_drugs':
            query['length'] = ATC_LENGTH[groups.get('level')]
        return query
    return None


def graph_enabled():
    return os.getenv('GRAPH_ENABLED', '0') == '1'


class Graph:
    """Read-only Neo4j access; every call runs a fixed template inside a read transaction."""

    def __init__(self, driver=None):
        if driver is None:
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(os.environ['NEO4J_URI'], auth=(
                os.getenv('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD']))
        self.driver = driver
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')

    def run(self, intent, parameters):
        if intent not in TEMPLATES:
            raise ValueError('Unknown graph template')
        from neo4j import unit_of_work
        @unit_of_work(timeout=float(os.getenv('GRAPH_TIMEOUT_SECONDS', '15')))
        def work(tx):
            return [dict(r) for r in tx.run(TEMPLATES[intent], parameters)]
        with self.driver.session(database=self.database, default_access_mode='READ') as session:
            return session.execute_read(work)

    def dataset_version(self):
        with self.driver.session(database=self.database, default_access_mode='READ') as session:
            row = session.execute_read(lambda tx: tx.run('MATCH (m:GraphMeta) RETURN m.dataset_version AS v LIMIT 1').single())
            return row['v'] if row else None

    def close(self):
        self.driver.close()


def card(struct_id):
    return f'https://drugcentral.org/drugcard/{struct_id}' if struct_id is not None else None


def _fact(drug, struct_id, text, section, table, pk):
    pk = ','.join(map(str, pk)) if isinstance(pk, list) else pk
    return {'drug': drug, 'text': text, 'url': card(struct_id), 'section': section, 'source_type': 'graph',
            'provenance': {'source_tables': [table] if table else [], 'source_primary_key': pk, 'struct_id': struct_id}}


def render(query, rows, resolved=None):
    """Return (answer text, evidence facts). Composition is deterministic."""
    intent = query['intent']
    if not rows:
        scope = {'drugs_by_target': 'the requested target' + (' and indication' if query.get('disease') else ''),
                 'shared_target_drugs': f"drugs sharing a target with {resolved}",
                 'same_atc_drugs': f"drugs in the same ATC group as {resolved}",
                 'atc_for_target': 'ATC groups of drugs targeting that protein',
                 'interactions_of': f"interaction records for {resolved}"}[intent]
        return (f'No matching records for {scope} were found in the loaded DrugCentral relationship graph. '
                'This is not proof that no such relationship exists.'), []
    facts, lines, sources = [], [], []
    shown = rows[:SHOW_ROWS]
    def cite(fact):
        url = fact['url']
        if url and url not in sources:
            sources.append(url)
        return f'[{sources.index(url) + 1}]' if url in sources else ''
    if intent == 'drugs_by_target':
        names = sorted({r['drug'] for r in rows})
        header = f"DrugCentral records {len(names)} drug(s) with a relationship to the {query['target_text']} target"
        if query.get('disease'):
            header += f" and an indication matching “{query['disease']}”"
        if query.get('moa'):
            header += ' (mechanism-of-action targets only)'
        header += ':'
        for r in shown:
            organism = f"; organism: {r['organism']}" if r.get('organism') else ''
            action = f"; action: {r['action']}" if r.get('action') else ''
            extra = f"; indication match: {', '.join(r['diseases'])}" if r.get('diseases') else ''
            fact = _fact(r['drug'], r['struct_id'], f"{r['drug']} targets {r['target']}{organism}{action}{extra}", 'targets',
                         r['source_table'], r['source_pk'])
            facts.append(fact); lines.append(f"- {r['drug']} — {r['target']}{organism}{action}{extra} {cite(fact)}")
    elif intent == 'shared_target_drugs':
        header = f'DrugCentral records {len(rows)} other drug(s) sharing at least one target with {resolved}:'
        for r in shown:
            fact = _fact(r['drug'], r['struct_id'], f"{r['drug']} shares targets with {resolved}: {', '.join(r['targets'])}",
                         'targets', r['source_table'], r['source_pk'])
            facts.append(fact); lines.append(f"- {r['drug']} — shared: {', '.join(r['targets'][:5])}"
                                              f"{' …' if len(r['targets']) > 5 else ''} {cite(fact)}")
    elif intent == 'same_atc_drugs':
        header = f'DrugCentral records {len(rows)} other drug(s) in the same ATC group as {resolved}:'
        for r in shown:
            fact = _fact(r['drug'], r['struct_id'], f"{r['drug']} has ATC {', '.join(r['codes'])}", 'atc', r['source_table'], None)
            facts.append(fact); lines.append(f"- {r['drug']} — ATC {', '.join(r['codes'])} {cite(fact)}")
    elif intent == 'atc_for_target':
        header = f"ATC subgroups of drugs recorded as targeting {query['target_text']}:"
        for r in shown:
            examples = ', '.join(r['examples'])
            fact = _fact(None, None, f"ATC {r['subgroup']} ({r['name']}): {r['drug_count']} drug(s), e.g. {examples}", 'atc', 'struct2atc', None)
            facts.append(fact); lines.append(f"- {r['subgroup']} {r['name'] or ''} — {r['drug_count']} drug(s), e.g. {examples}")
    else:
        header = f'DrugCentral records {len(rows)} interaction record(s) at the drug-class level for {resolved}:'
        for r in shown:
            description = f": {r['description']}" if r.get('description') else ''
            fact = _fact(resolved, None, f"{r['drug_class']} interacts with {r['interacting_class']} [{r['risk']}]{description}",
                         'interactions', r['source_table'], r['source_pk'])
            facts.append(fact); lines.append(f"- {r['drug_class']} × {r['interacting_class']} [{r['risk']}]{description}")
    text = header + '\n\n' + '\n'.join(lines)
    if len(rows) > SHOW_ROWS:
        text += f'\n\nPartial list: showing {SHOW_ROWS} of {len(rows)} matching records.'
    if sources:
        text += '\n\nSources: ' + ' · '.join(f'[{n}] [DrugCentral drug card]({u})' for n, u in enumerate(sources, 1))
    return text + '\n\n' + CAVEAT, facts


def resolve_drug(index, query, question):
    """Resolve the named drug exactly; an ambiguous or unknown name never guesses."""
    from chatbot.drug_resolution import resolve_question
    drugs, clarification = resolve_question(index, question)
    if clarification:
        return None, clarification
    drug = query['drug'].casefold()
    for candidate in drugs:
        if candidate.casefold() == drug or drug in candidate.casefold():
            return candidate, None
    return None, None


def answer_graph(index, question, graph=None, metrics=None):
    """Return None when the question is not a graph question or the graph is unavailable.

    Otherwise a dict with text, status, evidence (fact dicts) and a trace entry.
    """
    query = parse_graph_query(question)
    if query is None:
        return None
    parameters = {'limit': MAX_ROWS}
    resolved = None
    if 'drug' in query:
        resolved, clarification = resolve_drug(index, query, question)
        if clarification:
            return {'text': clarification, 'status': 'CLARIFICATION', 'evidence': [], 'trace': {'query': query}}
        if resolved is None:
            return None
        parameters['struct_id'] = index.ids[resolved]
    for key in ('target', 'organism', 'disease', 'length', 'moa'):
        if key in query:
            parameters[key] = query[key]
    parameters.setdefault('organism', None)
    parameters.setdefault('disease', None)
    parameters.setdefault('moa', None)
    try:
        graph = graph or Graph()
        rows = graph.run(query['intent'], parameters)
    except Exception as exc:
        # An unavailable derived index must not block the verified RAG path.
        if metrics is not None:
            metrics['graph_error_type'] = type(exc).__name__
        return None
    if metrics is not None:
        metrics['graph_rows'] = len(rows)
    if not rows and query.get('loose'):
        return None          # "associated with X" found no such target: X may be a disease or a drug, so let retrieval answer
    text, facts = render(query, rows, resolved)
    return {'text': text, 'status': 'SUPPORTED' if rows else 'NO_GRAPH_MATCH', 'evidence': facts,
            'trace': {'query': query, 'rows': len(rows), 'truncated': len(rows) >= MAX_ROWS}}
