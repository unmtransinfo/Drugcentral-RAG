"""Model-independent evidence filtering and lexical/vector rank fusion.

Inputs must already be restricted to one resolved drug and ordered by vector
similarity. Scores are ranking signals, not probabilities or clinical confidence.
"""
from collections import Counter
import math
import re
from chatbot.retrieval import EMPTY, SECTION_HEADERS, NAME_STOPWORDS, section_blocks, tokens
from chatbot.question_retrieval import requested_sections


def evidence_sections(candidate):
    meaningful = []
    for header, body in section_blocks(candidate['text']):
        facts = [re.sub(r'^[-*•]\s*', '', line.strip()).strip().rstrip('.;')
                 for line in body.splitlines()]
        if any(fact and fact.lower() not in EMPTY for fact in facts):
            meaningful.append(header.rsplit(' — ', 1)[-1])
    if not meaningful:
        return set()
    return {candidate['metadata'].get('section')} | {SECTION_HEADERS.get(label) for label in meaningful}


def lexical_scores(candidates, query):
    # Drug names occur in every v3 heading and provide no within-drug discrimination.
    ignored = set(NAME_STOPWORDS)
    for item in candidates:
        ignored.update(tokens(item['metadata'].get('drug_name', '')))
    terms = tokens(query) - ignored
    documents = [Counter(re.findall(r'[a-z0-9]+', item['text'].lower())) for item in candidates]
    lengths = [sum(doc.values()) for doc in documents]
    average = sum(lengths) / max(1, len(lengths)) or 1
    frequency = Counter(term for doc in documents for term in terms if term in doc)
    scores = []
    for doc, length in zip(documents, lengths):
        score = 0.0
        for term in terms:
            tf = doc[term]
            if tf:
                idf = math.log(1 + (len(documents) - frequency[term] + .5) / (frequency[term] + .5))
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / average))
        scores.append(score)
    return scores


def rank_candidates(candidates, query, limit, mode='section'):
    if mode not in {'raw', 'section', 'hybrid'}:
        raise ValueError('Ranking must be raw, section or hybrid')
    if mode == 'raw':
        return candidates[:limit]
    useful, sections, seen = [], [], set()
    for item in candidates:
        labels = evidence_sections(item)
        if not labels:
            continue
        # Keep different facts/continuation parts, remove only exact duplicate text.
        key = (item['metadata'].get('drug_name'), item['text'])
        if mode == 'hybrid' and key in seen:
            continue
        seen.add(key)
        useful.append(item)
        sections.append(labels)
    requested = requested_sections(query)
    if mode == 'section':
        order = sorted(range(len(useful)), key=lambda i: not bool(set(requested) & sections[i]))
    else:
        lexical = lexical_scores(useful, query)
        lexical_order = sorted((i for i in range(len(useful)) if lexical[i] > 0), key=lambda i: (-lexical[i], i))
        lexical_rank = {i: rank for rank, i in enumerate(lexical_order, 1)}
        # Reciprocal-rank fusion keeps inner-product and BM25 score scales separate.
        scores = [1 / (60 + i + 1) + (1 / (60 + lexical_rank[i]) if i in lexical_rank else 0)
                  for i in range(len(useful))]
        order = sorted(range(len(useful)), key=lambda i: (not bool(set(requested) & sections[i]), -scores[i], i))
        if len(requested) > 1:
            selected = []
            for section in requested:
                if any(section in sections[i] for i in selected):
                    continue
                best = next((i for i in order if section in sections[i]), None)
                if best is not None and best not in selected:
                    selected.append(best)
            # Preserve relevance order among selected representatives.
            chosen = set(selected)
            order = [i for i in order if i in chosen] + [i for i in order if i not in chosen]
    return [useful[i] for i in order[:limit]]
