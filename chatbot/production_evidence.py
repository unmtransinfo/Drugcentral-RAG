"""Immutable, content-addressed evidence and inexpensive claim safeguards."""
import hashlib
import json
import re
from decimal import Decimal
from dataclasses import dataclass
from chatbot.provenance import provenance


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source_id: str
    text: str
    metadata_json: str
    semantic_score: float | None = None
    lexical_score: float | None = None
    rerank_score: float | None = None

    @property
    def metadata(self):
        return json.loads(self.metadata_json)

    def as_dict(self):
        m=self.metadata
        return dict(evidence_id=self.evidence_id,source_id=self.source_id,text=self.text,
                    drug=m.get('drug_name'),section=m.get('section'),url=m.get('source_url'),
                    source_type=m.get('source_type'),route=m.get('route'),form=m.get('form'),
                    truncated=bool(m.get('truncated')),metadata=m,semantic_score=self.semantic_score,lexical_score=self.lexical_score,
                    rerank_score=self.rerank_score,provenance=provenance(m))


def evidence_from_record(record):
    m=dict(record['metadata'])
    source_id=m.get('source_id') or m.get('source_url')
    if not source_id:
        raise ValueError('Authoritative source identity is required')
    text=record['text']
    if not isinstance(text,str) or not text.strip():
        raise ValueError('Empty evidence')
    identity=json.dumps([source_id,m.get('dataset_version'),m.get('section'),text],ensure_ascii=False,separators=(',',':'))
    return Evidence('ev_'+hashlib.sha256(identity.encode()).hexdigest(),str(source_id),text,
        json.dumps(m,sort_keys=True),record.get('semantic_score'),record.get('lexical_score'),record.get('rerank_score'))


def source_quote(evidence,start=0,end=None):
    end=len(evidence.text) if end is None else end
    if type(start) is not int or type(end) is not int or not 0<=start<end<=len(evidence.text):
        raise ValueError('Invalid quote offsets')
    return evidence.text[start:end]


NUMBERS=re.compile(r'(?<![A-Za-z0-9])[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|(?<![A-Za-z0-9])\.\d+')
UNITS=re.compile(r'(?<![A-Za-z])(?:mg|mcg|µg|μg|g|kg|ml|mL|L|mmol|mol|nm|nM|µM|μM|IU|mEq|hours?|days?|weeks?)\b')
CODES=re.compile(r'\b(?:[A-Z][A-Z0-9-]{1,}|[A-Za-z]*\d+[A-Za-z0-9-]*)\b')
ROUTES={'oral','intravenous','intramuscular','subcutaneous','topical','ophthalmic','otic','inhalation','rectal','vaginal','transdermal','intrathecal'}
SPECIES={'human','mouse','rat','rabbit','canine','bovine','porcine','homo sapiens','mus musculus','rattus norvegicus'}


def deterministic_status(claim,evidence_by_id,index=None):
    if not isinstance(claim,dict) or set(claim)-{'claim_id','text','evidence_ids'}:
        return 'VALIDATION_ERROR'
    text=claim.get('text'); ids=claim.get('evidence_ids')
    if not isinstance(text,str) or not text.strip() or len(text)>1200 or re.search(r'https?://|www\.',text):
        return 'VALIDATION_ERROR'
    if not isinstance(ids,list) or not ids or not all(isinstance(i,str) and i in evidence_by_id for i in ids):
        return 'NO_SUPPORTING_EVIDENCE'
    sources=[evidence_by_id[i] for i in ids]
    if any(not e.source_id for e in sources):
        return 'NO_SUPPORTING_EVIDENCE'
    # A single atomic relation must have a single source: disallow stitching a
    # value from one drug to a property from another.
    if len(set(ids))!=1:
        return 'INSUFFICIENT_EVIDENCE'
    source=sources[0].text
    metadata=sources[0].metadata
    context=source+' '+str(metadata.get('drug_name',''))+' '+str(metadata.get('product_name',''))
    # Isotope digits inside an authoritative name are identifiers, not doses.
    measurement_text=text
    for field in ('drug_name','product_name'):
        name=metadata.get(field)
        if name:
            measurement_text=re.sub(r'(?<!\w)'+re.escape(str(name))+r'(?!\w)','',measurement_text,flags=re.I)
    def numeric_values(value):
        return {Decimal(n.replace(',','')) for n in NUMBERS.findall(value)}
    if numeric_values(measurement_text)-numeric_values(source):
        return 'NUMERIC_MISMATCH'
    if set(UNITS.findall(measurement_text))-set(UNITS.findall(source)):
        return 'UNIT_MISMATCH'
    # Check quantities together, preserving compound units and numeric equivalence.
    quantity=re.compile(r'([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(mg|mcg|g|kg|ml|mL|mmol|IU|mEq)(?:\s*/\s*(kg|m2|m²|day|h|ml|mL))?\b')
    def quantities(value):
        return {(Decimal(n.replace(',','')),unit+('/'+denominator if denominator else ''))
                for n,unit,denominator in quantity.findall(value)}
    if quantities(measurement_text)-quantities(source):
        return 'NUMERIC_MISMATCH'
    for frequency in ('once daily','twice daily','three times daily','four times daily'):
        if frequency in text.lower() and frequency not in source.lower():
            return 'DOSAGE_MISMATCH'
    for code in CODES.findall(text):
        code=code.rstrip('-')
        if code in {'A','I','The','Drug'} or code.isdigit() or re.fullmatch(r'\d+(?:mg|mcg|g|kg|ml|mL|mmol|IU|mEq)',code):
            continue
        if not re.search(r'(?<!\w)'+re.escape(code)+r'(?!\w)',context):
            return 'ENTITY_MISMATCH'
    def terms(value,words):
        return {word for word in words if re.search(r'(?<!\w)'+re.escape(word)+r'(?!\w)',value,re.I)}
    if terms(text,ROUTES)-terms(context+' '+str(metadata.get('route','')),ROUTES):
        return 'ENTITY_MISMATCH'
    if terms(text,SPECIES)-terms(context,SPECIES):
        return 'ENTITY_MISMATCH'
    if index is not None:
        for name in index.mentions(text):
            if name.casefold() not in context.casefold():
                return 'ENTITY_MISMATCH'
    from chatbot.vector_answers import contradicts_quote
    normalized_source=re.sub(r'\b(?:failed to|did not|absence of|excluded)\b','not',source,flags=re.I)
    normalized_claim=re.sub(r'\b(?:failed to|did not|absence of|excluded)\b','not',text,flags=re.I)
    if metadata.get('section')=='contraindications':
        normalized_claim=re.sub(r'\bcontraindicated\b','',normalized_claim,flags=re.I)
        normalized_source=re.sub(r'\bcontraindicated\b','',normalized_source,flags=re.I)
    if contradicts_quote(normalized_claim,normalized_source):
        return 'NEGATION_MISMATCH'
    low=source.casefold(); lower=text.casefold()
    if 'off-label' in low and re.search(r'\bapproved\b',lower) and not re.search(r'\bapproved\b',low):
        return 'CONTRADICTION'
    if metadata.get('section') in {'warnings','precautions'} and 'contraindicat' in lower and 'contraindicat' not in low:
        return 'CONTRADICTION'
    if ('associated with' in low or 'may ' in low or 'evaluated' in low) and re.search(r'\b(causes|cures|approved)\b',lower) and not re.search(r'\b(causes|cures|approved)\b',low):
        return 'INSUFFICIENT_EVIDENCE'
    codes=set(CODES.findall(measurement_text))
    # Symbols joined in a single receptor-complex bullet can describe one target.
    complex_target=metadata.get('section')=='targets' and any(
        '|' in line and '(' in line and codes <= set(CODES.findall(line))
        for line in source.splitlines() if re.match(r'\s*-\s',line))
    if re.search(r'\b(?:targets?|inhibits?|binds?)\b',text,re.I) and re.search(r',|\band\b',text,re.I) and len(codes)>1 and not complex_target:
        return 'NON_ATOMIC_CLAIM'
    if metadata.get('section')=='targets':
        # Target membership does not establish an action mechanism.
        for stem in ('inhibit','agonis','antagonis','activat'):
            if stem in text.lower() and stem not in source.lower():
                return 'INSUFFICIENT_EVIDENCE'
        target=re.search(r'\b(?:targets|binds to|inhibits)\s+(.+?)[.!]?$' ,text,re.I)
        if target:
            allowed=set(re.findall(r'[a-z0-9]+',context.lower()))|{'the','a','an','protein','receptor','with','gene','genes','symbol','symbols','subunit','subunits','and','of','include','includes','recorded','for','biological'}
            if set(re.findall(r'[a-z0-9]+',target[1].lower()))-allowed:
                return 'ENTITY_MISMATCH'
    # A list of effects/uses/targets is several independently testable claims.
    enumerated=re.search(r'\b(?:are|includes?|comprises?|consists? of)\s+(.+)',text,re.I)
    if enumerated and re.search(r',|;|\band\b',enumerated[1],re.I):
        return 'NON_ATOMIC_CLAIM'
    # Explicit multi-sentence/multi-relation output must be repaired into atoms.
    if re.search(r'[.!?]\s+[A-Z]|\b(?:and|;)\s+(?:it\s+)?(?:inhibits|targets|causes|treats|has|is|binds)\b',text):
        return 'NON_ATOMIC_CLAIM'
    return 'PENDING_ENTAILMENT'
