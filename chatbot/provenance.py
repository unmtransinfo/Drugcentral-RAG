"""Trace derived evidence back to the authoritative DrugCentral source tables.

The source database is the source of truth; vector chunks, FTS indexes, graph edges and
caches are derived representations. The map mirrors the aggregation queries in
ingestion/vectors.py, so a retrieved chunk names the tables and key it was built from.
"""

SECTION_SOURCE_TABLES = {
    'properties': ['structures'],
    'synonyms': ['synonyms', 'product', 'active_ingredient'],
    'indications': ['omop_relationship'],
    'contraindications': ['omop_relationship'],
    'targets': ['act_table_full', 'target_dictionary'],
    'approvals': ['approval'],
    'pharmacology': ['pharma_class'],
    'interactions': ['ddi', 'drug_class', 'struct2drgclass'],
    'adme': ['pka', 'property', 'property_type'],
    'atc': ['struct2atc', 'atc'],
    'patents': ['struct2obprod', 'ob_product', 'ob_patent', 'ob_exclusivity'],
    'structures': ['pdb'],
    'adverse_events': ['faers'],
    # Full-database coverage sections (ingestion/full_coverage.py)
    'ddd': ['atc_ddd'], 'identifiers': ['identifier', 'id_type', 'inxight_drugs'],
    'structure_info': ['structure_type', 'struct_type_def', 'struct2parent', 'parentmol'],
    'pk_measurements': ['pk_az'], 'veterinary': ['humanim', 'vetomop', 'vetprod', 'vetprod2struct', 'vetprod_type', 'vettype'],
    'adverse_events_male': ['faers_male'], 'adverse_events_female': ['faers_female'],
    'adverse_events_pediatric': ['faers_ped'], 'adverse_events_geriatric': ['faers_ger'],
    'adverse_events_top': ['faers_top'], 'patent_details': ['ob_patent_view', 'ob_exclusivity_view'],
    'doid_mapping': ['omop_relationship_doid_view'], 'lincs': ['lincs_signature'],
    'target_profile': ['target_dictionary', 'td2tc', 'target_component', 'tdgo2tc', 'tdkey2tc', 'act_table_full'],
    'ontology': ['target_go', 'target_keyword'], 'disease_profile': ['doid', 'doid_xref'],
    'references': ['reference', 'ref_type'], 'dictionary': ['dictionary tables'],
}
# Product-label sections come from the label extract of the same source archive.
LABEL_SOURCE_TABLES = ['label', 'section', 'prd2label', 'product', 'active_ingredient']
LABEL_SECTIONS = {'adverse_reactions', 'warnings', 'precautions', 'dosage', 'populations'}


def provenance(metadata):
    section = metadata.get('section')
    # Chunks built by ingestion/full_coverage.py name their exact source tables.
    tables = metadata.get('source_tables') or (LABEL_SOURCE_TABLES if metadata.get('source_type') == 'product_label' or section in LABEL_SECTIONS
              else SECTION_SOURCE_TABLES.get(section, []))
    key = {'struct_id': metadata.get('struct_id')}
    if metadata.get('entity_type'):
        key = {'entity_type': metadata['entity_type'], 'entity_id': metadata.get('entity_id')}
    return {'source_tables': list(tables), 'source_primary_key': key,
            'source_id': metadata.get('source_id'), 'dataset_version': metadata.get('dataset_version'),
            'section': section, 'source_type': metadata.get('source_type')}
