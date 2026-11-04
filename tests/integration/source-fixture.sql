-- Disposable synthetic source: only the tables and columns ingestion/vectors.py reads, three drugs.
-- Values are illustrative test data, not a copy of DrugCentral.
CREATE TABLE inn_stem (stem text PRIMARY KEY, definition text);
CREATE TABLE structures (id int PRIMARY KEY, name text, cd_formula text, cd_molweight numeric, clogp numeric,
  tpsa numeric, lipinski int, cas_reg_no text, stem text);
CREATE TABLE synonyms (id int, name text);
CREATE TABLE product (ndc_product_code text PRIMARY KEY, product_name text);
CREATE TABLE active_ingredient (ndc_product_code text, struct_id int);
CREATE TABLE omop_relationship (struct_id int, concept_name text, relationship_name text);
CREATE TABLE target_dictionary (id int PRIMARY KEY, name text);
CREATE TABLE act_table_full (struct_id int, target_id int, gene text, act_value numeric, action_type text, moa int);
CREATE TABLE approval (struct_id int, type text, approval date);
CREATE TABLE pharma_class (struct_id int, type text, name text);
CREATE TABLE drug_class (id int PRIMARY KEY, name text);
CREATE TABLE struct2drgclass (struct_id int, drug_class_id int);
CREATE TABLE ddi (drug_class1 text, drug_class2 text, ddi_risk text, description text);
CREATE TABLE pka (struct_id int, pka_type text, value numeric, pka_level text);
CREATE TABLE property_type (id int PRIMARY KEY, name text, symbol text, units text);
CREATE TABLE property (struct_id int, property_type_id int, value numeric);
CREATE TABLE atc (code text PRIMARY KEY, chemical_substance text, l1_name text, l2_name text, l3_name text, l4_name text);
CREATE TABLE struct2atc (struct_id int, atc_code text);
CREATE TABLE ob_product (id int PRIMARY KEY, appl_no text, product_no text);
CREATE TABLE struct2obprod (struct_id int, prod_id int);
CREATE TABLE ob_patent (appl_no text, product_no text, patent_no text, patent_expire_date date);
CREATE TABLE ob_exclusivity (appl_no text, product_no text, exclusivity_code text, exclusivity_date date);
CREATE TABLE pdb (struct_id int, pdb text);
CREATE TABLE faers (struct_id int, meddra_name text, llr numeric, drug_ae int);

INSERT INTO inn_stem VALUES ('-formin', 'antihyperglycaemics, phenformin derivatives'), ('-pril', 'angiotensin-converting enzyme inhibitors');
INSERT INTO structures VALUES
  (1725, 'metformin', 'C4H11N5', 129.17, -1.03, 91.49, 0, '657-24-9', '-formin'),
  (1573, 'lisinopril', 'C21H31N3O5', 405.50, -1.22, 132.96, 0, '76547-98-3', '-pril'),
  (2405, 'warfarin', 'C19H16O4', 308.33, 2.70, 63.60, 0, '81-81-2', NULL);
INSERT INTO synonyms VALUES (1725, 'metformin'), (1725, 'metformin hydrochloride'), (1573, 'lisinopril'), (2405, 'warfarin'), (2405, 'warfarin sodium');
INSERT INTO product VALUES ('0001-0001', 'Glucophage'), ('0001-0002', 'Zestril'), ('0001-0003', 'Coumadin');
INSERT INTO active_ingredient VALUES ('0001-0001', 1725), ('0001-0002', 1573), ('0001-0003', 2405);
INSERT INTO omop_relationship VALUES
  (1725, 'Type 2 diabetes mellitus', 'indication'), (1725, 'Polycystic ovary syndrome', 'off-label use'),
  (1725, 'Metabolic acidosis', 'contraindication'), (1725, 'Severe renal impairment', 'contraindication'),
  (1573, 'Hypertension', 'indication'), (1573, 'Heart failure', 'indication'), (1573, 'Angioedema', 'contraindication'),
  (2405, 'Venous thrombosis', 'indication'), (2405, 'Pulmonary embolism', 'indication'), (2405, 'Pregnancy', 'contraindication');
INSERT INTO target_dictionary VALUES (1, '5''-AMP-activated protein kinase'), (2, 'Angiotensin-converting enzyme'), (3, 'Vitamin K epoxide reductase complex subunit 1');
INSERT INTO act_table_full VALUES (1725, 1, 'PRKAB1', 4.0, 'ACTIVATOR', 1), (1573, 2, 'ACE', 8.7, 'INHIBITOR', 1), (2405, 3, 'VKORC1', 6.5, 'INHIBITOR', 1);
INSERT INTO approval VALUES (1725, 'FDA', '1994-12-29'), (1573, 'FDA', '1987-12-29'), (2405, 'FDA', '1954-06-08');
INSERT INTO pharma_class VALUES (1725, 'EPC', 'Biguanide'), (1573, 'EPC', 'Angiotensin Converting Enzyme Inhibitor'), (2405, 'EPC', 'Vitamin K Antagonist');
INSERT INTO drug_class VALUES (10, 'Biguanides'), (11, 'ACE inhibitors'), (12, 'Coumarin anticoagulants');
INSERT INTO struct2drgclass VALUES (1725, 10), (1573, 11), (2405, 12);
INSERT INTO ddi VALUES ('ACE inhibitors', 'Potassium-sparing diuretics', 'major', 'Risk of hyperkalaemia');
INSERT INTO pka VALUES (1725, 'B', 12.4, 'MOST_BASIC');
INSERT INTO property_type VALUES (1, 'Half-life', 'T1/2', 'h');
INSERT INTO property VALUES (1725, 1, 6.2), (1573, 1, 12.0), (2405, 1, 40.0);
INSERT INTO atc VALUES
  ('A10BA02', 'metformin', 'ALIMENTARY TRACT AND METABOLISM', 'DRUGS USED IN DIABETES', 'BLOOD GLUCOSE LOWERING DRUGS, EXCL. INSULINS', 'Biguanides'),
  ('C09AA03', 'lisinopril', 'CARDIOVASCULAR SYSTEM', 'AGENTS ACTING ON THE RENIN-ANGIOTENSIN SYSTEM', 'ACE INHIBITORS, PLAIN', 'ACE inhibitors, plain'),
  ('B01AA03', 'warfarin', 'BLOOD AND BLOOD FORMING ORGANS', 'ANTITHROMBOTIC AGENTS', 'ANTITHROMBOTIC AGENTS', 'Vitamin K antagonists');
INSERT INTO struct2atc VALUES (1725, 'A10BA02'), (1573, 'C09AA03'), (2405, 'B01AA03');
INSERT INTO pdb VALUES (2405, '2BXD');
INSERT INTO faers VALUES (1725, 'Lactic acidosis', 812.5, 1450), (1725, 'Diarrhoea', 301.2, 980), (1573, 'Cough', 655.0, 1210), (2405, 'Haemorrhage', 990.1, 2100);
