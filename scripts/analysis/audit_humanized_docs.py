import os
import re
from docx import Document
from lxml import etree

def audit_docx(orig_path, human_path, name):
    print(f"\n=======================================================")
    print(f"AUDITING: {name}")
    print(f"Original:  {orig_path}")
    print(f"Humanized: {human_path}")
    print(f"=======================================================")
    
    doc_orig = Document(orig_path)
    doc_human = Document(human_path)
    
    # 1. Structural Counts
    print("\n--- 1. Structural Element Counts ---")
    p_orig_count = len(doc_orig.paragraphs)
    p_human_count = len(doc_human.paragraphs)
    print(f"Paragraphs: Original = {p_orig_count}, Humanized = {p_human_count}")
    
    t_orig_count = len(doc_orig.tables)
    t_human_count = len(doc_human.tables)
    print(f"Tables:     Original = {t_orig_count}, Humanized = {t_human_count}")
    assert t_orig_count == t_human_count, f"Table count mismatch: {t_orig_count} vs {t_human_count}"
    
    # Check table cells
    for idx, (t_orig, t_human) in enumerate(zip(doc_orig.tables, doc_human.tables)):
        assert len(t_orig.rows) == len(t_human.rows), f"Table {idx} row mismatch"
        assert len(t_orig.columns) == len(t_human.columns), f"Table {idx} col mismatch"
    print(f"Table dimensions check passed: all {t_orig_count} tables have identical row/column structures.")
    
    # 2. XML Drawing and OMML math counts
    xml_orig = doc_orig._element
    xml_human = doc_human._element
    
    drawings_orig = len(xml_orig.xpath('.//w:drawing'))
    drawings_human = len(xml_human.xpath('.//w:drawing'))
    print(f"Drawings (<w:drawing>): Original = {drawings_orig}, Humanized = {drawings_human}")
    assert drawings_orig == drawings_human, f"Drawing mismatch: {drawings_orig} vs {drawings_human}"
    
    # OMML Math
    omath_orig = len(xml_orig.xpath('.//m:oMath'))
    omath_human = len(xml_human.xpath('.//m:oMath'))
    print(f"OMML Equations (<m:oMath>): Original = {omath_orig}, Humanized = {omath_human}")
    assert omath_orig == omath_human, f"OMML Equation mismatch: {omath_orig} vs {omath_human}"

    # 3. Check Forbidden AI Clichés & Em-dashes in Humanized text
    print("\n--- 2. Checking AI Clichés & Em-dashes in Humanized Document ---")
    cliches = [
        "undisputed champion",
        "catastrophic collapse",
        "profound structural duality",
        "acute machine learning challenge",
        "vanishingly small",
        "formidable",
        "extraordinary",
        "delve",
        "tapestry",
        "game-changer",
        "revolutionize",
        "testament to",
        "beacon",
        "crucial to remember",
    ]
    
    human_full_text = "\n".join([p.text for p in doc_human.paragraphs])
    
    em_dash_matches = [p.text for p in doc_human.paragraphs if "—" in p.text]
    print(f"Em-dashes ('—') found in humanized paragraphs: {len(em_dash_matches)}")
    if em_dash_matches:
        for p in em_dash_matches[:3]:
            print(f"  Sample em-dash: {p[:100]}...")
            
    found_cliches = {}
    for c in cliches:
        cnt = human_full_text.lower().count(c)
        if cnt > 0:
            found_cliches[c] = cnt
    print(f"AI clichés found in humanized document: {found_cliches}")
    
    # 4. Critical Key Number & Statistical Preservation Audit
    print("\n--- 3. Critical Numbers & Statistical Verdicts Audit ---")
    key_numbers = [
        "590,540",
        "118,108",
        "115,111",
        "0.5252",
        "0.9053",
        "0.4814",
        "0.8731",
        "0.4772",
        "0.4634",
        "0.1197",
        "0.1420",
        "0.0010",
        "0.0080",
        "12,682.25",
        "495,307.56",
        "507,989.81",
        "634,205.97",
        "629,411.78",
        "30.6%",
        "3.499%",
        "3.50%",
        "28.6%",
        "57.1%",
        "71.4%",
    ]
    
    missing_numbers = []
    for num in key_numbers:
        in_orig = num in "\n".join([p.text for p in doc_orig.paragraphs] + [c.text for t in doc_orig.tables for row in t.rows for c in row.cells])
        in_human = num in "\n".join([p.text for p in doc_human.paragraphs] + [c.text for t in doc_human.tables for row in t.rows for c in row.cells])
        if in_orig and not in_human:
            missing_numbers.append(num)
    
    print(f"Key numbers evaluated: {len(key_numbers)}")
    print(f"Missing key numbers in humanized doc: {missing_numbers}")
    assert len(missing_numbers) == 0, f"Critical numbers missing: {missing_numbers}"

    # 5. Citations Check
    print("\n--- 4. Citations & References Spot Check ---")
    orig_text = "\n".join([p.text for p in doc_orig.paragraphs])
    human_text = human_full_text
    
    # Check common author citations or bracket citations
    sample_cites = [
        "Ivanov & Prokhorenkova",
        "Grinsztajn",
        "McElfresh",
        "Shwartz-Ziv",
        "Borisov",
        "Gorishniy",
        "Kipf & Welling",
        "Hamilton",
        "Veličković",
        "Brody",
        "Zhu",
        "Chawla",
        "Lin",
        "He & Garcia",
        "Benjamini & Hochberg"
    ]
    for cite in sample_cites:
        orig_has = cite.lower() in orig_text.lower()
        human_has = cite.lower() in human_text.lower()
        if orig_has:
            print(f"Citation '{cite}': Present in Original = {orig_has}, Present in Humanized = {human_has}")
            assert human_has, f"Missing citation for {cite}"

    print(f"\n>>> AUDIT PASSED FOR {name} <<<\n")

if __name__ == "__main__":
    audit_docx(
        "IEEE_Technical_Paper_Ieman_Ihsan_25005466.docx",
        "IEEE_Technical_Paper_Ieman_Ihsan_25005466_humanized.docx",
        "IEEE Technical Paper"
    )
    audit_docx(
        "Dissertation_Ieman_Ihsan_25005466_UTP.docx",
        "Dissertation_Ieman_Ihsan_25005466_UTP_humanized.docx",
        "UTP Master Dissertation"
    )
