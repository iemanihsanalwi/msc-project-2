import docx
import re
from docx.oxml.ns import qn

doc = docx.Document("Dissertation_Ieman_Ihsan_25005466_UTP.docx")

print("="*80)
print("RUNNING IN-DEPTH FORENSIC AUDIT OF DISSERTATION DOCX")
print("="*80)

# 1. Count actual table captions for Table 4.1 (ignoring Front Matter list of tables which has dot leaders / tab)
t41_captions = [p.text.strip() for p in doc.paragraphs if re.match(r'^TABLE\s+4\.1[\.]', p.text.strip())]
print(f"Table 4.1 actual table captions: {len(t41_captions)}")
for c in t41_captions:
    print(f"  Found: {c}")
assert len(t41_captions) == 1, f"Expected exactly 1 Table 4.1 caption, found {len(t41_captions)}"

# 2. Check all table captions for duplicates (ignoring LOT entries with tabs)
all_table_captions = [p.text.strip() for p in doc.paragraphs if re.match(r'^TABLE\s+\d+\.\d+[\.]', p.text.strip()) and "\t" not in p.text]
print(f"\nTotal main numbered data tables found: {len(all_table_captions)}")
seen_caps = set()
dup_caps = []
for c in all_table_captions:
    m = re.match(r'^(TABLE\s+\d+\.\d+)', c)
    t_id = m.group(1) if m else c
    if t_id in seen_caps:
        dup_caps.append(t_id)
    seen_caps.add(t_id)
    print(f"  {c}")

if dup_caps:
    print(f"ERROR: Duplicate table IDs found: {dup_caps}")
else:
    print("PASS: Zero duplicate data tables found! Every table is unique.")
assert len(dup_caps) == 0

# 3. Check for stray tokens or backslashes before currency
stray_tokens = []
fused_words = []
for idx, p in enumerate(doc.paragraphs):
    txt = p.text
    if "__USD" in txt or r"\$" in txt:
        stray_tokens.append((idx+1, txt[:100]))
    # Check space-stripped words like \345,402 or \5.00
    if re.search(r'\\[0-9]', txt):
        stray_tokens.append((idx+1, f"Backslash-number: {txt[:100]}"))
    # Check space-stripped words around math
    fused_matches = re.findall(r'[a-z]{3,}[A-Z][a-z]{3,}', txt)
    for m in fused_matches:
        if m not in ['LightGBM', 'TreeSHAP', 'CatBoost', 'PyTorch', 'GraphSAGE', 'Scikit', 'DeepWalk', 'Node2Vec', 'Softplus']:
            fused_words.append((idx+1, m, txt[:100]))

print(f"\nStray currency / backslash tokens found: {len(stray_tokens)}")
for line_no, snip in stray_tokens:
    print(f"  Line {line_no}: {snip}")
assert len(stray_tokens) == 0, f"Found stray tokens: {stray_tokens}"

print(f"Potentially fused words found: {len(fused_words)}")
for line_no, w, snip in fused_words[:10]:
    print(f"  Line {line_no}: '{w}' in '{snip}'")

# 4. Check display equations count and structure
eq_tables = []
for t in doc.tables:
    if len(t.columns) == 2 and len(t.rows) == 1:
        c1_txt = t.cell(0, 1).text.strip()
        if re.match(r'^\(\d+\.\d+\)$', c1_txt):
            eq_tables.append(c1_txt)

print(f"\nTotal formatted display equation tables found: {len(eq_tables)}")
print(f"Sample equation numbers: {eq_tables[:5]} ... {eq_tables[-5:]}")
assert len(eq_tables) > 40, f"Expected > 40 display equations, found {len(eq_tables)}"

# 5. Check table widths in XML
table_overflows = []
for t_idx, t in enumerate(doc.tables):
    tblPr = t._tbl.tblPr
    tblW = tblPr.find(qn('w:tblW'))
    if tblW is not None:
        w_val = int(tblW.get(qn('w:w'), '0'))
        if w_val > 8220:
            table_overflows.append((t_idx, w_val))

print(f"\nTable width overflows (> 8220 dxa / 145 mm): {len(table_overflows)}")
for idx, w in table_overflows:
    print(f"  Table {idx}: width = {w} dxa")
assert len(table_overflows) == 0, f"Found table overflows: {table_overflows}"

print("\n" + "="*80)
print("FORENSIC AUDIT COMPLETE: 100% CLEAN AND COMPLIANT!")
print("="*80)
