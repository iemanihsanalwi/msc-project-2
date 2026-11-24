import pypdf
import re

pdf_path = "Dissertation_Ieman_Ihsan_25005466_UTP.pdf"
reader = pypdf.PdfReader(pdf_path)

print(f"Total Pages in Dissertation PDF: {len(reader.pages)}")

# Extract all text
full_text = []
for idx, page in enumerate(reader.pages):
    txt = page.extract_text() or ""
    full_text.append((idx + 1, txt))

# Check for stray tokens
stray_tokens = []
for pg_num, txt in full_text:
    if "__USD" in txt or r"\$" in txt:
        stray_tokens.append((pg_num, txt[:100]))
    if re.search(r'\\[0-9]', txt):
        stray_tokens.append((pg_num, f"Backslash-num: {txt[:100]}"))

print(f"\nStray currency / backslash tokens in PDF: {len(stray_tokens)}")
for pg, s in stray_tokens:
    print(f"  Page {pg}: {s}")

# Check Table 4.1 in PDF
t41_pages = []
for pg_num, txt in full_text:
    if "TABLE 4.1" in txt.upper() and ("MASTER EVALUATION" in txt.upper() or "COMPREHENSIVE EMPIRICAL" in txt.upper()):
        t41_pages.append((pg_num, txt[:150]))

print(f"\nTable 4.1 caption pages in PDF: {len(t41_pages)}")
for pg, s in t41_pages:
    print(f"  Page {pg}: {s.strip()[:100]}")

# Check for Chapters
for ch in range(1, 6):
    found_pages = [pg for pg, txt in full_text if f"CHAPTER {ch}" in txt.upper()]
    print(f"Chapter {ch} found on page(s): {found_pages[:3]}")

# Check key metrics in PDF text
checks = [
    (r'0\.5252', 'Champion AP 0.5252'),
    (r'0\.4814', 'Stage 1 AP 0.4814'),
    (r'0\.4772', 'Stage 3 AP 0.4772'),
    (r'0\.4634', 'Stage 4 AP 0.4634'),
    (r'345,402\.31', 'Champion Loss $345,402.31'),
    (r'629,411\.78', 'Stage 4 Loss $629,411.78'),
    (r'634,205\.97', 'Stage 3 Loss $634,205.97'),
    (r'365,728\.99', 'Stage 1 Loss $365,728.99'),
    (r'0\.0892', 'ECE 0.0892'),
    (r'0\.0055', 'Stage 2 ECE 0.0055'),
]

print("\n--- Quantitative Ground Truth Spot Checks in PDF ---")
for pat, label in checks:
    matches = [pg for pg, txt in full_text if re.search(pat, txt)]
    print(f"  {label:35s}: Found on {len(matches)} page(s) (e.g. p.{matches[0] if matches else 'NONE'})")
    assert len(matches) > 0, f"Missing {label} in PDF!"

print("\nPDF AUDIT PASSED 100%!")
