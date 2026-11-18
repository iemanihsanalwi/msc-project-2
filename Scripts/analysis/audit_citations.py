import re, json

chapters = [
    'Scripts/chapter_1_introduction.md',
    'Scripts/chapter_2_literature_review.md',
    'Scripts/chapter_3_methodology.md',
    'Scripts/chapter_4_results_and_discussion.md',
    'Scripts/chapter_5_conclusion_and_recommendations.md'
]

# Patterns for citations
# e.g. (Author et al., 2026), (Author & Author, 2021), (Author, 2017)
# or Author et al. (2026), Author (2021)
all_in_text = []

for c in chapters:
    with open(c, encoding='utf-8') as f:
        txt = f.read()
    
    # 1. Parenthetical: ( ... 20xx ... )
    for m in re.finditer(r'\(([^\)]*?(?:19|20)\d{2}[a-z]?[^\)]*?)\)', txt):
        inner = m.group(1)
        # ignore equation references like (3.1), (4.2), etc.
        if re.search(r'\b(?:19|20)\d{2}', inner):
            # split on semicolons
            for part in inner.split(';'):
                part = part.strip()
                if re.search(r'\b(?:19|20)\d{2}', part):
                    all_in_text.append((c, part))
                    
    # 2. Narrative: Author (20xx)
    for m in re.finditer(r'([A-Z][A-Za-z\-]+(?:\s+et\s+al\.)?(?:\s*&\s*[A-Z][A-Za-z\-]+)?)\s*\(((?:19|20)\d{2}[a-z]?)\)', txt):
        all_in_text.append((c, f"{m.group(1)} ({m.group(2)})"))

print(f"Total in-text citation instances: {len(all_in_text)}")
unique_cits = sorted(list(set(item[1] for item in all_in_text)))
for u in unique_cits:
    print("  ", u)
