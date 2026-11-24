import docx, os

doc = docx.Document('Extended Proposal_25005466.docx')
os.makedirs('figures', exist_ok=True)

for i, p in enumerate(doc.paragraphs):
    blips = p._p.xpath('.//a:blip')
    if blips:
        for b in blips:
            rId = b.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            rel = doc.part.rels.get(rId)
            target = rel.target_ref if rel else 'unknown'
            caption = ''
            for j in range(1, 4):
                if i+j < len(doc.paragraphs) and doc.paragraphs[i+j].text.strip():
                    caption = doc.paragraphs[i+j].text.strip()
                    break
            
            img_part = rel.target_part
            bname = os.path.basename(target)
            clean_cap = ''.join(c if c.isalnum() else '_' for c in caption[:50]).strip('_')
            fname = f'fig_{clean_cap}_{bname}'
            out_path = os.path.join('figures', fname)
            with open(out_path, 'wb') as f:
                f.write(img_part.blob)
            print(f'Saved: {out_path} for "{caption}"')
