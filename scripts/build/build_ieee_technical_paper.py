# -*- coding: utf-8 -*-
"""
build_ieee_technical_paper.py
================================================================================
Deterministic, Autonomous Build Pipeline for UTP Master Technical Paper:
"A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud
 Detection Using an Adapted Boost then Convolve Architecture"

Author: Ieman Ihsan Bin Alwi
Supervisor: Prof. Dr. Haruhiro Fujita
Degree: Master of Science in Data Science
Institution: Universiti Teknologi PETRONAS (UTP)
Target Output: IEEE_Technical_Paper_Ieman_Ihsan_25005466.docx

Strict Compliance:
- UTP Master by Coursework Guidelines Section 5.8 & 6.2 (Structure)
- Section 6.4(a) & (c) (Passive voice, target 3,200–4,200 words)
- Appendix 9 (Marking Scheme: Criterion 5 Result and Discussion x8 Multiplier)
- Appendix 12 (IEEE 2-column layout, A4, 1-col title/authors, 0 math in Title/Abstract, NO page numbers)
- Appendix 16 (Part B - Sequential IEEE citations [1]-[N] ordered strictly by first appearance)
================================================================================
"""

import os
import sys
import re
import docx
from docx.shared import Mm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree
import latex2mathml.converter

# ------------------------------------------------------------------------------
# 1. LaTeX to Native Word OMML Converter (for body text and equations)
# ------------------------------------------------------------------------------
XSLT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MML2OMML.XSL')
if not os.path.exists(XSLT_PATH):
    raise FileNotFoundError(f"Missing XSLT stylesheet at {XSLT_PATH}")

xslt_tree = etree.parse(XSLT_PATH)
mml_transform = etree.XSLT(xslt_tree)

def latex_to_omml(latex_code):
    """Convert LaTeX string to native Word OMML XML element."""
    s = latex_code.strip()
    s = s.replace(r'\text{', r'\mathrm{')
    s = s.replace(r'\mathbf{', r'\mathbf{')
    s = s.replace(r'\boldsymbol{', r'\mathbf{')
    s = s.replace(r'\mathbb{I}', r'\mathrm{I}')
    s = s.replace(r'\,', ' ')
    s = s.replace(r'\;', ' ')
    s = s.replace(r'\quad', ' ')
    s = s.replace(r'\qquad', '   ')
    s = s.replace(r'\land', r'\wedge')
    s = s.replace(r'\lor', r'\vee')
    s = s.replace(r'\oslash', r'⊘')
    s = s.replace(r'\_', '_')
    s = re.sub(r'\\arg\\?min', r'\\operatorname{argmin}', s)
    s = s.replace('__USD_TOKEN__', r'\mathrm{\$}').replace(r'\$', r'\mathrm{\$}')
    
    mathml_str = latex2mathml.converter.convert(s)
    mathml_tree = etree.fromstring(mathml_str.encode('utf-8'))
    omml_tree = mml_transform(mathml_tree)
    return omml_tree.getroot()

# ------------------------------------------------------------------------------
# 2. Strict IEEE Reference Catalog (37 Verified Publications)
# ------------------------------------------------------------------------------
REFERENCE_DB = {
    "NILSON": "Nilson Report, \"Card fraud losses worldwide: 2023–2028,\" Issue 1254, HSN Consultants, 2024.",
    "DAL_POZZOLO": "A. Dal Pozzolo, G. Boracchi, O. Caelen, C. Alippi, and G. Bontempi, \"Credit card fraud detection: A realistic modeling and a novel learning strategy,\" IEEE Transactions on Neural Networks and Learning Systems, vol. 29, no. 8, pp. 3784–3797, 2018.",
    "LIGHTGBM": "G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, \"LightGBM: A highly efficient gradient boosting decision tree,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 3146–3154, 2017.",
    "GRINSZTAJN": "L. Grinsztajn, E. Oyallon, and G. Varoquaux, \"Why do tree-based models still outperform deep learning on typical tabular data?\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 35, pp. 507–520, 2022.",
    "VAN_BELLE": "R. Van Belle, B. Baesens, and J. De Weerdt, \"CATCHM: A novel network-based credit card fraud detection method using node representation learning,\" Decision Support Systems, vol. 164, p. 113866, 2023.",
    "POURHABIBI": "T. Pourhabibi, K.-L. Ong, B. H. Kam, and Y. L. Boo, \"Fraud detection: A systematic literature review of graph-based anomaly detection approaches,\" Decision Support Systems, vol. 133, p. 113303, 2020.",
    "MA": "X. Ma, J. Wu, S. Xue, J. Yang, C. Zhou, Q. Z. Sheng, H. Xiong, and L. Akoglu, \"A comprehensive survey on graph anomaly detection with deep learning,\" IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 12, pp. 12012–12038, 2021.",
    "LIN_FOCAL": "T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, \"Focal loss for dense object detection,\" in Proc. IEEE Int. Conf. Computer Vision (ICCV), pp. 2980–2988, 2017.",
    "POPOV": "S. Popov, S. Morozov, and A. Babenko, \"Neural Oblivious Decision Ensembles for deep learning on tabular data,\" in International Conference on Learning Representations (ICLR), 2020.",
    "ARIK": "S. O. Arik and T. Pfister, \"TabNet: Attentive interpretable tabular learning,\" in Proc. AAAI Conf. Artificial Intelligence, vol. 35, no. 8, pp. 6679–6687, 2021.",
    "HAZIMEH": "H. Hazimeh, M. S. Pydi, and R. Mazumder, \"The Tree Ensemble Layer: Differentiability meets conditional computation,\" in Proc. 37th Int. Conf. Machine Learning (ICML), pp. 4138–4148, 2020.",
    "AKLI": "H. Akli, S. Hamrioui, and I. Stéphan, \"Heterophily outlier temporal aware graph neural network for online fraud detection,\" Expert Systems with Applications, vol. 331, p. 133278, 2026.",
    "ROSSI": "E. Rossi, B. Chamberlain, F. Frasca, D. Eynard, F. Monti, and M. Bronstein, \"Temporal Graph Networks for deep learning on dynamic graphs,\" in ICML Workshop on Graph Representation Learning, 2020.",
    "IVANOV": "S. Ivanov and L. Prokhorenkova, \"Boost then Convolve: Gradient boosting meets graph neural networks,\" in International Conference on Learning Representations (ICLR), 2021.",
    "FRIEDMAN": "J. H. Friedman, \"Greedy function approximation: A gradient boosting machine,\" The Annals of Statistics, vol. 29, no. 5, pp. 1189–1232, 2001.",
    "XGBOOST": "T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining (KDD), pp. 785–794, 2016.",
    "CATBOOST": "L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, \"CatBoost: Unbiased boosting with categorical features,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 31, pp. 6638–6648, 2018.",
    "KIPF": "T. N. Kipf and M. Welling, \"Semi-supervised classification with graph convolutional networks,\" in International Conference on Learning Representations (ICLR), 2017.",
    "HAMILTON": "W. L. Hamilton, R. Ying, and J. Leskovec, \"Inductive representation learning on large graphs,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 1024–1034, 2017.",
    "VELICKOVIC": "P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, and Y. Bengio, \"Graph attention networks,\" in International Conference on Learning Representations (ICLR), 2018.",
    "GILMER": "J. Gilmer, S. S. Schoenholz, P. F. Riley, O. Vinyals, and G. E. Dahl, \"Neural message passing for quantum chemistry,\" in Proc. 34th Int. Conf. Machine Learning (ICML), pp. 1263–1272, 2017.",
    "MORRIS": "C. Morris, M. Ritzert, M. Fey, W. L. Hamilton, J. E. Lenssen, G. Rattan, and M. Grohe, \"Weisfeiler and Leman go neural: Higher-order graph neural networks,\" in Proc. AAAI Conf. Artificial Intelligence, vol. 33, no. 01, pp. 4602–4609, 2019.",
    "XU": "K. Xu, W. Hu, J. Leskovec, and S. Jegelka, \"How powerful are graph neural networks?\" in International Conference on Learning Representations (ICLR), 2019.",
    "ZHU_HOMOPHILY": "J. Zhu, Y. Yan, L. Zhao, M. Heimann, L. Akoglu, and D. Koutra, \"Beyond homophily in graph neural networks: Current limitations and effective designs,\" in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, pp. 7793–7804, 2020.",
    "DOU": "Y. Dou, Z. Liu, L. Sun, Y. Deng, H. Peng, and P. S. Yu, \"Enhancing graph neural network-based fraud detectors against camouflaged fraudsters,\" in Proc. 29th ACM Int. Conf. Information and Knowledge Management (CIKM), pp. 315–324, 2020.",
    "LIU": "Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He, \"Pick and choose: A GNN-based imbalanced learning approach for fraud detection,\" in Proc. Web Conf. 2021 (WWW), pp. 3168–3177, 2021.",
    "WANG_GAT": "D. Wang, J. Lin, P. Cui, Q. Jia, Z. Wang, Y. Fang, Q. Yu, J. Zhou, S. Yang, and Y. Qi, \"A semi-supervised graph attentive network for financial fraud detection,\" in IEEE Int. Conf. Data Mining (ICDM), pp. 598–607, 2019.",
    "VAN_VLASSELAER": "V. Van Vlasselaer, T. Eliassi-Rad, L. Akoglu, M. Snoeck, and B. Baesens, \"GOTCHA! Network-based fraud detection for social security fraud,\" Management Science, vol. 63, no. 9, pp. 3090–3110, 2017.",
    "WEBER": "M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson, \"Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics,\" in KDD Workshop on Anomaly Detection in Finance, 2019.",
    "LAHCEN": "T. Lahcen, Y. Amadiaz, M. Bahbib, and A. Nait Sidi Moh, \"A heterogeneous graph representation learning approach for fraud detection in supply chain networks,\" Intelligent Systems with Applications, vol. 31, p. 200686, 2026.",
    "MAO": "X. Mao, J. Li, X. Zhu, and Y. Wang, \"Enhancing financial fraud detection with graph neural network and ensemble learning: Insights from Related Party Transactions network,\" International Review of Economics & Finance, vol. 108, p. 105217, 2026.",
    "MOHAWESH": "R. Mohawesh, S. Maqsood, H. Bany Salameh, and G. Elrefae, \"Leveraging GraphSAGE and large language models with cross-attention for transaction fraud detection: Evidence from credit card and PaySim datasets,\" Information Processing & Management, vol. 63, no. 1, p. 104860, 2026.",
    "ONWUDEBELU": "U. Onwudebelu, O. O. Fasola, E. C. Ugonna, and N. C. Woods, \"A behavior-aware intelligent decision framework for real-time credit card fraud detection using deep learning and adaptive user profiling,\" Engineering Applications of Artificial Intelligence, vol. 181, p. 115641, 2026.",
    "TANG": "Y. Tang and Y. Liang, \"Credit card fraud detection based on federated graph learning,\" Expert Systems with Applications, vol. 256, p. 124979, 2024.",
    "WANG_DGGCN": "N. Wang, X. Zhang, and M. Liu, \"DG-GCN: A dual similarity-guided graph convolutional network for imbalanced node classification,\" Neurocomputing, vol. 580, p. 134284, 2026.",
    "ZHANG": "J. Zhang, Z. Fang, Y. Tang, and L. Ni, \"TS-GNN: Modeling temporal-semantic discrepancy for graph-based credit card fraud detection,\" Expert Systems with Applications, vol. 331, p. 133233, 2026.",
    "ZHU_DISCO": "R. Zhu, S. Wang, Y. Li, and C. Chen, \"DISCO: Decoupling representation learning and risk control for reliable credit card fraud detection,\" Decision Support Systems, vol. 178, p. 114717, 2026."
}

# ------------------------------------------------------------------------------
# 3. Document Layout and Typography Helpers
# ------------------------------------------------------------------------------
def set_cell_margins(cell, top=60, bottom=60, left=80, right=80):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def preprocess_text(text):
    """Safeguards currency symbols and cleans typographical anomalies."""
    t = text.replace('--', '–')
    t = t.replace('`', '')
    t = t.replace(r'\$', '__USD_TOKEN__')
    # Currency like $33.8 billion or $365,728.99 or $5.00 (dollar followed by optional space and digit)
    t = re.sub(r'\$(?=\s*\d)', '__USD_TOKEN__', t)
    # If there was a trailing dollar like __USD_TOKEN__118,108$
    t = re.sub(r'__USD_TOKEN__([\d,\.]+)\$', r'__USD_TOKEN__\1', t)
    # Numbers or percentages wrapped in $ e.g. $118,108$ or $24.51%$ (must contain at least one digit)
    t = re.sub(r'\$(\s*[\d,\.]*\d[\d,\.]*\s*%?)\$', r'\1', t)
    return t

def parse_runs(text):
    """
    Parses a paragraph string with possible bold (**), italic (*), and math ($...$).
    Correctly handles currency and preserves spaces.
    """
    t = preprocess_text(text)
    bold_pattern = re.compile(r'\*\*(.+?)\*\*')
    parts = []
    last_idx = 0
    for m in bold_pattern.finditer(t):
        start, end = m.span()
        if start > last_idx:
            parts.append((t[last_idx:start], False))
        parts.append((m.group(1), True))
        last_idx = end
    if last_idx < len(t):
        parts.append((t[last_idx:], False))
        
    final_tokens = []
    for chunk, is_bold in parts:
        # Match only $...$ up to 150 chars without newlines
        math_tokens = re.split(r'(\$[^\$\n]{1,150}?\$)', chunk)
        for mt in math_tokens:
            if not mt:
                continue
            if mt.startswith('$') and mt.endswith('$') and len(mt) > 2:
                math_content = mt[1:-1]
                # If content is purely numbers, punctuation, spaces, or percent, treat as plain text
                if re.match(r'^[\d,\.\s%]+$', math_content):
                    text_clean = math_content.replace('__USD_TOKEN__', '$')
                    final_tokens.append(('text', text_clean, is_bold, False))
                else:
                    math_content = math_content.replace('__USD_TOKEN__', r'\mathrm{\$}').replace(r'\$', r'\mathrm{\$}')
                    final_tokens.append(('math', math_content, is_bold, False))
            else:
                text_clean = mt.replace('__USD_TOKEN__', '$')
                italic_pattern = re.compile(r'\*([^*]+)\*')
                sub_last = 0
                for im in italic_pattern.finditer(text_clean):
                    s, e = im.span()
                    if s > sub_last:
                        final_tokens.append(('text', text_clean[sub_last:s], is_bold, False))
                    final_tokens.append(('text', im.group(1), is_bold, True))
                    sub_last = e
                if sub_last < len(text_clean):
                    final_tokens.append(('text', text_clean[sub_last:], is_bold, False))
                    
    return final_tokens

def add_body_p(doc, text, first_line_indent_mm=3.5, space_before_pt=0, space_after_pt=0):
    """Adds a standard IEEE 10 pt body paragraph (justified, single-spaced)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(space_before_pt)
    p.paragraph_format.space_after = Pt(space_after_pt)
    if first_line_indent_mm > 0:
        p.paragraph_format.first_line_indent = Mm(first_line_indent_mm)
        
    runs_data = parse_runs(text)
    for run_type, content, is_bold, is_italic in runs_data:
        if not content:
            continue
        if run_type == 'math':
            try:
                omml = latex_to_omml(content)
                p._p.append(omml)
            except Exception:
                r = p.add_run(content)
                r.font.name = "Times New Roman"
                r.font.size = Pt(10)
                r.italic = True
                r.bold = is_bold
        else:
            r = p.add_run(content)
            r.font.name = "Times New Roman"
            r.font.size = Pt(10)
            r.bold = is_bold
            r.italic = is_italic
    return p

def add_heading_1(doc, title_text):
    """IEEE Major Heading: Centered, Roman Numeral, Uppercase, 10 pt."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(title_text.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.bold = True
    return p

def add_heading_2(doc, title_text):
    """IEEE Sub-Heading: Left-aligned, Capital Letter Prefix, Italic, 10 pt."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(title_text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.italic = True
    r.bold = False
    return p

def add_heading_3(doc, title_text):
    """IEEE Sub-Sub-Heading: Indented, Arabic Numeral Prefix, Italic, 10 pt."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.first_line_indent = Mm(3.5)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(title_text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    r.italic = True
    return p

def add_ieee_display_eq(doc, latex_input, eq_num_str):
    """
    Renders an IEEE display equation strictly within single-column width (3.385 in / 86 mm).
    Uses paragraph tab stops (1.69 in Center, 3.38 in Right) instead of tables to prevent column collision.
    Supports multi-line equations if latex_input is a list or tuple.
    """
    lines = latex_input if isinstance(latex_input, (list, tuple)) else [latex_input]
    
    for idx, line in enumerate(lines):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(0)
        p.paragraph_format.right_indent = Inches(0)
        p.paragraph_format.first_line_indent = Inches(0)
        p.paragraph_format.line_spacing = 1.0
        
        if len(lines) == 1:
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
        elif idx == 0:
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(1)
        elif idx == len(lines) - 1:
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(4)
        else:
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            
        p.paragraph_format.tab_stops.add_tab_stop(Inches(1.69), WD_TAB_ALIGNMENT.CENTER)
        if idx == len(lines) - 1:
            p.paragraph_format.tab_stops.add_tab_stop(Inches(3.38), WD_TAB_ALIGNMENT.RIGHT)
            
        p.add_run('\t')
        try:
            omml = latex_to_omml(line)
            p._p.append(omml)
        except Exception:
            r = p.add_run(line)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            r.italic = True
            
        if idx == len(lines) - 1:
            p.add_run('\t')
            r_num = p.add_run(f"({eq_num_str})")
            r_num.font.name = "Times New Roman"
            r_num.font.size = Pt(9.5)

def add_ieee_figure(doc, img_path, caption_str, width_mm=84):
    """Adds a single-column IEEE figure with 8 pt caption below."""
    if not os.path.exists(img_path):
        print(f"  WARNING: Image not found at {img_path}")
        return
        
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(6)
    p_img.paragraph_format.space_after = Pt(2)
    r_img = p_img.add_run()
    r_img.add_picture(img_path, width=Mm(width_mm))
    
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_cap.paragraph_format.line_spacing = 1.0
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(8)
    
    runs_data = parse_runs(caption_str)
    for run_type, content, is_bold, is_italic in runs_data:
        r = p_cap.add_run(content)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        r.bold = is_bold
        r.italic = is_italic

def add_ieee_table(doc, caption_str, headers, rows_data, col_widths_mm, is_full_width=False):
    """Adds a single-column or full-width IEEE academic table strictly bounded to column width."""
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(8)
    p_cap.paragraph_format.space_after = Pt(3)
    p_cap.paragraph_format.line_spacing = 1.0
    
    runs_data = parse_runs(caption_str)
    for run_type, content, is_bold, is_italic in runs_data:
        r = p_cap.add_run(content)
        r.font.name = "Times New Roman"
        r.font.size = Pt(8)
        r.bold = True
        r.italic = False

    num_rows = len(rows_data) + 1
    num_cols = len(headers)
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    # Force strict table width (4824 dxa = 3.35 in = 85 mm)
    tblPr = table._tbl.tblPr
    tblW = OxmlElement('w:tblW')
    tblW.set(qn('w:w'), '4824')
    tblW.set(qn('w:type'), 'dxa')
    tblPr.append(tblW)

    for c_idx, width in enumerate(col_widths_mm):
        table.columns[c_idx].width = Mm(width)

    # Header Row
    hdr_cells = table.rows[0].cells
    for c_idx, h_text in enumerate(headers):
        w_dxa = int(col_widths_mm[c_idx] * 56.6929)
        hdr_cells[c_idx].width = Mm(col_widths_mm[c_idx])
        p = hdr_cells[c_idx].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        
        runs = parse_runs(h_text)
        for rt, cnt, b, it in runs:
            r = p.add_run(cnt)
            r.font.name = "Times New Roman"
            r.font.size = Pt(7.5)
            r.bold = True
            r.italic = it
            
        hdr_cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(hdr_cells[c_idx], top=40, bottom=40, left=40, right=40)
        
        tcPr = hdr_cells[c_idx]._tc.get_or_add_tcPr()
        tcW = OxmlElement('w:tcW')
        tcW.set(qn('w:w'), str(w_dxa))
        tcW.set(qn('w:type'), 'dxa')
        tcPr.append(tcW)
        
        borders = parse_xml(
            r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            r'<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            r'<w:left w:val="none"/><w:right w:val="none"/>'
            r'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
            r'</w:tcBorders>'
        )
        tcPr.append(borders)

    # Body Rows
    for r_idx, row_values in enumerate(rows_data):
        row_cells = table.rows[r_idx + 1].cells
        is_last_row = (r_idx == len(rows_data) - 1)
        
        for c_idx, val in enumerate(row_values):
            w_dxa = int(col_widths_mm[c_idx] * 56.6929)
            row_cells[c_idx].width = Mm(col_widths_mm[c_idx])
            p = row_cells[c_idx].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            
            runs = parse_runs(str(val))
            for rt, cnt, b, it in runs:
                r = p.add_run(cnt)
                r.font.name = "Times New Roman"
                r.font.size = Pt(7.5)
                r.bold = b
                r.italic = it
                
            row_cells[c_idx].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(row_cells[c_idx], top=30, bottom=30, left=40, right=40)
            
            tcPr = row_cells[c_idx]._tc.get_or_add_tcPr()
            tcW = OxmlElement('w:tcW')
            tcW.set(qn('w:w'), str(w_dxa))
            tcW.set(qn('w:type'), 'dxa')
            tcPr.append(tcW)
            
            if is_last_row:
                b_xml = (
                    r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    r'<w:top w:val="none"/><w:left w:val="none"/><w:right w:val="none"/>'
                    r'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
                    r'</w:tcBorders>'
                )
            else:
                b_xml = (
                    r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                    r'<w:top w:val="none"/><w:left w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/>'
                    r'</w:tcBorders>'
                )
            borders = parse_xml(b_xml)
            tcPr.append(borders)

    p_post = doc.add_paragraph()
    p_post.paragraph_format.space_before = Pt(0)
    p_post.paragraph_format.space_after = Pt(4)

# ------------------------------------------------------------------------------
# 4. Section 1: Single-Column Title and Author Header Block (Appendix 12)
# ------------------------------------------------------------------------------
def build_ieee_header(doc):
    sec1 = doc.sections[0]
    sec1.page_width = Mm(210)
    sec1.page_height = Mm(297)
    sec1.top_margin = Mm(19)
    sec1.bottom_margin = Mm(25.4)
    sec1.left_margin = Mm(16)
    sec1.right_margin = Mm(16)
    
    # Strictly remove page numbers from header/footer
    sec1.header.is_linked_to_previous = False
    sec1.footer.is_linked_to_previous = False
    for p in sec1.header.paragraphs:
        p.text = ""
    for p in sec1.footer.paragraphs:
        p.text = ""

    # Paper Title (24 pt, Regular, Centered, Zero Math/Symbols)
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(0)
    p_title.paragraph_format.space_after = Pt(12)
    p_title.paragraph_format.line_spacing = 1.0
    r_title = p_title.add_run(
        "A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud Detection "
        "Using an Adapted Boost then Convolve Architecture"
    )
    r_title.font.name = "Times New Roman"
    r_title.font.size = Pt(24)
    r_title.bold = False

    # Author Block Table (2 columns, borderless, centered)
    tbl_auth = doc.add_table(rows=1, cols=2)
    tbl_auth.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_auth.autofit = False
    tbl_auth.columns[0].width = Mm(85)
    tbl_auth.columns[1].width = Mm(85)

    for cell in tbl_auth.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = parse_xml(
            r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            r'<w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/><w:right w:val="none"/>'
            r'</w:tcBorders>'
        )
        tcPr.append(borders)
        set_cell_margins(cell, top=0, bottom=60, left=40, right=40)

    # Author 1: Candidate
    c0 = tbl_auth.cell(0, 0)
    p0 = c0.paragraphs[0]
    p0.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p0.paragraph_format.line_spacing = 1.0
    p0.paragraph_format.space_before = Pt(0)
    p0.paragraph_format.space_after = Pt(0)
    
    r1 = p0.add_run("Ieman Ihsan Bin Alwi\n")
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(11)
    r1.bold = True
    
    r2 = p0.add_run("Department of Fundamental and Applied Sciences\nFaculty of Science and Information Technology\nUniversiti Teknologi PETRONAS\n")
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(10)
    r2.italic = True
    
    r3 = p0.add_run("Seri Iskandar, Perak, Malaysia\nieman_25005466@utp.edu.my")
    r3.font.name = "Times New Roman"
    r3.font.size = Pt(10)

    # Author 2: Supervisor
    c1 = tbl_auth.cell(0, 1)
    p1 = c1.paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.line_spacing = 1.0
    p1.paragraph_format.space_before = Pt(0)
    p1.paragraph_format.space_after = Pt(0)
    
    r4 = p1.add_run("Haruhiro Fujita\n")
    r4.font.name = "Times New Roman"
    r4.font.size = Pt(11)
    r4.bold = True
    
    r5 = p1.add_run("Department of Fundamental and Applied Sciences\nFaculty of Science and Information Technology\nUniversiti Teknologi PETRONAS\n")
    r5.font.name = "Times New Roman"
    r5.font.size = Pt(10)
    r5.italic = True
    
    r6 = p1.add_run("Seri Iskandar, Perak, Malaysia\nharuhiro.fujita@utp.edu.my")
    r6.font.name = "Times New Roman"
    r6.font.size = Pt(10)

    # Small separator before section break
    p_sep = doc.add_paragraph()
    p_sep.paragraph_format.space_before = Pt(0)
    p_sep.paragraph_format.space_after = Pt(4)

# ------------------------------------------------------------------------------
# 5. Full Document Content Specification with Dynamic Reference Tags
# ------------------------------------------------------------------------------
DOCUMENT_FLOW = [
    ('h1', 'I. INTRODUCTION'),
    ('h2', 'A. Background and Operational Context'),
    ('p', 'Electronic clearing networks process billions of card transactions daily. Over the same period, card fraud has expanded, with annual worldwide losses exceeding $33.8 billion {REF:NILSON}. In production clearing systems, automated fraud screening engines operate under strict latency limits, requiring incoming authorizations to be evaluated in flight within sub-100-millisecond budgets. In addition, ground-truth labels are delayed: fraudulent transactions are confirmed through chargeback disputes only 30 to 90 days after authorization {REF:DAL_POZZOLO}. Models running in streaming payment environments therefore cannot rely on immediate ground truth or future transaction data.'),
    ('h2', 'B. Structural Duality of Transaction Clearing Streams'),
    ('p', 'Payment transaction streams exhibit two distinct data structures. First, transactions arrive as tabular records containing transaction amounts, velocity counts, geographic codes, and high-cardinality categorical identifiers where missingness frequently exceeds 40%. For these tabular features, Gradient Boosted Decision Trees (GBDTs), such as LightGBM {REF:LIGHTGBM}, serve as the standard baseline due to their invariance to monotonic transformations, learned default routing for missing values, and orthogonal axis-aligned partitioning {REF:GRINSZTAJN}.'),
    ('p', 'Second, transactions are linked through shared payment entities, including cards, terminals, devices, and IP addresses. Graph Neural Networks (GNNs) provide an established approach for modeling these relational structures {REF:VAN_BELLE}, {REF:POURHABIBI}, {REF:MA}. Most academic GNN benchmarks, however, construct static, transductive graphs that connect nodes bidirectionally across time, introducing temporal look-ahead leakage {REF:VAN_BELLE}.'),
    ('h2', 'C. Problem Statement and Systemic Vulnerabilities'),
    ('p', 'Deploying graph architectures into production fraud detection highlights four practical vulnerabilities:'),
    ('p', '1) Architectural Optimization Separation: Two-stage hybrid pipelines, such as Residual Graph Neural Networks (Res-GNN) {REF:IVANOV}, feed fixed tabular representations into GNN layers without backpropagating structural gradients into the tree ensemble. Conversely, differentiable decision trees, including Neural Oblivious Decision Ensembles {REF:POPOV}, TabNet {REF:ARIK}, and Tree Ensemble Layers {REF:HAZIMEH}, approximate step splits with smooth sigmoidal activations, which degrades partition accuracy on tabular boundaries.'),
    ('p', '2) Temporal Look-Ahead Leakage: Academic GNN benchmarks frequently evaluate static graphs with random cross-validation. In this setting, early transactions aggregate relational embeddings from future transactions associated with the same cardholder, inflating reported metrics {REF:AKLI}, {REF:ROSSI}.'),
    ('p', '3) Open-World Cold-Start Conditions: High entity turnover is common in transaction streams. In production payment clearing, over 30% of transactions involve cards or devices not present in historical training data. When relational neighborhoods are empty, multi-hop message passing has no historical edges to aggregate, leaving neural representations uninformative {REF:AKLI}.'),
    ('p', '4) Objective and Cost Misalignment: Standard cross-entropy objectives treat misclassification symmetrically, overlooking extreme class imbalance (~3.5% fraud). They also neglect operational queue limits, where fraud triage teams can manually review at most 1.0% of transaction volume.'),
    ('h2', 'D. Research Objectives and Pre-Specified Scope'),
    ('p', 'To evaluate these issues under production constraints, this study formulates an adapted Boost then Convolve (BGNN) framework {REF:IVANOV} for streaming transaction data. The research addresses five specific objectives: (1) specify a strictly time-causal dynamic heterogeneous graph $G_{<t}$ that eliminates future look-ahead edges; (2) establish the performance baseline of tuned tabular GBDTs (Stage 1); (3) measure the empirical gain from historical graph feature engineering within GBDTs (Stage 2); (4) evaluate decoupled neural message passing over fixed tabular embeddings (Stage 3); and (5) test whether alternating functional-gradient optimization between decision trees and graph convolutions with an adaptive fallback blending gate $\\beta_i$ (Stage 4) improves upon tabular baselines under Out-of-Time (OOT) evaluation on the IEEE-CIS benchmark ($N = 590,540$ transactions).'),
    ('h1', 'II. RELATED WORK AND THEORETICAL BACKGROUND'),
    ('h2', 'A. Tabular Decision Trees vs. Relational Graph Networks'),
    ('p', 'GBDTs build on the greedy function approximation framework of Friedman {REF:FRIEDMAN}, implemented in libraries such as XGBoost {REF:XGBOOST}, LightGBM {REF:LIGHTGBM}, and CatBoost {REF:CATBOOST}. They construct an additive ensemble of $K$ decision trees, $F_K(\\mathbf{x}) = F_0(\\mathbf{x}) + \\sum_{k=1}^K \\eta f_k(\\mathbf{x})$, where each tree partitions the input space into orthogonal hyper-rectangles. Grinsztajn et al. {REF:GRINSZTAJN} showed that GBDTs remain competitive with deep neural models on tabular datasets due to their invariance to monotonic feature transformations, native handling of missing values via learned default split directions, and capability to represent sharp threshold discontinuities. However, standard GBDTs treat transactions as independent and identically distributed (i.i.d.) observations, without mechanisms to aggregate relational dependencies across accounts {REF:POURHABIBI}, {REF:MA}.'),
    ('p', 'Conversely, Message Passing Neural Networks (MPNNs) {REF:GILMER}, including Graph Convolutional Networks (GCN) {REF:KIPF}, GraphSAGE {REF:HAMILTON}, and Graph Attention Networks (GAT) {REF:VELICKOVIC}, aggregate representations across node neighborhoods: $\\mathbf{h}_v^{(l)} = \\text{UPDATE}(\\mathbf{h}_v^{(l-1)}, \\text{AGG}(\\{\\mathbf{h}_u^{(l-1)} : u \\in \\mathcal{N}(v)\\}))$. The expressive capacity of standard MPNNs is bounded by the 1-Weisfeiler-Lehman (1-WL) graph isomorphism test {REF:MORRIS}, {REF:XU}. In transaction networks, MPNNs often encounter low homophily {REF:ZHU_HOMOPHILY}. Because fraudulent transactions are interleaved among normal cardholder activity across merchants and devices, financial transaction graphs exhibit heterophily. In these structures, spatial averaging functions as a low-pass filter, smoothing anomalous fraud signals into the surrounding legitimate transactions {REF:ZHU_HOMOPHILY}.'),
    ('h2', 'B. Hybrid Graph-Tabular Learning Architectures'),
    ('p', 'Several hybrid designs attempt to combine tabular and graph learning. Two-stage pipelines (Res-GNN) {REF:IVANOV} feed pre-trained tabular embeddings into downstream GNN layers, but prevent tree splits from adjusting to errors identified during graph propagation. Differentiable tree models, including Neural Oblivious Decision Ensembles (NODE) {REF:POPOV}, TabNet {REF:ARIK}, and Tree Ensemble Layers {REF:HAZIMEH}, enable gradient backpropagation through soft continuous routing, but frequently encounter training instabilities on high-cardinality categorical data.'),
    ('p', 'The Boost then Convolve (BGNN) framework by Ivanov and Prokhorenkova {REF:IVANOV} introduced an alternating scheme between GBDT tree fitting and GNN message passing on static homogeneous graphs. In each outer iteration, the GNN trains neural weights and evaluates functional gradients with respect to node representation coordinates: $\\mathbf{R}_i^m = -\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. Decision trees then fit these residual vectors to update coordinates additively: $\\mathbf{z}_i^m = \\mathbf{z}_i^{m-1} + \\eta_{\\text{boost}} F_{\\text{tree}}(\\mathbf{x}_i)$ {REF:IVANOV}.'),
    ('h2', 'C. Streaming Vulnerabilities in Operational Financial Graphs'),
    ('p', 'Recent surveys {REF:POURHABIBI}, {REF:MA} and empirical evaluations {REF:DOU}, {REF:WANG_GAT}, {REF:VAN_VLASSELAER}, {REF:WEBER}, {REF:LAHCEN}, {REF:TANG}, {REF:MAO} document relational methods for fraud detection. Several benchmark designs, however, diverge from production conditions. Akli et al. {REF:AKLI} and Van Belle et al. {REF:VAN_BELLE} showed that static graph partitioning introduces future look-ahead information into training sets. While dynamic formulations like Temporal Graph Networks (TGN) {REF:ROSSI} maintain causal ordering, their memory state requirements can challenge sub-100-millisecond clearing latency limits.'),
    ('h2', 'D. Conceptual Synthesis Matrix'),
    ('p', 'To adapt BGNN to streaming transaction constraints, Table I summarizes the design differences between canonical BGNN {REF:IVANOV} and the time-causal formulation evaluated in this study.'),
    ('table', ('TABLE I. CONCEPTUAL MATRIX: CANONICAL BGNN VS. PROPOSED TIME-CAUSAL FRAUD BGNN', ['Dimension', 'Canonical BGNN {REF:IVANOV}', 'Proposed Time-Causal Fraud BGNN'], [['Graph Structure', 'Static, undirected homogeneous graph; transductive evaluation', 'Dynamic time-causal heterogeneous graph G<t; strictly inductive OOT'], ['Initial Latent Z0', 'Raw normalized tabular features; continuous scaling only', '49-dim matrix: 32 standardized continuous + 15 PCA (69.5% var) + 1 OOF Logit + 1 ID flag'], ['Lookback Horizon', 'Global unconstrained multi-hop neighborhood aggregation', 'Strict 30-day bounded lookback (tau_max = 2,592,000 s)'], ['Loss Objective', 'Symmetric cross-entropy; unweighted class balance', 'Normalized Mini-Batch Focal Loss (alpha=0.25, gamma=2.0)'], ['Functional Boosting', 'Unbounded gradient vectors across d_emb trees; prone to divergence', 'Rank-1 coordinate displacement via tree fitted to CE residuals, projected along d, clipped to delta_R = 0.20'], ['Cold-Start Handling', 'Assumes all nodes have structural graph edges', 'Monotonic fallback blending gate with connectivity indicator beta_i = 0 on isolated nodes']], [23, 31, 31])),
    ('h1', 'III. TIME-CAUSAL BGNN METHODOLOGY'),
    ('h2', 'A. Chronological Partitioning and Information Policy'),
    ('p', 'The empirical study uses the IEEE-CIS fraud detection benchmark, comprising 590,540 verified transactions over 182 days (TransactionDT in [86,400, 15,811,131]) with a 3.499% overall fraud prevalence. The data is partitioned into five chronological blocks of 118,108 transactions each. Blocks 1 through 4 serve as the expanding training and validation sets. Block 5 (covering Days 140.12 to 182.00, with 115,111 post-embargo transactions and 3,929 frauds) is held out as a sealed Out-of-Time (OOT) test set, evaluated only once. A 1-day embargo window and a 30-day chargeback latency buffer prevent temporal leakage across boundaries {REF:DAL_POZZOLO}.'),
    ('h2', 'B. Time-Causal Heterogeneous Graph Formulation'),
    ('p', 'At query timestamp $t_i$, the historical graph $G_{<t_i} = (\\mathcal{V}_{<t_i}, \\mathcal{E}_{<t_i})$ contains only entities and edges recorded at timestamps $t < t_i$. Entities are resolved using deterministic composite keys: card entities $\\text{CardID} = \\langle\\text{card1}, \\dots, \\text{card6}, \\text{addr1}\\rangle$ and device entities $\\text{DeviceID} = \\langle\\text{DeviceInfo}, \\text{id\\_31}, \\text{id\\_30}, \\text{DeviceType}\\rangle$. Each incoming transaction $v_i$ is evaluated within an ephemeral query graph $G_i$ that draws directed edges from historical nodes within a 30-day lookback window ($K_{\\text{lookback}} = 2{,}592{,}000\\,\\text{s}$). After inference completes, $v_i$ is committed to the persistent graph.'),
    ('h2', 'C. Adapted Boost then Convolve Architecture'),
    ('p', 'The model initializes an aligned continuous representation matrix $\\mathbf{Z}_0 \\in \\mathbb{R}^{N \\times 49}$ consisting of 32 standardized continuous attributes, 15 principal components capturing 69.5% of variance from the behavioral features, one out-of-fold cross-validated LightGBM risk logit, and a binary identity availability indicator. Using Scaled Exponential Linear Unit (SELU) activations, 1-layer causal 1-hop unweighted mean pooling ($K_{\\text{card}} \\le 10$, $K_{\\text{dev}} \\le 5$, 30-day window), and a residual skip connection, message passing computes:'),
    ('eq', ('\\mathbf{h}_i = \\text{SELU}\\left( \\mathbf{W}_{\\text{self}} \\mathbf{z}_i + \\sum_{r \\in \\mathcal{R}} \\frac{1}{|\\mathcal{N}_r(i)|} \\sum_{j \\in \\mathcal{N}_r(i)} \\mathbf{W}_r \\mathbf{z}_j \\right) + \\mathbf{W}_{\\text{res}} \\mathbf{z}_i', '1')),
    ('p', 'To address class imbalance, neural parameters are optimized using Normalized Mini-Batch Focal Loss {REF:LIN_FOCAL}:'),
    ('eq', ('\\mathcal{L}_{\\text{Focal}} = -\\frac{1}{|B|} \\sum_{i \\in B} \\alpha_t (1 - p_{t,i})^\\gamma \\log(p_{t,i})', '2')),
    ('p', 'where $\\alpha_t = 0.25$ and $\\gamma = 2.0$. In canonical BGNN {REF:IVANOV}, node coordinates $\\mathbf{Z}_0$ are updated by fitting trees directly to feature-level loss gradients $-\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. To maintain computational tractability over streaming data, this implementation evaluates a Rank-1 Projected Functional Coordinate Update ($M = 4$ outer rounds, $J = 2$ inner epochs). For outer round $m < M$, a scalar regression tree $T_m$ fits cross-entropy residuals $r_i = y_i - \\sigma(\\text{logit}_i)$, projected along empirical covariance direction $\\mathbf{d} = \\text{Normalize}(\\mathbb{E}[\\mathbf{Z}_0 \\cdot r_i]) \\in \\mathbb{R}^{49}$ and bounded to $[-\\delta_R, \\delta_R]$ with $\\delta_R = 0.20$ and $\\eta_{\\text{boost}} = 0.10$:'),
    ('eq', (['\\Delta \\mathbf{z}_i^{(m)} = \\text{clip}\\left( T_m(\\mathbf{x}_i) \\cdot \\mathbf{d}, -\\delta_R, \\delta_R \\right)', '\\mathbf{z}_i^{(m)} = \\mathbf{z}_i^{(m-1)} + \\eta_{\\text{boost}} \\cdot \\Delta \\mathbf{z}_i^{(m)}'], '3')),
    ('h2', 'D. Robust Inference and Inductive Cold-Start Mechanics'),
    ('p', 'To protect cold-start entities from missing structural context, a monotonic decision blending gate $\\beta_i$ arbitrates between GNN predictions and the base GBDT logits:'),
    ('eq', ('\\beta_i = \\mathbb{I}(\\text{connected}_i) \\cdot \\sigma\\left( \\text{softplus}(w_c) \\cdot \\text{card\\_mask}_i + \\text{softplus}(w_d) \\cdot \\text{dev\\_mask}_i + b_g \\right)', '4')),
    ('p', 'where $\\text{card\\_mask}_i, \\text{dev\\_mask}_i \\in \\{0, 1\\}$ denote historical card and device connectivity masks, and $\\mathbb{I}(\\text{connected}_i) = \\mathbb{I}(\\text{card\\_mask}_i + \\text{dev\\_mask}_i > 0)$. For the 796 strictly isolated holdout transactions (0.69%), $\\mathbb{I}(\\text{connected}_i) = 0 \\implies \\beta_i \\equiv 0$, providing an exact mathematical fallback to Stage 1. On connected nodes (99.31%), the gate converges to an average weighting of $\\bar{\\beta} = 0.8726$ toward the neural head.'),
    ('h2', 'E. Operational Financial Loss and Evaluation Protocol'),
    ('p', 'Models are assessed using non-interpolated trapezoid-free Average Precision (Raw AP as primary selection metric), Recall at fixed operational false-positive rates (Recall@0.1%FPR, Recall@1.0%FPR), and realized scenario-weighted financial loss ($L_{\\text{fin}}$):'),
    ('eq', (['L_{\\text{fin}} = \\sum_{i=1}^N [ y_i (1 - a_i) c_{\\text{miss}} A_i + a_i c_{\\text{review}} +', '(1 - y_i) a_i c_{\\text{friction}} + y_i a_i c_{\\text{residual}} A_i ]'], '5')),
    ('p', 'where $A_i$ is transaction amount, $a_i = \\mathbb{I}(p_i \\ge \\theta)$ is the alert decision, and financial outcomes are evaluated under unified cost coefficients ($c_{\\text{miss}} = 1.0$, $c_{\\text{review}} =$ \\$5.00, $c_{\\text{friction}} =$ \\$15.00, $c_{\\text{residual}} = 0.0$). The Primary Regime enforces a fixed 1.0% queue capacity constraint ($K = 1{,}151$ alerts), while the Sensitivity Regime applies an unconstrained Bayes-optimal threshold rule ($\\tau_i^* = 20.0 / (A_i + 15.0)$) evaluated on post-hoc isotonically calibrated probabilities alongside raw predictions. Statistical significance is evaluated using 1,000 paired 24-hour stationary block bootstrap resamples under Benjamini–Hochberg False Discovery Rate (FDR) control at $\\alpha = 0.05$ ($m = 9$ confirmatory tests).'),
    ('h1', 'IV. EMPIRICAL RESULTS AND FORENSIC DISCUSSION'),
    ('h2', 'A. Master Benchmark Performance on Sealed Out-of-Time Holdout'),
    ('p', 'Table II summarizes benchmark performance on the sealed Block 5 holdout ($N = 115,111$ evaluated post-embargo transactions, including 3,929 frauds). Under the primary 1.0% operational queue constraint ($K = 1{,}151$ alerts), Stage 1 incurs a realized Queue Loss of \\$503,596.43, intercepting 1,017 frauds, with a Raw Average Precision (AP) of 0.4814 (calibrated AP 0.4614) and ROC-AUC of 0.8731.'),
    ('p', 'Stage 2 (Graph-Augmented GBDT) achieves the strongest empirical performance across all evaluated systems. Adding 19 backward-looking relational features lowers Queue Loss to \\$490,253.81 (intercepting 1,026 frauds), saving \\$13,342.62 compared with Stage 1 (95% bootstrap CI: [+\\$3,870.88, +\\$22,682.03], $p_{\\text{adj}} = 0.0045$, confirming $H_1$ and $H_{3\\text{c}}$). Under post-hoc calibration and unconstrained Bayes thresholding ($\\tau_i^* = 20.0 / (A_i + 15.0)$), Stage 2 also records the lowest calibrated Bayes loss at \\$283,904.49 (saving \\$17,717.34 over Stage 1\'s \\$301,621.83), alongside the highest Raw AP (0.5252), calibrated AP (0.5088), ROC-AUC (0.9053), and Recall@1.0%FPR (43.90%).'),
    ('p', 'In comparison, the neural models obtain smaller queue reductions within the top 1% cap (\\$495,999.13 for Stage 3 and \\$496,737.84 for Stage 4, versus \\$503,596.43 for Stage 1) and incur higher calibrated Bayes losses (\\$305,803.72 for Stage 3, \\$312,534.70 for Stage 4, \\$307,424.67 for Stage 3n, and \\$306,866.66 for Stage 4n), with lower Raw AP (0.4772 for Stage 3; 0.4634 for Stage 4). When deployed without calibration under Bayes thresholding, uncalibrated Focal Loss scores generate alert surges to 23.59% (27,149 alerts) in Stage 3 and 23.15% (26,650 alerts, \\$498,365.00 in review and friction costs) in Stage 4, driving raw operational Bayes losses to \\$587,411.49 (Stage 3) and \\$576,749.67 (Stage 4)—a +\\$269.5K penalty relative to Stage 1 (\\$307,201.37 raw).'),
    ('table', ('TABLE II. MASTER BENCHMARK PERFORMANCE ON SEALED BLOCK 5 HOLDOUT (N = 115,111 evaluated post-embargo)', ['Model Architecture', 'Raw AP', 'Cal. AP', 'ROC-AUC', 'Rec@1.0%', 'Queue ($k)', 'Cal. Bayes ($k)'], [['Random Baseline', '0.0342', '0.0342', '0.5016', '1.04%', '$601.3', '$1,724.8'], ['Amount-Only Floor', '0.0359', '0.0359', '0.4788', '0.87%', '$545.5', '$462.7'], ['Relational-Only (19 cnt)', '0.3026', '0.3026', '0.8338', '25.66%', '$512.5', '$368.5'], ['Stage 1: Tabular LightGBM', '0.4814', '0.4614', '0.8731', '39.83%', '$503.6', '$301.6'], ['Stage 2: Graph GBDT*', '0.5252', '0.5088', '0.9053', '43.90%', '$490.3*', '$283.9*'], ['Stage 3: Res-GNN (Decoupled)', '0.4772', '0.4612', '0.8806', '39.96%', '$496.0', '$305.8'], ['Stage 4: Joint BGNN (Adapted)', '0.4634', '0.4478', '0.8756', '39.53%', '$496.7', '$312.5'], ['Stage 3n: Res-GNN No-Graph', '0.4763', '0.4726', '0.8830', '39.60%', '$498.8', '$307.4'], ['Stage 4n: BGNN Gate-Only', '0.4811', '0.4764', '0.8807', '39.98%', '$497.2', '$306.9']], [25, 9, 9, 10, 10, 11, 11])),
    ('h2', 'B. Confirmatory Hypothesis Testing and Statistical Verification'),
    ('p', 'Table III presents confirmatory hypothesis test results across 1,000 paired 24-hour stationary block bootstrap resamples under Benjamini–Hochberg False Discovery Rate (FDR) control ($\\alpha = 0.05$, $m = 9$ confirmatory tests):'),
    ('p', '1) Relational Feature Augmentation ($H_1$ & $H_{3\\text{c}}$ Supported): Adding 19 time-causal backward relational features to the tabular GBDT yields $\\Delta_{2-1} = +0.0438$ AP (95% bootstrap CI: $[+0.0377, +0.0504]$, $p_{\\text{adj}} = 0.0045$), while reducing 1% Queue Loss by \\$13,342.62 (95% CI: [+\\$3,870.88, +\\$22,682.03], $p_{\\text{adj}} = 0.0045$). On cards with prior history ($R_1$), the gain increases to $+0.0587$ AP ($p = 0.0010$).'),
    ('p', '2) Learned Structure vs. Engineered Features ($H_2$ Not Supported): Replacing explicit relational features with the two-stage Res-GNN results in $\\Delta_{3-2} = -0.0480$ AP (95% CI: $[-0.0549, -0.0415]$, $p_{\\text{adj}} = 1.0000$).'),
    ('p', '3) Functional Gradient Coupling ($H_3$ Not Supported): Within this rank-1 coordinate-update variant ($M=4, J=2$), alternating updates produce $\\Delta_{4-3} = -0.0138$ AP (95% CI: $[-0.0191, -0.0087]$, $p_{\\text{adj}} = 1.0000$), failing to improve upon the frozen Res-GNN.'),
    ('p', '4) Macro System Deployment ($H_4$ Not Supported): The adapted rank-1 BGNN achieves lower Raw AP than the standalone tabular LightGBM baseline ($\\Delta_{4-1} = -0.0181$ AP, 95% CI: $[-0.0257, -0.0117]$, $p_{\\text{adj}} = 1.0000$).'),
    ('table', ('TABLE III. PAIRED BOOTSTRAP HYPOTHESIS TESTING RESULTS (1,000 RESAMPLES, BH-FDR CONTROL, m = 9)', ['Hypothesis / Contrast', 'Code', 'Delta AP / Cost', '95% Bootstrap CI', 'p_adj', 'Verdict'], [['Relational Augmentation Value', 'S2 - S1', '+0.0438', '[+0.0377, +0.0504]', '0.0045', 'Supported (H1)'], ['Learned Structure vs. GBDT', 'S3 - S2', '-0.0480', '[-0.0549, -0.0415]', '1.0000', 'Null (H2)'], ['Joint Functional Coupling', 'S4 - S3', '-0.0138', '[-0.0191, -0.0087]', '1.0000', 'Null (H3)'], ['Macro System Deployment', 'S4 - S1', '-0.0181', '[-0.0257, -0.0117]', '1.0000', 'Null (H4)'], ['Message Passing Signal', 'S3 - S3n', '+0.0009', '[-0.0028, +0.0046]', '0.6975', 'Null (H2a)'], ['Connected Node Message Passing', 'S3 - S3n(c)', '+0.0003', '[-0.0032, +0.0037]', '0.7596', 'Null (H2b)'], ['Boosting Displacement Impact', 'S4 - S4n', '-0.0178', '[-0.0218, -0.0136]', '1.0000', 'Null (H3a)'], ['Monotonic Gating Alone', 'S4n - S3', '+0.0039', '[+0.0006, +0.0072]', '0.0300', 'Supported < Delta (H3b)'], ['Queue Loss at 1% Cap', 'S2 - S1 (QL)', '+$13,342.62', '[+$3,870.88, +$22,682.03]', '0.0045', 'Supported (H3c)'], ['Relational Value on Seen Cards', 'S2 - S1*', '+0.0587', '[+0.0501, +0.0668]', '--', 'Exploratory (R1)']], [25, 11, 11, 18, 9, 11])),
    ('h2', 'C. Diagnostic Visualizations'),
    ('p', 'The empirical dynamics across all stages are illustrated in Figs. 1 through 4.'),
    ('figure', ('data/figures/diagnostics_output/pr_curves_comparison.png', 'Fig. 1. Precision-recall curves across benchmark stages and reference baselines on sealed Block 5 holdout.', 84)),
    ('figure', ('data/figures/diagnostics_output/forest_plot_hypotheses.png', 'Fig. 2. Forest plot of bootstrap confidence intervals for pre-specified hypotheses H1–H4.', 84)),
    ('figure', ('data/figures/diagnostics_output/cohort_temporal_breakdown.png', 'Fig. 3. Stratified performance breakdown across entity cohorts (seen cards vs. cold-start) and longitudinal time slices.', 84)),
    ('figure', ('data/figures/diagnostics_output/calibration_curves_comparison.png', 'Fig. 4. Calibration reliability diagrams illustrating raw Focal Loss probability inflation versus isotonic recalibration.', 84)),
    ('h2', 'D. Forensic Failure Analysis and Breakdown Mechanisms'),
    ('p', 'To account for the null result ($H_4$ rejection) in this rank-1 coordinate-update variant, forensic analysis identifies four breakdown mechanisms:'),
    ('h3', '1) The Multiplicative Ratio Advantage vs. Additive Linear Smoothing:'),
    ('p', 'Gradient Boosted Decision Trees isolate fraud anomalies through discrete, non-linear split boundaries on ratio and velocity features. Feature gain importance analysis (Table IV) indicates that the highest split gains come from payment verification metrics and historical entity counters: V258 (gain 30,362.39), C1 (gain 23,408.28), P_emaildomain_seen_before (gain 22,058.14), card2_seen_before (gain 20,854.73), and card1_seen_before (gain 19,691.68). Exactly 10 relational features rank in the top 20, accounting for 47.86% of top-20 gain and 29.79% of total model split gain across all 19 counters. A decision tree isolates an anomalous deviation with a single split on an amount ratio or velocity counter. In contrast, unweighted mean-pooling layers compute linear additive combinations, $\\bar{\\mathbf{z}}_{\\mathcal{N}(i)} = \\frac{1}{|\\mathcal{N}(i)|} \\sum_{j \\in \\mathcal{N}(i)} \\mathbf{z}_j$. Because financial fraud networks exhibit heterophily {REF:ZHU_HOMOPHILY}, where illicit transactions appear alongside normal activity at popular merchants, spatial averaging acts as a low-pass filter that dampens localized transaction anomalies.'),
    ('table', ('TABLE IV. TOP FEATURE GAIN IMPORTANCE FROM STAGE 2 GRAPH-AUGMENTED GBDT', ['Rank', 'Feature Identifier', 'Split Gain', 'Feature Type', 'Operational Role'], [['1', 'V258', '30,362.39', 'Raw Tabular', 'Vesta engineered payment verification feature'], ['2', 'C1', '23,408.28', 'Raw Tabular', 'Aggregated transaction count attribute'], ['3', 'P_emaildomain_seen_before', '22,058.14', 'Relational Counter', 'Expanding count of purchaser email domain'], ['4', 'card2_seen_before', '20,854.73', 'Relational Counter', 'Historical count of card issuer ID'], ['5', 'card1_seen_before', '19,691.68', 'Relational Counter', 'Historical count of primary card BIN'], ['6', 'C13', '19,625.80', 'Raw Tabular', 'Aggregated transaction velocity counter'], ['7', 'C14', '19,230.97', 'Raw Tabular', 'Aggregated transaction frequency counter'], ['8', 'addr1_seen_before', '16,528.27', 'Relational Counter', 'Historical count of billing region / zip'], ['9', 'D2', '16,110.37', 'Raw Tabular', 'Temporal delta since previous transaction'], ['10', 'uid_device_seen_before', '14,814.72', 'Relational Counter', 'Historical observation count of device key']], [8, 25, 14, 16, 22])),
    ('h3', '2) Open-World Cold-Start Dynamics and Continuous Covariate Drift:'),
    ('p', 'Streaming financial networks experience continuous entity turnover. In the evaluated Block 5 holdout ($N = 115,111$), 35,197 transactions (30.58%) involve payment cards not observed during training (Fig. 3). Furthermore, it is critical to distinguish the 75.58% dataset-wide identity-join missingness from the 83.80% Block 5 routing into the missing-device supernode, leaving only 796 transactions (0.69%) strictly isolated across both card and device keys ($\\mathcal{N}(v) = \\emptyset$). While GBDTs handle unobserved categorical levels via learned default routing, unanchored GNN nodes route novel cards sharing weak device edges into smoothed embeddings {REF:AKLI}. Simultaneously, continuous feature distributions show temporal covariate drift across the 182-day period, reaching a maximum feature dimension drift of $9.564\\sigma$ in Block 5 driven by cumulative elapsed-time attributes (D11, D15, D10, D2) and verification spikes (V151, V152, V159). Whereas monotonic tree splits are invariant to positive coordinate expansions, unclipped $9.564\\sigma$ inputs distort linear neural projections and SELU activations.'),
    ('h3', '3) Latent Manifold Mutation vs. Pure Coordinate Boosting:'),
    ('p', 'In canonical BGNN {REF:IVANOV}, outer trees fit full $d_{\\text{emb}}$-dimensional gradients $-\\nabla_{\\mathbf{z}_i}\\mathcal{L}$. Here, displacing $\\mathbf{z}_i$ via rank-1 empirical-covariance projections across $M = 4$ outer rounds ($J = 2$ inner epochs) mutates the continuous coordinate manifold between training iterations. Decision trees in native feature space construct orthogonal boundaries aligned with physical attributes. Displacing latent coordinates across rounds alters reference axes, requiring subsequent inner GNN epochs to adapt to coordinate shifts rather than isolating residual risk. This dynamic explains why removing rank-1 coordinate displacements (Stage 4n) yields higher Average Precision than full alternating updates (Stage 4).'),
    ('h3', '4) Probability Miscalibration and Financial Loss Explosion:'),
    ('p', 'Although Stage 4 achieves a Raw AP of 0.4634, its uncalibrated predicted probabilities exhibit substantial calibration error (Fig. 4). Mini-Batch Focal Loss ($\\alpha = 0.25, \\gamma = 2.0$) shifts predicted scores upward: Stage 4 shows an Expected Calibration Error (ECE) of 0.0892 and a mean predicted probability of 12.69%, approximately 3.7 times the Block 5 fraud prevalence of 3.41%. Under unconstrained Bayes thresholding ($\\tau_i^* = 20.0 / (A_i + 15.0)$), uncalibrated scores trigger a 23.15% alert rate (26,650 alerts in Stage 4; 23.59% or 27,149 alerts in Stage 3), generating \\$498,365.00 in review and friction costs and escalating raw losses to \\$576,749.67 (Table II notes). Post-hoc isotonic calibration reduces Stage 4 ECE to 0.0077 and calibrated Bayes loss to \\$312,534.70, though monotonic rescaling cannot improve rank-order Average Precision.'),
    ('h2', 'E. Critical Deviation from Prevailing Transductive GNN Literature'),
    ('p', 'These empirical results contrast with reports in the graph learning literature where GNNs demonstrate large margins over tree baselines {REF:DOU}, {REF:WANG_GAT}, {REF:MAO}, {REF:TANG}. Three methodological factors account for this discrepancy:'),
    ('p', '1) Transductive Look-Ahead Leakage: Many benchmark studies construct static, bidirectional transaction graphs across full datasets. Transactions early in the sequence aggregate embeddings from subsequent transactions on the same cardholder account. When evaluated on dynamic graphs ($G_{<t}$) restricted to historical edges, this look-ahead advantage is eliminated {REF:VAN_BELLE}, {REF:AKLI}.'),
    ('p', '2) Random Partitioning Bias: Cross-validation on time-stamped transactions places events from the same account into both training and evaluation folds. Under chronological Out-of-Time evaluation with 30.58% cold-start card turnover, neural message passing cannot rely on seen historical neighborhoods.'),
    ('p', '3) Baseline Feature Completeness: Published evaluations often compare GNNs against basic decision tree configurations without historical relational features. When GBDTs incorporate backward-looking relational counters (Stage 2), they yield the lowest calibrated Bayes loss (\\$283,904.49), lowest Queue Loss (\\$490,253.81), and highest calibrated AP (0.5088), outperforming neural configurations with lower inference latency.'),
    ('h1', 'V. CONCLUSION AND RECOMMENDATIONS'),
    ('h2', 'A. Synthesis of Findings'),
    ('p', 'This study investigated joint graph-tabular learning for credit card fraud detection under strict streaming constraints. Explicit relational feature engineering in Stage 2 achieved the strongest overall performance ($H_1$ and $H_{3\\text{c}}$ confirmed, $\\Delta_{2-1} = +0.0438$, calibrated AP 0.5088, Queue Loss \\$490,253.81 saving \\$13,342.62, 95% CI: [+\\$3,870.88, +\\$22,682.03], $p_{\\text{adj}} = 0.0045$; calibrated Bayes loss \\$283,904.49 saving \\$17,717.34 over Stage 1\'s \\$301,621.83). Moreover, measured single-transaction CPU inference latency confirms Stage 2 executes in 0.123 ms median (0.181 ms $p_{95}$), running 3.1x to 4.5x faster than neural Stages 3 (0.387 ms median) and 4 (0.559 ms median). Scoped strictly to the evaluated rank-1 empirical-covariance coordinate-update BGNN variant ($M = 4, J = 2$), alternating functional updates did not improve upon frozen Res-GNN ($H_3$ null, $\\Delta_{4-3} = -0.0138$, $p_{\\text{adj}} = 1.0000$) or surpass standalone LightGBM ($H_4$ null, $\\Delta_{4-1} = -0.0181$, $p_{\\text{adj}} = 1.0000$). This negative result does not refute full high-dimensional BGNN formulations under broader tuning, but establishes practical failure boundaries for low-rank coordinate updates under streaming covariate drift.'),
    ('h2', 'B. Actionable Recommendations for Academic Benchmarking'),
    ('p', '1) Mandate Time-Causal Graph Construction ($G_{<t}$): Benchmarks for transaction fraud should prohibit static, undirected graphs across splits. Graph construction must restrict edge induction to historical events strictly preceding the query timestamp {REF:VAN_BELLE}, {REF:AKLI}.'),
    ('p', '2) Require Chronological Out-of-Time Holdouts: Financial risk models should be evaluated on forward chronological holdouts rather than random cross-validation, preserving realistic entity turnover and distribution shift.'),
    ('p', '3) Standardize Operational Loss Metrics: Evaluations should supplement ROC-AUC with Average Precision, Recall at fixed operational false-positive rates (0.1% and 1.0% FPR), and dollar loss under realistic review capacity limits.'),
    ('h2', 'C. Actionable Recommendations for Financial Risk Engineering'),
    ('p', '1) Prioritize Relational Feature Engineering in GBDTs: Production risk teams achieve stronger detection performance and lower operational complexity by computing rolling relational counts and ratio features for GBDT models, rather than deploying deep GNN pipelines.'),
    ('p', '2) Implement Cold-Start Fallback Routing: When hybrid models are deployed, scoring systems should route isolated or novel entities directly to tabular trees ($\\beta_i \\to 0$), bypassing unanchored neural layers.'),
    ('p', '3) Apply Isotonic Probability Calibration: Models optimized with asymmetric objectives like Focal Loss require post-hoc isotonic calibration before score-based queue triage, preventing queue overflows caused by probability inflation.'),
    ('h2', 'D. Future Research Vectors'),
    ('p', 'Future research vectors include: (1) integrating continuous-time memory networks (Temporal Graph Networks {REF:ROSSI}) with low-latency caching; (2) formulating differentiable ratio and product-norm layers in GNNs to model scale-invariant deviations without smoothing; and (3) exploring logit-space functional boosting to avoid continuous coordinate displacements.'),
]

# ------------------------------------------------------------------------------
# 6. Body Section Builder
# ------------------------------------------------------------------------------
def build_ieee_body_section(doc):
    """
    Constructs Section 2: Continuous 2-Column Body Layout with Abstract,
    Keywords, Sections I–V, Tables, Figures, and References.
    """
    sec2 = doc.add_section(WD_SECTION.CONTINUOUS)
    sec2.page_width = Mm(210)
    sec2.page_height = Mm(297)
    sec2.top_margin = Mm(19)
    sec2.bottom_margin = Mm(25.4)
    sec2.left_margin = Mm(16)
    sec2.right_margin = Mm(16)
    
    # Configure 2 columns: w:cols num="2" space="360" (0.25 in / 6.35 mm)
    sectPr = sec2._sectPr
    cols = sectPr.find(qn('w:cols'))
    if cols is None:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    cols.set(qn('w:num'), '2')
    cols.set(qn('w:space'), '360')

    # Strictly unlink and empty footers to prevent pagination
    sec2.header.is_linked_to_previous = False
    sec2.footer.is_linked_to_previous = False
    for p in sec2.header.paragraphs:
        p.text = ""
    for p in sec2.footer.paragraphs:
        p.text = ""

    # --------------------------------------------------------------------------
    # Abstract (9 pt Bold Times New Roman, Zero Math/Symbols per Appendix 12)
    # --------------------------------------------------------------------------
    p_abs = doc.add_paragraph()
    p_abs.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_abs.paragraph_format.line_spacing = 1.0
    p_abs.paragraph_format.space_before = Pt(4)
    p_abs.paragraph_format.space_after = Pt(4)
    
    r_abs_lead = p_abs.add_run("Abstract—")
    r_abs_lead.font.name = "Times New Roman"
    r_abs_lead.font.size = Pt(9)
    r_abs_lead.bold = True
    r_abs_lead.italic = True
    
    abstract_text = (
        "Credit card fraud detection in payment clearing networks requires reliable risk "
        "discrimination under sub-100-millisecond latency budgets and delayed chargeback feedback. While Gradient "
        "Boosted Decision Trees remain the primary operational choice for tabular transaction data, Graph Neural Networks "
        "are widely studied for modeling relational connections across cards and devices. However, many published "
        "graph benchmarks use static, transductive partitions that introduce temporal look-ahead "
        "leakage and ignore the open-world cold-start conditions of production streams. This paper evaluates an adapted Boost then "
        "Convolve architecture that combines continuous tabular embeddings, relation-aware message passing on a strictly "
        "time-causal dynamic graph, monotonic decision blending, and rank-1 alternating functional-gradient optimization. "
        "Across the IEEE-CIS benchmark dataset (590,540 transactions over 182 days) and a sealed Out-of-Time holdout "
        "(115,111 evaluated post-embargo transactions, Block 5), results show a clear separation between tree-based and neural methods. "
        "Under a fixed one percent operational review queue, handcrafted relational feature engineering in Stage 2 achieves the lowest "
        "queue loss at USD 490,253.81, saving USD 13,342.62 compared with the Stage 1 tabular baseline "
        "(95 percent confidence interval USD 3,870.88 to USD 22,682.03, adjusted p-value 0.0045) with a Raw Average Precision of 0.5252, "
        "and achieves the lowest calibrated Bayes loss at USD 283,904.49 versus USD 301,621.83 for Stage 1 (saving USD 17,717.34). "
        "In contrast, the evaluated rank-1 Boost then Convolve variant decreases Average Precision by 0.0181 relative to Stage 1, "
        "and uncalibrated Focal Loss predictions raise unconstrained Bayes loss by USD 269,548.30 (USD 576,749.67 versus USD 307,201.37) "
        "through excessive false alerts. Forensic analysis identifies three primary breakdown mechanisms: "
        "spatial smoothing of ratio anomalies, an inductive cold-start bottleneck from unseen entities, and probability miscalibration."
    )
    r_abs_body = p_abs.add_run(abstract_text)
    r_abs_body.font.name = "Times New Roman"
    r_abs_body.font.size = Pt(9)
    r_abs_body.bold = True

    # --------------------------------------------------------------------------
    # Keywords (9 pt Bold Italic Times New Roman per Appendix 12)
    # --------------------------------------------------------------------------
    p_key = doc.add_paragraph()
    p_key.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_key.paragraph_format.line_spacing = 1.0
    p_key.paragraph_format.space_before = Pt(2)
    p_key.paragraph_format.space_after = Pt(10)
    
    r_key_lead = p_key.add_run("Keywords—")
    r_key_lead.font.name = "Times New Roman"
    r_key_lead.font.size = Pt(9)
    r_key_lead.bold = True
    r_key_lead.italic = True
    
    r_key_body = p_key.add_run("credit card fraud detection, boosted graph neural networks, tabular representation learning, time-causal graph, out-of-time evaluation, financial risk minimization.")
    r_key_body.font.name = "Times New Roman"
    r_key_body.font.size = Pt(9)
    r_key_body.bold = True
    r_key_body.italic = True

    # --------------------------------------------------------------------------
    # Dynamic Citation Resolution Pipeline
    # --------------------------------------------------------------------------
    cite_map = {}
    ordered_refs = []
    
    def resolve_ref(match):
        key = match.group(1)
        if key not in REFERENCE_DB:
            raise KeyError(f"Reference key '{key}' not found in REFERENCE_DB catalog!")
        if key not in cite_map:
            idx = len(cite_map) + 1
            cite_map[key] = idx
            ordered_refs.append((idx, key, REFERENCE_DB[key]))
        return f"[{cite_map[key]}]"

    # Render Document Elements
    for elem_type, payload in DOCUMENT_FLOW:
        if elem_type == "h1":
            resolved_h1 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            add_heading_1(doc, resolved_h1)
        elif elem_type == "h2":
            resolved_h2 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            add_heading_2(doc, resolved_h2)
        elif elem_type == "h3":
            resolved_h3 = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            add_heading_3(doc, resolved_h3)
        elif elem_type == "p":
            resolved_p = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, payload)
            add_body_p(doc, resolved_p)
        elif elem_type == "eq":
            latex_code, eq_num = payload
            add_ieee_display_eq(doc, latex_code, eq_num)
        elif elem_type == "figure":
            img_path, caption, width_mm = payload
            resolved_caption = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, caption)
            add_ieee_figure(doc, img_path, resolved_caption, width_mm)
        elif elem_type == "table":
            cap_str, headers, rows, col_widths = payload
            resolved_cap = re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, cap_str)
            resolved_headers = [re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, h) for h in headers]
            resolved_rows = [
                [re.sub(r'\{REF:([A-Za-z0-9_]+)\}', resolve_ref, str(c)) for c in row]
                for row in rows
            ]
            add_ieee_table(doc, resolved_cap, resolved_headers, resolved_rows, col_widths)

    # --------------------------------------------------------------------------
    # REFERENCES (IEEE FORMAT, STRICT SEQUENTIAL [1]-[N])
    # --------------------------------------------------------------------------
    add_heading_1(doc, "REFERENCES")
    
    print(f"  Rendering {len(ordered_refs)} sequentially ordered IEEE references...")
    for idx, key, full_ref in ordered_refs:
        p_r = doc.add_paragraph()
        p_r.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p_r.paragraph_format.line_spacing = 1.0
        p_r.paragraph_format.space_before = Pt(0)
        p_r.paragraph_format.space_after = Pt(2)
        p_r.paragraph_format.left_indent = Mm(3.5)
        p_r.paragraph_format.first_line_indent = Mm(-3.5)
        
        r_ref = p_r.add_run(f"[{idx}] {full_ref}")
        r_ref.font.name = "Times New Roman"
        r_ref.font.size = Pt(8)

# ------------------------------------------------------------------------------
# 7. Main Execution Function
# ------------------------------------------------------------------------------
def main():
    print("="*80)
    print("STARTING DETERMINISTIC UTP IEEE TECHNICAL PAPER BUILD PIPELINE")
    print("="*80)
    
    doc = docx.Document()
    doc.core_properties.author = "Ieman Ihsan Bin Alwi"
    doc.core_properties.title = "A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud Detection Using an Adapted Boost then Convolve Architecture"
    doc.core_properties.subject = "IEEE Technical Paper, Universiti Teknologi PETRONAS"

    
    print("--> Building Section 1: Single-Column Title and Author Header...")
    build_ieee_header(doc)
    
    print("--> Building Section 2: Continuous 2-Column Body Layout...")
    build_ieee_body_section(doc)
    
    output_filename = os.path.join("docs", "manuscripts", "IEEE_Technical_Paper_Ieman_Ihsan_25005466.docx")
    print(f"--> Saving Publication-Grade IEEE Technical Paper to {output_filename}...")
    doc.save(output_filename)
    
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print("="*80)
    print(f"IEEE TECHNICAL PAPER GENERATED SUCCESSFULLY!")
    print(f"Output File: {output_filename}")
    print(f"File Size:   {file_size_mb:.2f} MB")
    print("="*80)

if __name__ == "__main__":
    main()
