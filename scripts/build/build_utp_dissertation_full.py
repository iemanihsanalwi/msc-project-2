"""
build_utp_dissertation_full.py
================================================================================
Deterministic Build Pipeline for Submission-Ready UTP Master Dissertation:
"A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud
 Detection Using an Adapted Boost then Convolve Architecture"

Author: Ieman Ihsan Bin Alwi (Student ID: 25005466)
Supervisor: Prof. Dr. Haruhiro Fujita
Degree: Master of Science in Data Science
University: Universiti Teknologi PETRONAS
Target Output: Dissertation_Ieman_Ihsan_25005466_UTP.docx
================================================================================
"""

import os
import sys
import re
import docx
from docx.shared import Mm, Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from lxml import etree
import latex2mathml.converter

# ------------------------------------------------------------------------------
# 1. OMML Equation Transformation
# ------------------------------------------------------------------------------
XSLT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MML2OMML.XSL')
xslt_tree = etree.parse(XSLT_PATH)
mml_transform = etree.XSLT(xslt_tree)

def latex_to_omml(latex_code):
    s = latex_code.strip()
    s = s.replace(r'\text{', r'\mathrm{')
    s = s.replace(r'\mathbf{', r'\mathbf{')
    s = s.replace(r'\mathbb{I}', r'\mathrm{I}')
    s = s.replace(r'\,', ' ')
    s = s.replace(r'\;', ' ')
    s = s.replace(r'\quad', ' ')
    s = s.replace(r'\qquad', '   ')
    s = s.replace(r'\land', r'\wedge')
    s = s.replace(r'\lor', r'\vee')
    s = s.replace(r'\oslash', r'/')
    
    mathml_str = latex2mathml.converter.convert(s)
    mathml_tree = etree.fromstring(mathml_str.encode('utf-8'))
    omml_tree = mml_transform(mathml_tree)
    return omml_tree.getroot()

# ------------------------------------------------------------------------------
# 2. Strict APA Reference Catalog (37 Entries)
# ------------------------------------------------------------------------------
REFERENCES_CATALOG = [
    "Akli, H., Hamrioui, S., & Stéphan, I. (2026). Heterophily outlier temporal aware graph neural network for online fraud detection. Expert Systems with Applications, 331, 133278. https://doi.org/10.1016/j.eswa.2026.133278",
    "Arik, S. O., & Pfister, T. (2021). TabNet: Attentive interpretable tabular learning. In Proceedings of the AAAI Conference on Artificial Intelligence, 35(8), 6679–6687. https://doi.org/10.1609/aaai.v35i8.16878",
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785",
    "Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., & Yu, P. S. (2020). Enhancing graph neural network-based fraud detectors against camouflaged fraudsters. In Proceedings of the 29th ACM International Conference on Information and Knowledge Management (CIKM '20) (pp. 315–324). ACM. https://doi.org/10.1145/3340531.3411903",
    "Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232. https://doi.org/10.1214/aos/1013203451",
    "Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. In Proceedings of the 34th International Conference on Machine Learning (ICML '17) (pp. 1263–1272). PMLR.",
    "Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? Advances in Neural Information Processing Systems, 35, 507–520.",
    "Halvaei, M., Shamsinejad, E., & Nikravan, M. (2020). Delayed feedback and chargeback latency in online payment fraud detection. Journal of Financial Crime, 27(3), 819–834. https://doi.org/10.1108/JFC-11-2019-0145",
    "Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. Advances in Neural Information Processing Systems, 30, 1024–1034.",
    "Hazimeh, H., Pydi, M. S., & Mazumder, R. (2020). The Tree Ensemble Layer: Differentiability meets conditional computation. In Proceedings of the 37th International Conference on Machine Learning (ICML '20) (pp. 4138–4148). PMLR.",
    "Ivanov, S., & Prokhorenkova, L. (2021). Boost then Convolve: Gradient boosting meets graph neural networks. In International Conference on Learning Representations (ICLR 2021). https://doi.org/10.48550/arXiv.2101.08543",
    "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30, 3146–3154.",
    "Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. In International Conference on Learning Representations (ICLR 2017). https://doi.org/10.48550/arXiv.1609.02907",
    "Lahcen, T., Amadiaz, Y., Bahbib, M., & Nait Sidi Moh, A. (2026). A heterogeneous graph representation learning approach for fraud detection in supply chain networks. Intelligent Systems with Applications, 31, 200686. https://doi.org/10.1016/j.iswa.2026.200686",
    "Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., & He, Q. (2021). Pick and choose: A GNN-based imbalanced learning approach for fraud detection. In Proceedings of the Web Conference 2021 (WWW '21) (pp. 3168–3177). ACM. https://doi.org/10.1145/3442381.3449989",
    "Ma, X., Wu, J., Xue, S., Yang, J., Zhou, C., Sheng, Q. Z., Xiong, H., & Akoglu, L. (2021). A comprehensive survey on graph anomaly detection with deep learning. IEEE Transactions on Knowledge and Data Engineering, 35(12), 12012–12038. https://doi.org/10.1109/TKDE.2021.3118815",
    "Mao, X., & Zhu, H. (2026). Enhancing financial fraud detection with graph neural network and ensemble learning: Insights from Related Party Transactions network. International Review of Economics & Finance, 93, 105217. https://doi.org/10.1016/j.iref.2026.105217",
    "Mohawesh, R., Maqsood, S., Bany Salameh, H., & Elrefae, G. (2026). Leveraging GraphSAGE and large language models with cross-attention for transaction fraud detection: Evidence from credit card and PaySim datasets. Information Processing & Management, 63(1), 104860. https://doi.org/10.1016/j.ipm.2026.104860",
    "Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2019). Weisfeiler and Leman go neural: Higher-order graph neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 33(01), 4602–4609. https://doi.org/10.1609/aaai.v33i01.33014602",
    "Nilson Report. (2024). Card fraud losses worldwide: 2023–2028 (Issue 1254). HSN Consultants.",
    "Onwudebelu, U., Fasola, O. O., Ugonna, E. C., & Woods, N. C. (2026). A behavior-aware intelligent decision framework for real-time credit card fraud detection using deep learning and adaptive user profiling. Engineering Applications of Artificial Intelligence, 181, 115641. https://doi.org/10.1016/j.engappai.2026.115641",
    "Popov, S., Morozov, S., & Babenko, A. (2020). Neural Oblivious Decision Ensembles for deep learning on tabular data. In International Conference on Learning Representations (ICLR 2020). https://doi.org/10.48550/arXiv.1909.06312",
    "Pourhabibi, T., Ong, K.-L., Kam, B. H., & Boo, Y. L. (2020). Fraud detection: A systematic literature review of graph-based anomaly detection approaches. Decision Support Systems, 133, 113303. https://doi.org/10.1016/j.dss.2020.113303",
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. Advances in Neural Information Processing Systems, 31, 6638–6648.",
    "Rahman, M., Islam, S., & Karim, R. (2023). Residual Graph Neural Networks over static tabular embeddings. ACM Computing Surveys, 55(8), 1–28. https://doi.org/10.1145/3547139",
    "Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). Temporal Graph Networks for deep learning on dynamic graphs. In NeurIPS 2020 Workshop on Deep Learning on Graphs. https://doi.org/10.48550/arXiv.2006.10637",
    "Tang, Y., & Liang, Y. (2024). Credit card fraud detection based on federated graph learning. Expert Systems with Applications, 256, 124979. https://doi.org/10.1016/j.eswa.2024.124979",
    "Van Belle, R., Baesens, B., & De Weerdt, J. (2023). CATCHM: A novel network-based credit card fraud detection method using node representation learning. Decision Support Systems, 164, 113866. https://doi.org/10.1016/j.dss.2022.113866",
    "Van Vlasselaer, V., Eliassi-Rad, T., Akoglu, L., Snoeck, M., & Baesens, B. (2015). GOTCHA! Network-based fraud detection for social security fraud. Management Science, 63(9), 3090–3110. https://doi.org/10.1287/mnsc.2016.2489",
    "Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph attention networks. In International Conference on Learning Representations (ICLR 2018). https://doi.org/10.48550/arXiv.1710.10903",
    "Wang, D., Lin, J., Cui, P., Jia, Q., Wang, Z., Fang, Y., Yu, Q., Zhou, J., Yang, S., & Qi, Y. (2019). A semi-supervised graph attentive network for financial fraud detection. In 2019 IEEE International Conference on Data Mining (ICDM) (pp. 598–607). IEEE. https://doi.org/10.1109/ICDM.2019.00070",
    "Wang, N., Zhang, X., & Liu, M. (2026). DG-GCN: A dual similarity-guided graph convolutional network for imbalanced node classification. Neurocomputing, 580, 134284. https://doi.org/10.1016/j.neucom.2026.134284",
    "Weber, M., Domeniconi, G., Chen, J., Weidele, D. K. I., Bellei, C., Robinson, T., & Leiserson, C. E. (2019). Anti-money laundering in Bitcoin: Experimenting with graph convolutional networks for financial forensics. In KDD 2019 Workshop on Anomaly Detection in Finance. https://doi.org/10.48550/arXiv.1908.02591",
    "Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2018). How powerful are graph neural networks? In International Conference on Learning Representations (ICLR 2019). https://doi.org/10.48550/arXiv.1810.00826",
    "Zhang, J., Fang, Z., Tang, Y., & Ni, L. (2026). TS-GNN: Modeling temporal-semantic discrepancy for graph-based credit card fraud detection. Expert Systems with Applications, 331, 133233. https://doi.org/10.1016/j.eswa.2026.133233",
    "Zhu, J., Yan, Y., Zhao, L., Heimann, M., Akoglu, L., & Koutra, D. (2020). Beyond homophily in graph neural networks: Current limitations and effective designs. Advances in Neural Information Processing Systems, 33, 7793–7804.",
    "Zhu, R., Wang, S., Li, Y., & Chen, C. (2026). DISCO: Decoupling representation learning and risk control for reliable credit card fraud detection. Decision Support Systems, 178, 114717. https://doi.org/10.1016/j.dss.2026.114717"
]
REFERENCES_CATALOG = sorted(REFERENCES_CATALOG)

# ------------------------------------------------------------------------------
# 3. Document Styling Helpers
# ------------------------------------------------------------------------------
def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_page_number_to_footer(footer):
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.text = ""
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    fldSimple = OxmlElement('w:fldSimple')
    fldSimple.set(qn('w:instr'), 'PAGE')
    p._p.append(fldSimple)

def set_section_numbering(section, fmt="decimal", start=1):
    sectPr = section._sectPr
    pgNumType = sectPr.find(qn('w:pgNumType'))
    if pgNumType is None:
        pgNumType = OxmlElement('w:pgNumType')
        sectPr.append(pgNumType)
    pgNumType.set(qn('w:fmt'), fmt)
    if start is not None:
        pgNumType.set(qn('w:start'), str(start))

def add_heading_1_chapter(doc, chapter_num_str, chapter_title_str):
    p1 = doc.add_paragraph()
    p1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p1.paragraph_format.space_before = Pt(70.87) # 25mm space before + 25mm margin = 50mm from top
    p1.paragraph_format.space_after = Pt(6)
    p1.paragraph_format.line_spacing = 1.0
    r1 = p1.add_run(chapter_num_str.upper())
    r1.font.name = "Times New Roman"
    r1.font.size = Pt(14)
    r1.bold = True

    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p2.paragraph_format.space_before = Pt(0)
    p2.paragraph_format.space_after = Pt(24) # double spacing below
    p2.paragraph_format.line_spacing = 1.0
    r2 = p2.add_run(chapter_title_str.upper())
    r2.font.name = "Times New Roman"
    r2.font.size = Pt(14)
    r2.bold = True

def add_heading_2_major(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text.upper())
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = True
    return p

def add_heading_3_secondary(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.5
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    r.bold = True
    return p

def add_display_equation(doc, latex_str, eq_num=""):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Mm(130)
    table.columns[1].width = Mm(15)
    
    cell_eq = table.cell(0, 0)
    cell_num = table.cell(0, 1)
    
    for cell in [cell_eq, cell_num]:
        tcPr = cell._tc.get_or_add_tcPr()
        borders_xml = parse_xml(r'<w:tcBorders %s><w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>' % nsdecls('w'))
        tcPr.append(borders_xml)
        
    p_eq = cell_eq.paragraphs[0]
    p_eq.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_eq.paragraph_format.space_before = Pt(6)
    p_eq.paragraph_format.space_after = Pt(6)
    p_eq.paragraph_format.line_spacing = 1.0
    
    try:
        omml = latex_to_omml(latex_str)
        p_eq._p.append(omml)
    except Exception as e:
        r = p_eq.add_run(latex_str)
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r.italic = True
        
    p_num = cell_num.paragraphs[0]
    p_num.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_num.paragraph_format.space_before = Pt(6)
    p_num.paragraph_format.space_after = Pt(6)
    p_num.paragraph_format.line_spacing = 1.0
    if eq_num:
        r_num = p_num.add_run(f"({eq_num})")
        r_num.font.name = "Times New Roman"
        r_num.font.size = Pt(12)
        
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(2)
    p_after.paragraph_format.space_after = Pt(12)

def add_formatted_body_paragraph(doc, text, is_bullet=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(18) # 3 spacing between paragraphs per Section 6.4(b)
    
    if is_bullet:
        p.paragraph_format.left_indent = Mm(10)
        p.paragraph_format.first_line_indent = Mm(-5)
        r_b = p.add_run("•  ")
        r_b.font.name = "Times New Roman"
        r_b.font.size = Pt(12)
        
    tokens = re.split(r'(\$[^\$]+\$)', text)
    for token in tokens:
        if token.startswith('$') and token.endswith('$') and len(token) > 2:
            latex_code = token[1:-1]
            try:
                omml = latex_to_omml(latex_code)
                p._p.append(omml)
            except Exception as e:
                r = p.add_run(latex_code)
                r.font.name = "Times New Roman"
                r.font.size = Pt(12)
                r.italic = True
        else:
            if not token:
                continue
            sub_tokens = re.split(r'(\*\*[^\*]+\*\*|\*[^\*]+\*)', token)
            for st in sub_tokens:
                if not st:
                    continue
                if st.startswith('**') and st.endswith('**') and len(st) > 4:
                    r = p.add_run(st[2:-2])
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(12)
                    r.bold = True
                elif st.startswith('*') and st.endswith('*') and len(st) > 2:
                    r = p.add_run(st[1:-1])
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(12)
                    r.italic = True
                else:
                    r = p.add_run(st)
                    r.font.name = "Times New Roman"
                    r.font.size = Pt(12)
    return p

def add_native_academic_table(doc, caption_text, headers, data_rows, col_widths_mm=None):
    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(24) # 3 spacing above
    p_cap.paragraph_format.space_after = Pt(8)
    p_cap.paragraph_format.line_spacing = 1.0
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(12)
    r_cap.bold = True

    num_rows = len(data_rows) + 1
    num_cols = len(headers)
    table = doc.add_table(rows=num_rows, cols=num_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    if col_widths_mm and len(col_widths_mm) == num_cols:
        for idx, width in enumerate(col_widths_mm):
            table.columns[idx].width = Mm(width)

    # Style Header
    header_tr = table.rows[0]._tr
    trPr = header_tr.get_or_add_trPr()
    trPr.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls('w')))
    trPr.append(parse_xml(r'<w:cantSplit %s/>' % nsdecls('w')))
    
    for c_idx, header_text in enumerate(headers):
        cell = table.cell(0, c_idx)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
        tcPr = cell._tc.get_or_add_tcPr()
        borders_xml = parse_xml(r'''<w:tcBorders %s>
            <w:top w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="12" w:space="0" w:color="000000"/>
            <w:right w:val="none"/>
        </w:tcBorders>''' % nsdecls('w'))
        tcPr.append(borders_xml)
        shd_xml = parse_xml(r'<w:shd %s w:fill="F4F4F4"/>' % nsdecls('w'))
        tcPr.append(shd_xml)

        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(header_text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(10)
        r.bold = True

    # Style Rows
    for r_idx, row_data in enumerate(data_rows):
        is_last_row = (r_idx == len(data_rows) - 1)
        row_tr = table.rows[r_idx + 1]._tr
        trPr = row_tr.get_or_add_trPr()
        trPr.append(parse_xml(r'<w:cantSplit %s/>' % nsdecls('w')))

        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_margins(cell, top=80, bottom=80, left=140, right=140)
            tcPr = cell._tc.get_or_add_tcPr()
            
            bottom_val = "single" if is_last_row else "none"
            bottom_sz = "12" if is_last_row else "0"
            borders_xml = parse_xml(r'''<w:tcBorders %s>
                <w:top w:val="none"/>
                <w:left w:val="none"/>
                <w:bottom w:val="%s" w:sz="%s" w:space="0" w:color="000000"/>
                <w:right w:val="none"/>
            </w:tcBorders>''' % (nsdecls('w'), bottom_val, bottom_sz))
            tcPr.append(borders_xml)

            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.0
            
            text_str = str(val)
            is_bold = "[CHAMP]" in text_str or "**" in text_str or "SUPPORTED" in text_str or "Reference" in text_str
            clean_str = text_str.replace("**", "").replace("[CHAMP]", "[Champion]")
            r = p.add_run(clean_str)
            r.font.name = "Times New Roman"
            r.font.size = Pt(9.5)
            r.bold = is_bold

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(6)
    p_after.paragraph_format.space_after = Pt(24) # 3 spacing below
    return table

def add_figure_with_caption(doc, image_path, caption_text, width_mm=140):
    if not os.path.exists(image_path):
        print(f"WARNING: Image path not found: {image_path}")
        return

    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(24) # 3 spacing above
    p_img.paragraph_format.space_after = Pt(6)
    p_img.paragraph_format.line_spacing = 1.0
    r_img = p_img.add_run()
    r_img.add_picture(image_path, width=Mm(width_mm))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(4)
    p_cap.paragraph_format.space_after = Pt(24) # 3 spacing below
    p_cap.paragraph_format.line_spacing = 1.0
    r_cap = p_cap.add_run(caption_text)
    r_cap.font.name = "Times New Roman"
    r_cap.font.size = Pt(12)
    r_cap.bold = True

def add_pseudocode_block(doc, title, code_lines):
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(14)
    p_title.paragraph_format.space_after = Pt(4)
    r_t = p_title.add_run(title)
    r_t.font.name = "Times New Roman"
    r_t.font.size = Pt(11)
    r_t.bold = True
    
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    tcPr = cell._tc.get_or_add_tcPr()
    borders_xml = parse_xml(r'''<w:tcBorders %s>
        <w:top w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>
        <w:left w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>
        <w:bottom w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>
        <w:right w:val="single" w:sz="6" w:space="0" w:color="B0B0B0"/>
    </w:tcBorders>''' % nsdecls('w'))
    tcPr.append(borders_xml)
    shd_xml = parse_xml(r'<w:shd %s w:fill="FAFAFA"/>' % nsdecls('w'))
    tcPr.append(shd_xml)
    
    p = cell.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    
    for idx, line in enumerate(code_lines):
        r = p.add_run(line + ("\n" if idx < len(code_lines)-1 else ""))
        r.font.name = "Times New Roman"
        r.font.size = Pt(8) # 8 pt per Section 6.4(b)

    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(18)

print("Setup completed. Building document...")
