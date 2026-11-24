# -*- coding: utf-8 -*-
"""
build_humanized_dissertation.py
Generates Dissertation_Ieman_Ihsan_25005466_UTP_humanized.docx using build_full_submission_dissertation.py logic.
"""

import os
import docx
from docx.shared import Mm
from docx.enum.section import WD_SECTION

import build_full_submission_dissertation as bsd

def build_humanized_dissertation(output_filename="Dissertation_Ieman_Ihsan_25005466_UTP_humanized.docx"):
    print("="*80)
    print("STARTING HUMANIZED UTP MASTER DISSERTATION BUILD PIPELINE")
    print("="*80)
    
    doc = docx.Document()
    doc.core_properties.author = "Ieman Ihsan Bin Alwi"
    doc.core_properties.title = "A Time-Causal Joint Graph-Tabular Learning Framework for Credit Card Fraud Detection Using an Adapted Boost then Convolve Architecture"
    doc.core_properties.subject = "Master of Science in Data Science Dissertation, Universiti Teknologi PETRONAS"

    print("--> Step 1: Generating Title Page...")
    bsd.build_title_page(doc)

    print("--> Step 2: Generating Front Matter Section...")
    bsd.build_front_matter_section(doc)

    print("--> Step 3: Generating Main Body Section...")
    sec_main = doc.add_section(WD_SECTION.NEW_PAGE)
    sec_main.page_width = Mm(210)
    sec_main.page_height = Mm(297)
    sec_main.left_margin = Mm(40)
    sec_main.right_margin = Mm(25)
    sec_main.top_margin = Mm(25)
    sec_main.bottom_margin = Mm(25)
    sec_main.footer.is_linked_to_previous = False
    bsd.add_page_number_to_footer(sec_main.footer)
    bsd.set_section_numbering(sec_main, fmt="decimal", start=1)

    # Chapter 1: Introduction
    bsd.parse_and_append_chapter(
        doc,
        'Scripts/chapter_1_introduction.md',
        1,
        "INTRODUCTION"
    )

    # Chapter 2: Literature Review
    bsd.parse_and_append_chapter(
        doc,
        'Scripts/chapter_2_literature_review.md',
        2,
        "LITERATURE REVIEW AND THEORETICAL FOUNDATIONS"
    )

    # Chapter 3: Methodology
    bsd.parse_and_append_chapter(
        doc,
        'Scripts/chapter_3_methodology.md',
        3,
        "RESEARCH METHODOLOGY AND SYSTEM FORMULATION"
    )

    # Chapter 4: Results and Discussion
    bsd.parse_and_append_chapter(
        doc,
        'Scripts/chapter_4_results_and_discussion.md',
        4,
        "EMPIRICAL RESULTS AND FORENSIC DISCUSSION"
    )

    # Chapter 5: Conclusion and Recommendations
    bsd.parse_and_append_chapter(
        doc,
        'Scripts/chapter_5_conclusion_and_recommendations.md',
        5,
        "CONCLUSION AND RECOMMENDATIONS"
    )

    # References and Appendices
    bsd.build_references_and_appendices(doc)

    print(f"--> Step 4: Saving Complete Dissertation to {output_filename}...")
    doc.save(output_filename)
    
    file_size_mb = os.path.getsize(output_filename) / (1024 * 1024)
    print("="*80)
    print(f"DISSERTATION GENERATION COMPLETED SUCCESSFULLY!")
    print(f"Output File: {output_filename}")
    print(f"File Size:   {file_size_mb:.2f} MB")
    print("="*80)

if __name__ == "__main__":
    build_humanized_dissertation()
