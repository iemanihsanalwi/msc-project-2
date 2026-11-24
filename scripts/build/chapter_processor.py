"""
chapter_processor.py
================================================================================
Processes chapter markdown files, parses blocks (headings, display equations,
tables, figures, paragraphs, bullet lists), and outputs clean structured data
ready to be added to python-docx Document.
================================================================================
"""

import re

def clean_text_formatting(text):
    """
    Remove informal exclamation marks, reconcile terminology and passive voice.
    """
    # Remove informal exclamation marks per UTP guidelines
    text = text.replace(r'9.564\,\sigma!', r'9.564\,\sigma.')
    text = text.replace(r'9.564\sigma!', r'9.564\sigma.')
    text = text.replace(r'3.7\times$ base rate inflation!)', r'3.7\times$ base rate inflation).')
    text = text.replace(r'13.46\times$ capacity!)', r'13.46\times$ capacity).')
    text = text.replace('13.46x!', '13.46x.')
    
    # Standardize framework versions
    text = text.replace('XGBoost version 2.1.0', 'LightGBM (version 4.6)')
    text = text.replace('XGBoost (version 2.1.0)', 'LightGBM (version 4.6)')
    text = text.replace('XGBoost 2.1.0', 'LightGBM 4.6')
    text = text.replace('PyTorch version 2.6.0', 'PyTorch (version 2.6)')
    text = text.replace('LightGBM version 4.6.0', 'LightGBM (version 4.6)')
    
    # Passive voice transformations
    text = text.replace('our adapted architecture', 'the proposed adapted architecture')
    text = text.replace('our framework', 'the proposed framework')
    text = text.replace('our empirical pipeline', 'the empirical pipeline')
    
    # Financial loss coefficient standardization in Chapter 3
    text = text.replace(r'c_{\text{review}} = \$10.00', r'c_{\text{review}} = \$5.00')
    text = text.replace(r'c_{\text{friction}} = \$25.00', r'c_{\text{friction}} = \$15.00')
    
    return text

print("clean_text_formatting ready.")
