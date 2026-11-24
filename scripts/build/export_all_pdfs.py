# -*- coding: utf-8 -*-
"""
export_all_pdfs.py
Deterministic Word COM export for both UTP Dissertation and IEEE Technical Paper.
"""

import os
import sys
import time

def export_file(word, docx_filename, pdf_filename):
    docx_path = os.path.abspath(docx_filename)
    pdf_path = os.path.abspath(pdf_filename)
    
    print(f"Exporting: {docx_filename} -> {pdf_filename}...")
    if not os.path.exists(docx_path):
        raise FileNotFoundError(f"Missing docx: {docx_path}")
        
    doc = word.Documents.Open(docx_path)
    try:
        doc.Fields.Update()
    except Exception as e:
        print(f"  Note on fields update: {e}")
        
    doc.SaveAs(pdf_path, FileFormat=17) # 17 = wdFormatPDF
    doc.Close(False)
    
    time.sleep(1)
    if os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"  SUCCESS: Exported {pdf_filename} ({size_mb:.2f} MB)")
    else:
        raise RuntimeError(f"Failed to generate {pdf_path}")

def main():
    try:
        import win32com.client
    except ImportError:
        print("ERROR: pywin32 / win32com is required for native Word COM export.")
        sys.exit(1)
        
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    
    try:
        export_file(word, "Dissertation_Ieman_Ihsan_25005466_UTP.docx", "Dissertation_Ieman_Ihsan_25005466_UTP.pdf")
        export_file(word, "IEEE_Technical_Paper_Ieman_Ihsan_25005466.docx", "IEEE_Technical_Paper_Ieman_Ihsan_25005466.pdf")
    finally:
        word.Quit()
        
    print("\nALL PDF EXPORTS COMPLETED DETERMINISTICALLY!")

if __name__ == '__main__':
    main()
