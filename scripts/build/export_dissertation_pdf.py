import os
import sys
import time

docx_path = os.path.abspath("Dissertation_Ieman_Ihsan_25005466_UTP.docx")
pdf_path = os.path.abspath("Dissertation_Ieman_Ihsan_25005466_UTP.pdf")

print(f"Exporting DOCX to PDF...")
print(f"  Input:  {docx_path}")
print(f"  Output: {pdf_path}")

try:
    import win32com.client
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    
    doc = word.Documents.Open(docx_path)
    # Update all fields in document (including TOC/LOT/LOF if applicable)
    try:
        doc.Fields.Update()
    except Exception as e:
        print(f"Fields update note: {e}")
        
    doc.SaveAs(pdf_path, FileFormat=17) # 17 = wdFormatPDF
    doc.Close(False)
    word.Quit()
    
    time.sleep(1)
    if os.path.exists(pdf_path):
        size_mb = os.path.getsize(pdf_path) / (1024 * 1024)
        print(f"SUCCESS: Exported PDF ({size_mb:.2f} MB)")
    else:
        print("ERROR: PDF file was not created!")
        sys.exit(1)

except Exception as ex:
    print(f"Word COM export failed: {ex}")
    sys.exit(1)
