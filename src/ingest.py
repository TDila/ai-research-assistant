import fitz

def extract_text_from_pdf(pdfPath: str) -> str:
    doc = fitz.open(pdfPath)

    text = ""
    for page in doc:
        text += page.get_text()
    
    doc.close()
    return text

if __name__ in "__main__":
    path = "../data/sample.pdf"
    print(extract_text_from_pdf(path))