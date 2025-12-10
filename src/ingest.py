import fitz

def extract_text_from_pdf(pdfPath: str) -> str:
    doc = fitz.open(pdfPath)
    
    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

if __name__ == "__main__":
    path = "../data/sample.pdf"
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    print(f"How many chunks we received: {len(chunks)}")
    print("First chunk preview: \n", chunks[1])