from fastapi import FastAPI
from src.ingest import extract_text_from_pdf, chunk_text
from src.embeddings import get_embedding
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline

app = FastAPI(title="AI Research Assistant")

vector_store = None
rag_pipeline = None

@app.post("/ingest")
def ingest_pdf(pdf_path: str):
    global vector_store, rag_pipeline

    text = extract_text_from_pdf(pdf_path)

    chunks = chunk_text(text)

    embeddings = [get_embedding(c) for c in chunks]

    vector_store = VectorStore(embedding_dim=len(embeddings[0]))
    vector_store.add_embeddings(embeddings, chunks)

    rag_pipeline = RAGPipeline(vector_store)

    return {"message": "PDF ingested successfully", "chunks": len(chunks)}

@app.post("/query")
def query_document(question: str):
    if rag_pipeline is None:
        return {"error": "No document ingested yet"}
    
    answer = rag_pipeline.ask(question)
    return {"question":question, "answer":answer}