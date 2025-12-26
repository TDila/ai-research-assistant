from fastapi import FastAPI
from src.ingest import extract_text_from_pdf, chunk_text
from src.embeddings import get_embedding
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from pydantic import BaseModel

app = FastAPI(title="AI Research Assistant")

vector_store = None
rag_pipeline = None

class IngestRequest(BaseModel):
    pdf_path: str

class QueryRequest(BaseModel):
    question: str

class SummaryRequest(BaseModel):
    max_chunks: int = 5

class LiteratureReviewRequest(BaseModel):
    max_chunks: int = 6

@app.post("/ingest")
def ingest_pdf(request: IngestRequest):
    global vector_store, rag_pipeline

    text = extract_text_from_pdf(request.pdf_path)
    chunks = chunk_text(text)

    embeddings = [get_embedding(c) for c in chunks]

    vector_store = VectorStore(embedding_dim=len(embeddings[0]))
    vector_store.add_embeddings(embeddings, chunks)

    rag_pipeline = RAGPipeline(vector_store)

    return {"message": "PDF ingested successfully", "chunks": len(chunks)}

@app.post("/query")
def query_document(request: QueryRequest):
    if rag_pipeline is None:
        return {"error": "No document ingested yet"}
    
    answer = rag_pipeline.ask(request.question)
    return {"question":request.question, "answer":answer}

@app.post("/summary")
def summarize_document(request: SummaryRequest):
    if rag_pipeline is None:
        return {"error": "No document ingested yet"}
    
    query_embedding = get_embedding("Summarize the document")
    top_chunks = vector_store.search(
        query_embedding,
        top_k=request.max_chunks
    )

    context = "\n\n".join(top_chunks)

    prompt = f"""
Provide a concise academic-style summary of the following document.

Context:
{context}

Summary:
"""
    answer = rag_pipeline.ask(prompt)
    return {"summary":answer}

@app.post("/literature-review")
def literature_review(request: LiteratureReviewRequest):
    if rag_pipeline is None:
        return {"error": "No document ingested yet"}
    
    query_embedding = get_embedding("Generated a structured literature review")
    chunks = vector_store.search(
        query_embedding,
        top_k=request.max_chunks
    )

    context = "\n\n".join(chunks)

    prompt = f"""
You are an academic research assistant.

Based ONLY on the context below, generate a structured literature review using the following format:

Problem:
- ...

Methodology:
- ...

Dataset:
- ...

Results:
- ...

Limitations:
- ...

Context:
{context}
"""
    answer = rag_pipeline.ask(prompt)
    return {"literature_review": answer}