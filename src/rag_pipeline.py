from src.embeddings import get_embedding
from src.vector_store import VectorStore
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

class RAGPipeline:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
    
    def ask(self, question: str, top_k: int = 5) -> str:
        question_embedding = get_embedding(question)

        relevant_chunks = self.vector_store.search(
            question_embedding, top_k=top_k
        )

        context = "\n\n".join(relevant_chunks)

        prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content":prompt}]
        )

        return response.choices[0].message.content
    
if __name__ == "__main__":
    from src.ingest import extract_text_from_pdf, chunk_text

    text = extract_text_from_pdf("../data/sample.pdf")
    chunks = chunk_text(text)

    embeddings = [get_embedding(c) for c in chunks]

    store = VectorStore(embedding_dim=len(embeddings[0]))
    store.add_embeddings(embeddings, chunks)

    rag = RAGPipeline(store)
    answer = rag.ask("What is the main contribution of this document?")

    print("AI Answer:\n", answer)