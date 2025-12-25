import faiss
import numpy as np

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.text_chunks = []

    def add_embeddings(self, embeddings: list, chunks: list):
        vectors = np.array(embeddings).astype("float32")
        self.index.add(vectors)
        self.text_chunks.extend(chunks)
    
    def search(self, query_embedding: list, top_k: int = 5):
        query_vector = np.array([query_embedding]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for idx in indices[0]:
            results.append(self.text_chunks[idx])

        return results

if __name__ == "__main__":
    from src.embeddings import get_embedding

    chunks = [
        "Machine learning is a subset of AI.",
        "Quantum physics studies particles.",
        "Deep learning uses neural networks."
    ]

    embeddings = [get_embedding(c) for c in chunks]

    store = VectorStore(embedding_dim=len(embeddings[0]))
    store.add_embeddings(embeddings, chunks)

    query = "What is neural network?"
    query_embedding = get_embedding(query)

    results = store.search(query_embedding, top_k=2)

    print("Search results:")
    for r in results:
        print("-", r)