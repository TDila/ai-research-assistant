from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return model.encode(text).tolist()

if __name__ == "__main__":
    sample = "Artificial intelligence is transforming research."
    vector = get_embedding(sample)
    print("Embedding length:", len(vector))
    print("First 10 numbers:", vector[:10])