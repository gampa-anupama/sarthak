#!/usr/bin/env python3
import faiss
import json
from sentence_transformers import SentenceTransformer
import ollama

# ----------------------
# Load FAISS + Metadata
# ----------------------
index = faiss.read_index("output/faiss_index.idx")

with open("output/chunks_metadata.json", "r", encoding="utf-8") as f:
    chunks_metadata = json.load(f)

model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Search in FAISS
# ----------------------
def search(query, top_k=3):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks_metadata):
            results.append({
                "text": chunks_metadata[idx]["text"],
                "source": chunks_metadata[idx]["source"],
                "type": chunks_metadata[idx]["type"],
                "score": float(distances[0][i])
            })
    return results

# ----------------------
# Ask Ollama with Context
# ----------------------
def ask_ollama(query, context_chunks, model_name="llama3"):
    context_text = "\n\n".join([c["text"] for c in context_chunks])
    prompt = f"""
You are a helpful assistant. 
Here is some context from documents:

{context_text}

Now, answer this question: {query}
"""
    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# ----------------------
# Main
# ----------------------
if __name__ == "__main__":
    while True:
        query = input("\n❓ Enter your question (or 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            break

        print("\n🔍 Searching FAISS index...")
        results = search(query, top_k=5)

        print("\n📄 Context Retrieved:")
        for r in results:
            print(f"- Source: {r['source']} | Type: {r['type']} | Score: {r['score']:.2f}")

        print("\n🤖 Asking Ollama...")
        answer = ask_ollama(query, results, model_name="llama3")  # you can change model to mistral, codellama, etc.
        print("\n✅ Answer:\n", answer)
