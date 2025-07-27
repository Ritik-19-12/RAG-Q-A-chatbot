import faiss
import pickle
import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# ✅ Load environment variables (if needed)
load_dotenv()

# ✅ Paths
base_path = os.path.dirname(__file__)
faiss_index_path = os.path.join(base_path, "loan_index.faiss")
chunks_path = os.path.join(base_path, "rag_chunks.pkl")

# ✅ Load FAISS index and chunks
index = faiss.read_index(faiss_index_path)
with open(chunks_path, "rb") as f:
    chunk_texts = pickle.load(f)

# ✅ Load lightweight embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Load lightweight Hugging Face model (FLAN-T5-small)
try:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    llm_pipeline = None

# ✅ Embedding function
def get_embedding(text):
    try:
        embedding = embedder.encode([text])[0]
        return np.array(embedding, dtype=np.float32)
    except Exception as e:
        print(f"❌ Embedding Error: {e}")
        return None

# ✅ Retrieve top-k chunks
def retrieve_docs(query, k=3):
    embedding = get_embedding(query)
    if embedding is None:
        return [], "❌ Could not compute embedding."
    distances, indices = index.search(np.array([embedding]), k)
    return [chunk_texts[i] for i in indices[0]], None

# ✅ Generate Answer
def generate_answer(query, k=3):
    chunks, error = retrieve_docs(query, k)
    if error:
        return error

    context = "\n".join(chunks)
    prompt = f"Answer the question based on the context.\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

    if llm_pipeline:
        try:
            response = llm_pipeline(prompt, max_new_tokens=100)[0]['generated_text']
            return response.strip()
        except Exception as e:
            return f"❌ Generation Error: {e}"
    else:
        return f"🤖 Context:\n{context}\n\n📌 No model loaded. Please install or connect an LLM."

# ✅ Example use
if __name__ == "__main__":
    query = "What is the eligibility for the gold loan?"
    print(generate_answer(query))
