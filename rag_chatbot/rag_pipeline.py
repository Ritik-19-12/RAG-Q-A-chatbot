import faiss
import pickle
import os
import numpy as np
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# ✅ Load API key from environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Load chunked text (Pickle format)
chunks_path = os.path.join(os.path.dirname(__file__), "rag_chunks.pkl")
with open(chunks_path, "rb") as f:
    chunk_texts = pickle.load(f)

# ✅ Load FAISS index
index_path = os.path.join(os.path.dirname(__file__), "loan_index.faiss")
index = faiss.read_index(index_path)


def get_embedding(text):
    """Generate embedding for the input text using OpenAI"""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return np.array(response.data[0].embedding, dtype="float32")
    except OpenAIError as e:
        return None


def get_answer(query, k=3):
    """Retrieve top-k chunks and generate answer using OpenAI Chat API"""
    query_embedding = get_embedding(query)
    if query_embedding is None:
        return "❌ Failed to generate embedding. Check your API key or internet connection."

    # Search similar vectors
    _, indices = index.search(np.array([query_embedding]), k)

    # Retrieve top-k chunks
    context = "\n\n".join([chunk_texts[i] for i in indices[0]])

    # Prompt for Chat Completion
    prompt = f"""You are a helpful assistant. Use the following information to answer the question.

Context:
{context}

Question: {query}
Answer:"""

    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return completion.choices[0].message.content.strip()
    except OpenAIError as e:
        return f"❌ Error while getting answer: {str(e)}"
