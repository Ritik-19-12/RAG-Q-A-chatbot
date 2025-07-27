import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load embedding model
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError("❌ Failed to load SentenceTransformer model: all-MiniLM-L6-v2") from e

# Load FAISS index
INDEX_PATH = "rag_chatbot/loan_index.faiss"
try:
    index = faiss.read_index(INDEX_PATH)
except Exception as e:
    raise FileNotFoundError(f"❌ FAISS index not found at: {INDEX_PATH}") from e

# Load chunked documents
CHUNKS_PATH = "rag_chatbot/chunk_texts.txt"
try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as file:
        chunks = [line.strip() for line in file.readlines()]
except Exception as e:
    raise FileNotFoundError(f"❌ Could not load chunked texts from: {CHUNKS_PATH}") from e


def retrieve_docs(query: str, top_k: int = 3) -> list[str]:
    """
    Encode the query and retrieve top_k most similar chunks from FAISS index.

    Args:
        query (str): User query.
        top_k (int): Number of top documents to retrieve.

    Returns:
        List[str]: Top matching document chunks.
    """
    try:
        query_vector = model.encode([query])
        _, indices = index.search(np.array(query_vector), top_k)
        return [chunks[i] for i in indices[0]]
    except Exception as e:
        print(f"❌ Error retrieving documents: {e}")
        return ["Error retrieving documents."]


def generate_answer(query: str) -> str:
    """
    Generate an answer from OpenAI using context from retrieved chunks.

    Args:
        query (str): User query.

    Returns:
        str: Generated answer from OpenAI.
    """
    context_chunks = retrieve_docs(query)
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an AI assistant helping with loan dataset analysis.
Use the following context to answer the question precisely and helpfully.
Do not guess if the answer is not in the context.

Context:
{context}

Question: {query}
Answer:
""".strip()

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ OpenAI API error: {str(e)}"
