import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import openai
import os

# Load environment variable
from dotenv import load_dotenv
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load FAISS index and model
index = faiss.read_index("rag_chatbot/loan_index.faiss")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load text chunks
with open("rag_chatbot/chunk_texts.txt", "r", encoding="utf-8") as f:
    chunks = f.readlines()

def retrieve_docs(query, top_k=3):
    query_vec = model.encode([query])
    _, indices = index.search(np.array(query_vec), top_k)
    results = [chunks[i].strip() for i in indices[0]]
    return results

def generate_answer(query):
    docs = retrieve_docs(query)
    context = "\n".join(docs)
    
    prompt = f"Answer the question based on the data below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or any LLM you're using
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response['choices'][0]['message']['content'].strip()
