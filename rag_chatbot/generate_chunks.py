import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle

# Define paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, 'data', 'Training_Dataset.csv')
output_dir = os.path.join(base_dir, 'rag_chatbot')
index_path = os.path.join(output_dir, 'loan_index.faiss')
chunks_path = os.path.join(output_dir, 'rag_chunks.pkl')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_path)
chunks = []

# Convert each row to a string chunk
for _, row in df.iterrows():
    text = " ".join([f"{col}: {row[col]}" for col in df.columns])
    chunks.append(text)

# Embed chunks using SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Create and save FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
faiss.write_index(index, index_path)

# Save chunks using Pickle
with open(chunks_path, "wb") as f:
    pickle.dump(chunks, f)

print("✅ Embeddings and chunks saved successfully.")
