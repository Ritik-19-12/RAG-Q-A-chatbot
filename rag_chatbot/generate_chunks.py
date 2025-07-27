import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os

# Load CSV
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, 'data', 'Training_Dataset.csv')
df = pd.read_csv(file_path)
chunks = []

# Chunk by rows (ex: 1 row = 1 document)
for i, row in df.iterrows():
    text = " ".join([f"{col}: {row[col]}" for col in df.columns])
    chunks.append(text)

# Embed using Sentence Transformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Save index using FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, "rag_chatbot/loan_index.faiss")

# Save chunk mapping
with open("rag_chatbot/chunk_texts.txt", "w", encoding="utf-8") as f:
    for chunk in chunks:
        f.write(chunk + "\n")

print("✅ Chunks generated and indexed.")
