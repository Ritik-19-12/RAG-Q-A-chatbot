# 🤖 RAG Q&A Chatbot (Local + Lightweight Models)

This is a **local Retrieval-Augmented Generation (RAG)** chatbot that:
- Uses **FAISS** for fast document similarity search
- Embeds text using `all-MiniLM-L6-v2` from `sentence-transformers`
- Answers questions using Hugging Face’s **`falcon-rw-1b`** or any other lightweight language model
- Runs **completely offline** (no API keys required)

---

## 📁 Project Structure

RAG-Q-A-chatbot/

├── eda/
│ └── load_eda.ipynb # Eda analysis
├── rag_chatbot/
│ ├── generate_chunks.py # Script to chunk and embed text, and create FAISS index
│ ├── rag_main.py # Main script to ask questions and get answers
│ ├── loan_index.faiss # Saved FAISS index of vector embeddings
│ ├── rag_chunks.pkl # Saved text chunks (used to map FAISS results)
│
├── data/
│ └── Training Dataset.csv # Raw text data or document file
│
├── requirements.txt # Python libraries used
└── README.md # Project documentation

---

## ⚙️ Setup Instructions

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/Ritik-19-12/RAG-Q-A-chatbot.git
cd RAG-Q-A-chatbot/rag_chatbot
# For Windows
python -m venv venv
venv\Scripts\activate

pip install -r ../requirements.txt

# Make sure you have the file: data/Training Dataset.csv

python .\rag_chatbot\generate_chunks.py

python .\rag_chatbot\rag_pipeline.py 

streamlit run app.py
```
---

## 📬 Author

- **Ritik Sotwal**
- 4th Year, Electronics and Computer Engineering, MBM University

---