import streamlit as st
from rag_chatbot.rag_pipeline import generate_answer, get_context_chunks

# Streamlit Page Config
st.set_page_config(page_title="Loan Approval Q&A", page_icon="📊", layout="wide")

# Custom CSS for better visibility
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stTextInput input {
        padding: 0.75rem;
        border-radius: 8px;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .big-font {
        font-size: 26px !important;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar instructions
st.sidebar.title("📘 Instructions")
st.sidebar.markdown("""
- This is a Q&A chatbot for the **Loan Approval Dataset**.
- Ask natural language questions like:
    - ✅ *What factors affect loan approval?*
    - ✅ *How many applicants were denied?*
    - ✅ *What's the average income of approved applicants?*
- It uses **RAG (Retrieval-Augmented Generation)** to pull answers from data chunks.

🧠 Powered by Sentence Transformers + FAISS + OpenAI GPT.
""")

# Main title
st.markdown("<p class='big-font'>📊 Loan Approval RAG Chatbot</p>", unsafe_allow_html=True)

# User input
query = st.text_input("🔎 Ask your question:")

if st.button("Get Answer"):
    if query.strip():
        with st.spinner("🧠 Thinking..."):
            answer = generate_answer(query)
            context_chunks = get_context_chunks(query)  # Optional: show matched chunks

        st.success("✅ Answer:")
        st.markdown(f"**{answer}**")

        # Show context
        with st.expander("📄 Show matched context chunks"):
            for idx, chunk in enumerate(context_chunks):
                st.markdown(f"**Chunk {idx+1}:**\n```\n{chunk}\n```")
    else:
        st.warning("Please enter a question.")
