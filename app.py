import streamlit as st
from rag_chatbot.rag_pipeline import generate_answer, retrieve_docs

# Page config
st.set_page_config(page_title="Loan Approval RAG Chatbot", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
        }
        .reportview-container .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        .stTextInput > div > div > input {
            color: white !important;
        }
        .stMarkdown h3 {
            color: #1f4e79;
        }
        .stMarkdown h1 {
            color: #124076;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 Loan Approval RAG Q&A Chatbot")
st.markdown("Ask any question about the loan dataset, and get a smart answer!")

# Sidebar
with st.sidebar:
    st.header("📚 Instructions")
    st.markdown("""
    - This chatbot answers questions using a **Loan Approval dataset**.
    - It retrieves the most relevant document chunks using **FAISS** and sends the context to **OpenAI GPT-3.5** for answering.

    ### 💡 Example Questions
    - What is the average loan amount?
    - How many loans were approved?
    - Which employment type has the most loans?

    """)

# Input
query = st.text_input("🔎 Enter your question here:")

# Button to submit
if st.button("Submit Question"):
    if query:
        with st.spinner("🔍 Thinking..."):
            answer = generate_answer(query)
            matched_chunks = retrieve_docs(query)

        st.markdown("### ✅ Answer:")
        st.success(answer)

        st.markdown("### 📄 Matched Context Chunks:")
        for i, chunk in enumerate(matched_chunks, 1):
            st.markdown(f"**Chunk {i}:** {chunk}")
    else:
        st.warning("Please enter a question before submitting.")
