# rag_streamlit_safe.py
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="📚 RAG Bot — By Kavi", layout="wide")
st.title("📚 RAG Bot — By Kavi")

# -----------------------------
# HuggingFace LLM Setup
# -----------------------------
# Replace "YOUR_HF_TOKEN" with your HuggingFace API token
HUGGINGFACEHUB_API_TOKEN = "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"

llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    model_kwargs={"temperature": 0, "max_new_tokens": 512},
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)

# -----------------------------
# Embedding model
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])
if uploaded_file is not None:
    # Load PDF and split into chunks
    loader = PyMuPDFLoader(uploaded_file)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Build FAISS vector store
    with st.spinner("Indexing PDF..."):
        vectorstore = FAISS.from_documents(docs, embedding_model)
    st.success("Indexing completed!")

    # -----------------------------
    # Create QA Chain
    # -----------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False  # only return the answer
    )

    # -----------------------------
    # Chat input
    # -----------------------------
    query = st.text_input("Ask a question about your PDF:")
    if query:
        with st.spinner("Generating answer..."):
            answer = qa.run(query)
        st.markdown("**Answer:**")
        st.write(answer)



