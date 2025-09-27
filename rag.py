# rag_streamlit_safe.py

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import fitz  # PyMuPDF
from ollama import chat  # Ollama Python client
import numpy as np

# ---------------- Config ----------------
INDEX_PATH = "faiss.index"
DOCS_PATH = "docs.pkl"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_LLAMA_MODEL = "llama3.2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
BATCH_SIZE = 16  # batch embedding to reduce CPU load
# ----------------------------------------

@st.cache_resource
def load_embedding_model(name=EMBEDDING_MODEL_NAME):
    return SentenceTransformer(name)

# ---------------- PDF/Text Extraction ----------------
def extract_text_from_pdf_bytes(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = [page.get_text() for page in doc]
    return "\n".join(text)

# ---------------- Safe Chunking ----------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Safe increment
        if end == length:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------- FAISS Index ----------------
def build_faiss_index(embeddings_np):
    d = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings_np)
    index.add(embeddings_np)
    return index

def save_index_and_docs(index, docs):
    faiss.write_index(index, INDEX_PATH)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump(docs, f)

def load_index_and_docs():
    if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(DOCS_PATH, "rb") as f:
            docs = pickle.load(f)
        return index, docs
    return None, None

# ---------------- Index Uploaded Files ----------------
def index_uploaded_files(uploaded_files, embed_model, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    all_texts = []
    metadata = []
    for up in uploaded_files:
        name = up.name
        b = up.read()
        if name.lower().endswith(".pdf"):
            text = extract_text_from_pdf_bytes(b)
        else:
            try:
                text = b.decode("utf-8")
            except:
                text = str(b)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            all_texts.append(c)
            metadata.append({"source": name, "chunk_id": i})

    if not all_texts:
        return None, None

    # ---------------- Batch embedding ----------------
    embeddings_list = []
    for i in range(0, len(all_texts), BATCH_SIZE):
        batch = all_texts[i:i+BATCH_SIZE]
        batch_emb = embed_model.encode(batch, convert_to_numpy=True, show_progress_bar=True)
        embeddings_list.append(batch_emb)
    embeddings_np = np.vstack(embeddings_list).astype("float32")

    index = build_faiss_index(embeddings_np)
    docs = [{"text": t, "meta": m} for t, m in zip(all_texts, metadata)]
    save_index_and_docs(index, docs)
    return index, docs

# ---------------- Retriever ----------------
def retrieve(query, index, docs, embed_model, top_k=TOP_K):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, top_k)
    results = []
    for idx in I[0]:
        if idx < 0 or idx >= len(docs):
            continue
        results.append(docs[idx])
    return results

# ---------------- Ollama Chat ----------------
def ask_ollama_with_context(model_name, question, context_chunks):
    system_prompt = (
        "You are a helpful research assistant. Use ONLY the provided context to answer the question. "
        "If the answer is not in the context, say you don't know. "
        "Provide concise answers and list source filenames if possible."
    )
    context_text = "\n\n".join(
        [f"--- Source: {c['meta'].get('source','unknown')} (chunk {c['meta'].get('chunk_id',0)}) ---\n{c['text']}" for c in context_chunks]
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
    response = chat(model=model_name, messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ])
    return response.get("message", {}).get("content", "").strip()

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="RAG Bot — By Kavi", layout="wide")
st.title("📚 RAG Bot — By Kavi")

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("Ollama model name", value=DEFAULT_LLAMA_MODEL)
    chunk_size = st.number_input("Chunk size (chars)", value=CHUNK_SIZE, step=100)
    chunk_overlap = st.number_input("Chunk overlap (chars)", value=CHUNK_OVERLAP, step=50)
    top_k = st.number_input("Retriever top_k", value=TOP_K, min_value=1, max_value=10)
    if st.button("Clear saved index"):
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists(DOCS_PATH):
            os.remove(DOCS_PATH)
        st.success("Cleared saved index & docs.")

# Upload documents
st.subheader("1) Upload documents (PDF / TXT / MD)")
uploaded = st.file_uploader("Upload one or more files", accept_multiple_files=True, type=["pdf","txt","md"])
if uploaded:
    st.info(f"{len(uploaded)} file(s) ready to index.")
    if st.button("Index uploaded files"):
        with st.spinner("Indexing — extracting, chunking, embedding, building FAISS..."):
            emb_model = load_embedding_model()
            idx, docs = index_uploaded_files(uploaded, emb_model, chunk_size=chunk_size, overlap=chunk_overlap)
            if idx is None:
                st.error("No text extracted from uploaded files.")
            else:
                st.success(f"Index built with {len(docs)} chunks. Saved to disk.")
else:
    st.info("No files uploaded. You can also use an existing saved index if present on disk.")

# Load existing index
index, docs = load_index_and_docs()
if index is not None and docs is not None:
    st.success(f"Loaded saved index with {len(docs)} chunks.")
else:
    st.warning("No saved index found. Upload & index documents to create one.")

# Ask questions
st.subheader("2) Ask questions")
question = st.text_input("Enter your question about the uploaded documents")
if st.button("Ask") and question.strip():
    if index is None or docs is None:
        st.error("No index available. Please upload and index documents first.")
    else:
        emb_model = load_embedding_model()
        with st.spinner("Retrieving relevant chunks..."):
            ctx = retrieve(question, index, docs, emb_model, top_k=top_k)
        if not ctx:
            st.warning("No relevant chunks found.")
        else:
            with st.spinner("Generating answer with Ollama..."):
                try:
                    answer = ask_ollama_with_context(model_name, question, ctx)
                except Exception as e:
                    st.error(f"Ollama call failed: {e}")
                    answer = ""
            st.markdown("**Answer:**")
            st.write(answer)
            #st.markdown("**Sources / retrieved chunks:**")
            #for i, c in enumerate(ctx):
                #src = c.get("meta", {}).get("source", "unknown")
                #st.write(f"- Source: **{src}** (chunk {c.get('meta',{}).get('chunk_id',0)}) — {c['text'][:400].replace('\\n',' ')}{'...' if len(c['text'])>400 else ''}")
