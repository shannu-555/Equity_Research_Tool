# app.py
import os
import io
import csv
import json
import re
import streamlit as st
from datetime import datetime
from typing import List
from dotenv import load_dotenv

# LangChain loaders / splitters / embeddings / vectorstore
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Groq SDK (primary LLM)
from groq import Groq

# Local fallback (transformers)
from transformers import pipeline

# Load env and config
load_dotenv()
import config  # expects GROQ_API_KEY and GROQ_MODEL in config.py (GROQ_API_KEY from .env)

# --- UI config ---
st.set_page_config(page_title="📈 Equity Research Tool", page_icon="📊", layout="wide")
st.title("📈 Equity Research Tool")
st.markdown("Process short news article links and ask concise, context-aware questions.")

# --- Session init for history ---
if "history" not in st.session_state:
    st.session_state["history"] = []  # newest first
if "vdb" not in st.session_state:
    st.session_state["vdb"] = None

# --- Utility: postprocess answer to remove polite prefaces and shorten ---
_PREFACE_RE = re.compile(
    r'^\s*(I(\'m| am) happy to help[,\.]?\s*|I don\'t have any specific information[,\.]?\s*|However[,\.]?\s*|It seems (like|that)\s*|If you\'d like.*$|If you want.*$)',
    flags=re.IGNORECASE
)

def postprocess_answer(text: str) -> str:
    if not text:
        return text
    # Remove common leading preface patterns
    t = _PREFACE_RE.sub('', text).strip()

    # Keep first 2 sentences max
    sentences = re.split(r'(?<=[.!?])\s+', t)
    if len(sentences) > 2:
        t = ' '.join(sentences[:2]).strip()

    # Clean whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    # If still empty, fallback to original trimmed
    if not t:
        t = text.strip()
    return t

# --- History helpers ---
def append_history(urls_text: str, question: str, answer: str):
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "urls": urls_text,
        "question": question,
        "answer": answer
    }
    st.session_state["history"].insert(0, record)

def history_to_csv_bytes():
    if not st.session_state["history"]:
        return None
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp", "urls", "question", "answer"])
    for r in st.session_state["history"]:
        writer.writerow([r["timestamp"], r["urls"], r["question"], r["answer"]])
    return output.getvalue().encode("utf-8")

def history_to_txt_bytes():
    if not st.session_state["history"]:
        return None
    output = io.StringIO()
    for i, r in enumerate(st.session_state["history"], 1):
        output.write(f"Record #{i}\n")
        output.write(f"Time: {r['timestamp']}\n")
        output.write(f"URLs:\n{r['urls']}\n")
        output.write(f"Question: {r['question']}\n")
        output.write(f"Answer:\n{r['answer']}\n")
        output.write("\n" + ("-"*40) + "\n\n")
    return output.getvalue().encode("utf-8")

# --- Prompt builder (concise and flexible) ---
def build_prompt_from_docs(question: str, docs: List[Document], max_chars: int = 900) -> str:
    if not docs:
        ctx = ""
    else:
        combined = "\n\n".join([d.page_content for d in docs[:3]])
        ctx = combined[:max_chars]
    prompt = (
        "You are a concise financial research assistant. Use the context below to answer the question. "
        "If the context contains the answer, respond in **one short sentence** (no preface). "
        "If not, provide a brief answer (1-2 sentences) using general knowledge. "
        "Do NOT use filler like 'I’m happy to help' or 'It seems like'. If unsure, say 'I am unsure' or 'Insufficient info'.\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return prompt

# --- Groq generation using SDK ---
def generate_with_groq(prompt: str, model: str, api_key: str, max_tokens: int = 180, temperature: float = 0.0) -> str:
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")
    client = Groq(api_key=api_key)
    messages = [
        {"role": "system", "content": "You are a concise, factual financial research assistant."},
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        stream=False
    )
    # Try object-style and dict-style responses
    try:
        raw = resp.choices[0].message.content
    except Exception:
        try:
            raw = resp["choices"][0]["message"]["content"]
        except Exception:
            raw = json.dumps(resp)
    return postprocess_answer(raw)

# --- Local fallback using GPT-2 (for development) ---
_local_gen = None
def generate_with_local(prompt: str, max_length: int = 160) -> str:
    global _local_gen
    if _local_gen is None:
        _local_gen = pipeline("text-generation", model="gpt2", device=-1)
    out = _local_gen(prompt, max_length=max_length, do_sample=False)
    text = out[0]["generated_text"]
    return postprocess_answer(text)

# --- Sidebar (settings + history) ---
st.sidebar.header("⚙️ Settings")
st.sidebar.write("Groq key found:", "✅" if config.GROQ_API_KEY else "❌")
temperature = st.sidebar.slider("Temperature (creativity)", 0.0, 1.0, 0.0, 0.1)
max_tokens = st.sidebar.slider("Max tokens (answer length)", 50, 400, 180, 10)
st.sidebar.markdown("---")

# History UI
st.sidebar.subheader("📚 History")
if st.sidebar.button("Clear history"):
    st.session_state["history"].clear()
    st.sidebar.success("History cleared")

if st.session_state["history"]:
    max_show = st.sidebar.number_input("Show recent", min_value=1, max_value=50, value=6, step=1)
    for idx, rec in enumerate(st.session_state["history"][:max_show]):
        with st.sidebar.expander(f"{rec['timestamp']} — Q: {rec['question'][:40]}...", expanded=False):
            st.markdown(f"**Question:** {rec['question']}")
            st.markdown(f"**Answer:** {rec['answer']}")
            st.markdown(f"**URLs:** {rec['urls'][:200]}")
            try:
                if st.button(f"Copy answer #{idx}", key=f"copy_{idx}"):
                    # Streamlit's clipboard_set might not be available in older versions
                    try:
                        st.clipboard_set(rec['answer'])
                        st.sidebar.success("Copied to clipboard")
                    except Exception:
                        st.sidebar.info("Copy not supported in this Streamlit version.")
            except Exception:
                pass

    csv_bytes = history_to_csv_bytes()
    txt_bytes = history_to_txt_bytes()
    c1, c2 = st.sidebar.columns(2)
    if csv_bytes:
        c1.download_button("Download CSV", data=csv_bytes, file_name="equity_history.csv", mime="text/csv")
    if txt_bytes:
        c2.download_button("Download TXT", data=txt_bytes, file_name="equity_history.txt", mime="text/plain")
else:
    st.sidebar.info("No history yet — ask a question to record it.")
st.sidebar.markdown("---")
st.sidebar.info("Tip: Use short news links to avoid large token use and quotas.")

# --- Main layout: two columns ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Load articles")
    urls_text = st.text_area("Enter news URLs (one per line)", height=180, placeholder="https://...")
    process_btn = st.button("🚀 Process Articles")

with col2:
    st.subheader("2) Ask a question")
    query = st.text_input("Question (short):")
    ask_btn = st.button("💬 Get Answer")

# --- Processing articles ---
if process_btn:
    if not urls_text.strip():
        st.error("Please add at least one URL.")
    else:
        with st.spinner("Loading and embedding (short test mode)..."):
            url_list = [u.strip() for u in urls_text.splitlines() if u.strip()]
            docs = []
            try:
                loader = UnstructuredURLLoader(urls=url_list)
                docs = loader.load()
            except Exception as e:
                st.error(f"Error loading URL(s): {e}")
                docs = []

            if docs:
                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                chunks = splitter.split_documents(docs)
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vectorstore = FAISS.from_documents(chunks, embeddings)
                st.session_state["vdb"] = vectorstore
                st.success(f"Processed {len(chunks)} chunks from {len(docs)} document(s).")
            else:
                st.warning("No documents loaded. Try a short news link.")

# --- Answering ---
if ask_btn:
    if not st.session_state.get("vdb"):
        st.error("Please process an article first.")
    elif not query.strip():
        st.error("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            retriever = st.session_state["vdb"].as_retriever(search_type="similarity", search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(query)
            prompt = build_prompt_from_docs(query, docs, max_chars=900)

            # Try Groq first (SDK)
            if config.GROQ_API_KEY and getattr(config, "GROQ_MODEL", None):
                try:
                    answer = generate_with_groq(prompt, model=config.GROQ_MODEL, api_key=config.GROQ_API_KEY, max_tokens=max_tokens, temperature=temperature)
                    st.success("Answer (Groq):")
                    st.write(answer)
                    append_history(urls_text, query, answer)
                except Exception as e:
                    st.error(f"Groq API error: {e}. Falling back to local model.")
                    try:
                        local_answer = generate_with_local(prompt, max_length=200)
                        st.success("Answer (Local fallback):")
                        st.write(local_answer)
                        append_history(urls_text, query, local_answer)
                    except Exception as lee:
                        st.error(f"Local fallback also failed: {lee}")
            else:
                st.info("No Groq key or model configured — using local small model for testing.")
                try:
                    local_answer = generate_with_local(prompt, max_length=200)
                    st.success("Answer (Local):")
                    st.write(local_answer)
                    append_history(urls_text, query, local_answer)
                except Exception as lee:
                    st.error(f"Local generation failed: {lee}")
