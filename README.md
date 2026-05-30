# 🏔️ J&K Tourism RAG AI Assistant

A clean, production-ready Retrieval-Augmented Generation (RAG) chatbot application designed to serve verified, context-grounded travel guidelines and informational lookups for Jammu & Kashmir. Built with **Streamlit**, **LangChain**, and **FAISS**, this tool reads uploaded reference material directly and answers tourist queries with zero hallucinations.

---

## ✨ Features

* **Strict Context Boundary:** Fully restricted system instructions force the AI model to reply *only* using your uploaded documents. If information isn't present, it safe-fails to an official suggestion.
* **Streamlined Document Ingestion:**
  * Support for simultaneous multiple PDF uploads via standard file drop.
  * Optimized text-splitting structure using `RecursiveCharacterTextSplitter` configured for high semantic context retention ($1000$ character size with a $200$ character overlap).
* **High-Performance In-Memory Store:**
  * Uses an optimized vector space powered by **FAISS** (Facebook AI Similarity Search).
  * Fast embeddings map locally using a cached Hugging Face transformer pipeline (`sentence-transformers/all-MiniLM-L6-v2`).
* **Live RAG Process Trace:** An integrated logging component displays the extraction journey right in the sidebar, tracing steps instantly from **Ingestion** $\rightarrow$ **Retrieval** $\rightarrow$ **Generation**.
* **Deployment Shielding:** Built-in programmatic `try...finally` teardowns delete local temporary working files automatically upon block processing, eliminating cloud infrastructure memory and file leaks.

---

## 🛠️ Tech Stack

* **Frontend Framework:** Streamlit
* **RAG Orchestration:** LangChain
* **Vector Storage Engine:** FAISS (In-Memory CPU variant)
* **Embedding Model Strategy:** `sentence-transformers/all-MiniLM-L6-v2`
* **Inference Platform:** OpenRouter API / OpenAI API

---
