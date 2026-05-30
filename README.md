# 🏔️ J&K Tourism RAG AI Assistant

A production-ready Retrieval-Augmented Generation (RAG) chatbot designed to provide accurate, context-grounded information about Jammu & Kashmir tourism. Built with **Streamlit**, **LangChain**, and **FAISS**, this assistant eliminates hallucinations by answering questions strictly using documents uploaded or live websites scraped by the administrator.

---

## ✨ Features

* **Multi-Source Data Ingestion (Phase 1):**
  * **PDF Ingestion:** Drop multiple tourism PDFs into the sidebar for chunking and indexing.
  * **Pro Web Crawler:** Paste any base URL (e.g., official government portals) to dynamically deep-crawl pages using `RecursiveUrlLoader` and `BeautifulSoup`.
* **Sub-Domain Isolation:** The built-in crawler restricts itself to the base domain, ensuring it doesn't leak into irrelevant external sites.
* **Vector Storage:** Utilizing **FAISS** (Facebook AI Similarity Search) and cached `sentence-transformers` for super-fast in-memory document matching.
* **RAG Pipeline Trace:** A real-time debugging visualizer in the sidebar that updates on every chat interaction, demonstrating how the data goes from **Ingestion** $\rightarrow$ **Retrieval** $\rightarrow$ **Generation**.
* **Memory & Storage Guardrails:** Built-in automatic deletion of temporary files to eliminate server memory leaks and a capped trace log array to prevent UI lag.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **RAG Orchestration:** LangChain
* **Vector Database:** FAISS (In-memory)
* **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (via Hugging Face)
* **LLM Engine:** OpenRouter API (Defaults to Nemotron Nano Omni for free tiers, fallback to GPT models)

---

## 🚀 Local Installation & Setup

Follow these steps to run the RAG agent locally on your computer:

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
