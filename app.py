import streamlit as st
import os
import tempfile
import re
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_openai import ChatOpenAI

# --- SETUP & CACHING ---
st.set_page_config(page_title="J&K Tourism RAG Agent", page_icon="🏔️", layout="wide")

st.markdown("""
<style>
    .phase-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #444; background-color: #262730; }
    .phase-title { color: #4CAF50; font-weight: bold; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_embedding_model():
    """Loads the embedding model once and caches it in memory for speed."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def bs4_extractor(html: str) -> str:
    """Extracts clean text from raw HTML using BeautifulSoup."""
    soup = BeautifulSoup(html, "html.parser")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()

def add_trace_log(phase, msg, data):
    """Helper to add logs and keep the array capped at 10 items to prevent UI lag."""
    st.session_state.trace_logs.append({"phase": phase, "msg": msg, "data": data})
    if len(st.session_state.trace_logs) > 10:
        st.session_state.trace_logs.pop(0)

# --- SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the J&K Tourism RAG Agent."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "trace_logs" not in st.session_state:
    st.session_state.trace_logs = []

# --- SIDEBAR & CONFIGURATION ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Enter OpenRouter Key", type="password")
    
    if api_key.startswith("sk-or-v1"):
        base_url = "https://openrouter.ai/api/v1"
        model_name = "inclusionai/ling-2.6-1t:free" 
        st.caption("✅ OpenRouter Key Detected: Using Nemotron 3 Nano Omni ")
    else:
        base_url = None
        model_name = "gpt-3.5-turbo"
        st.caption("ℹ️ Standard Key Detected: Using GPT-3.5")

    st.divider()

    st.subheader("📂 Knowledge Base (Phase 1)")
    
    # PDF Upload
    uploaded_files = st.file_uploader("Upload Tourism PDFs", accept_multiple_files=True, type="pdf")
    
    # URL Input & Crawler Settings
    website_url = st.text_input("Or paste a base URL to crawl (e.g., https://tourism.jk.gov.in)")
    crawl_depth = st.slider("Crawl Depth", min_value=1, max_value=3, value=1, help="1 = Main page + its links. Higher depths take longer to process.")
    
    if st.button("Process Documents & URLs"):
        if not api_key:
            st.error("Please enter an API Key first.")
        elif not uploaded_files and not website_url:
            st.warning("Please upload a PDF or enter a URL.")
        else:
            with st.spinner("Ingesting, Crawling, and Embedding... (This may take a while)"):
                documents = []
                
                # 1A. Process PDFs
                if uploaded_files:
                    for uploaded_file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_file_path = tmp_file.name
                        
                        try:
                            loader = PyPDFLoader(tmp_file_path)
                            documents.extend(loader.load())
                        finally:
                            os.remove(tmp_file_path) 
                            
                # 1B. Process Website URL (Deep Crawl)
                if website_url:
                    try:
                        st.toast(f"Starting deep crawl of {website_url} at depth {crawl_depth}...")
                        loader = RecursiveUrlLoader(
                            url=website_url,
                            max_depth=crawl_depth,
                            extractor=bs4_extractor,
                            prevent_outside=True # Prevents crawler from leaving the base domain
                        )
                        web_docs = loader.load()
                        documents.extend(web_docs)
                        st.toast(f"Successfully scraped {len(web_docs)} pages from the website!")
                    except Exception as e:
                        st.error(f"Failed to crawl URL: {e}")
                
                # 2. Split Text 
                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    chunks = text_splitter.split_documents(documents)
                    
                    # 3. Create Embeddings & Store in FAISS
                    embeddings = get_embedding_model()
                    
                    # Create vector store only if chunks were extracted
                    if chunks:
                        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                        st.success(f"Indexed {len(chunks)} chunks!")
                        
                        num_web_pages = len(web_docs) if website_url and 'web_docs' in locals() else 0
                        add_trace_log(
                            "Phase 1: Ingestion",
                            f"Processed {len(uploaded_files) if uploaded_files else 0} PDFs and crawled {num_web_pages} web pages. Created {len(chunks)} chunks.",
                            [d.metadata.get('source', 'Unknown') for d in documents[:3]]
                        )
                    else:
                        st.warning("No text could be extracted from the provided files/URLs.")
                else:
                    st.warning("No documents were loaded. Please check your files or URL.")

    st.divider()
    
    # Trace Visualizer
    st.subheader("🔍 RAG Pipeline Trace")
    if st.session_state.trace_logs:
        for log in reversed(st.session_state.trace_logs):
            st.markdown(f"<div class='phase-box'><div class='phase-title'>{log['phase']}</div>{log['msg']}</div>", unsafe_allow_html=True)

# --- MAIN CHAT LOGIC ---
st.title("🏔️ J&K Tourism AI Assistant")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Ask about Gulmarg, Safety, etc..."):
    if not api_key:
        st.error("API Key required.")
        st.stop()
    if not st.session_state.vector_store:
        st.error("Please upload documents or scrape a URL first.")
        st.stop()

    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # RAG Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Phase 2: Retrieval
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(prompt)
        
        add_trace_log(
            "Phase 2: Retrieval",
            f"Retrieved {len(relevant_docs)} chunks based on similarity.",
            [d.page_content[:100] for d in relevant_docs]
        )

        # Phase 3: Generation
        context = "\n\n".join([d.page_content for d in relevant_docs])
        
        # Enhanced System Prompt to enforce strict RAG behavior
        system_prompt = """You are an expert J&K Tourism Assistant. 
Use the provided context to answer the user's question. 
If the answer is not contained in the context, say "I don't have enough information about that in my current documents, but I recommend checking the official J&K Tourism website." 
Do NOT make up information."""
        
        # Initialize LLM
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=0
        )
        
        try:
            response = llm.invoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt}"}
            ])
            full_response = response.content

            if full_response:
                full_response = full_response.replace("<s>", "").replace("[/out]", "").replace("[out]", "").replace("[INST]", "").strip()
            
            # SAFETY CHECK
            if not full_response:
                full_response = "⚠️ The model returned an empty response. Please try asking again."

            message_placeholder.markdown(full_response)
            
            add_trace_log(
                "Phase 3: Generation",
                f"Generated answer using {model_name}",
                full_response[:100] + "..."
            )
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"API Error: {e}")
