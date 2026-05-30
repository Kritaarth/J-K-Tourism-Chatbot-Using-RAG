import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_openai import ChatOpenAI


st.set_page_config(page_title="J&K Tourism RAG Agent", page_icon="🏔️", layout="wide")


st.markdown("""
<style>
    .phase-box { padding: 10px; border-radius: 5px; margin-bottom: 10px; border: 1px solid #444; background-color: #262730; }
    .phase-title { color: #4CAF50; font-weight: bold; font-family: monospace; }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am the J&K Tourism RAG Agent."}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "trace_logs" not in st.session_state:
    st.session_state.trace_logs = []


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
    uploaded_files = st.file_uploader("Upload Tourism PDFs", accept_multiple_files=True, type="pdf")
    
    if uploaded_files and st.button("Process Documents"):
        if not api_key:
            st.error("Please enter an API Key first.")
        else:
            with st.spinner("Ingesting and Embedding... (Using FAISS)"):
                documents = []
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file_path = tmp_file.name
                    loader = PyPDFLoader(tmp_file_path)
                    documents.extend(loader.load())
                
                # Split Text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(documents)
                
                # Create Embeddings using HuggingFace (Local CPU)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                
                # Store in FAISS
                st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
                
                st.success(f"Indexed {len(chunks)} chunks!")
                st.session_state.trace_logs.append({
                    "phase": "Phase 1: Ingestion",
                    "msg": f"Processed {len(uploaded_files)} files. Created {len(chunks)} vector chunks using FAISS.",
                    "data": [d.metadata.get('source', 'Unknown') for d in documents[:3]]
                })

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
        st.error("Please upload documents first.")
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
        
        st.session_state.trace_logs.append({
            "phase": "Phase 2: Retrieval",
            "msg": f"Retrieved {len(relevant_docs)} chunks based on similarity.",
            "data": [d.page_content[:100] for d in relevant_docs]
        })

        # Phase 3: Generation
        context = "\n\n".join([d.page_content for d in relevant_docs])
        system_prompt = "You are a helpful J&K Tourism Assistant. Answer based ONLY on the context provided."
        
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
            
            st.session_state.trace_logs.append({
                "phase": "Phase 3: Generation",
                "msg": f"Generated answer using {model_name}",
                "data": full_response[:100] + "..."
            })
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"API Error: {e}")