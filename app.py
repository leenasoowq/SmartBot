import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Additional import for reading Word docs
import docx2txt
import nbformat

# Local imports
from services.language_service import LanguageService
from services.document_service import DocumentService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Instantiate Services
lang_service = LanguageService(client)
doc_service = DocumentService(client)
embedding_model = doc_service.embedding_model

# Initialize session state
for key in ["processed_files", "conversation_history", "quiz_mode", "selected_file", "file_categories"]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key in ["processed_files", "conversation_history"] 
            else False if key == "quiz_mode"
            else {} if key == "file_categories"
            else None
        )

def preprocess_files(files):
    """
    Upload & parse PDF, DOC, DOCX, TXT, PY, and IPYNB files ONLY.
    Extract text, then embed into the vectorstore with 'file_name' metadata.
    Audio/Video are handled elsewhere (transcribe_translate.py).
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Filter only new files
    new_files = [f.name for f in files if f.name not in st.session_state["processed_files"]]
    if not new_files:
        return "No new files to process."

    all_docs = []
    for file in files:
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as out_file:
            out_file.write(file.getbuffer())

        ext = file.name.lower()
        category = "chatbot"  # Default to chatbot-only files
        
        # 1) PDF
        if ext.endswith(".pdf"):
            pdf_docs = doc_service.process_pdf(file_path)
            for d in pdf_docs:
                d.metadata["file_name"] = file.name
            all_docs.extend(pdf_docs)
            category = "both"

        # 2) Word (doc/docx)
        elif ext.endswith((".docx", ".doc")):
            try:
                text = docx2txt.process(file_path)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
            text_docs = doc_service.process_text(text)
            for d in text_docs:
                d.metadata["file_name"] = file.name
            all_docs.extend(text_docs)
            category = "both"

        # 3) Plain text
        elif ext.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
            text_docs = doc_service.process_text(text)
            for d in text_docs:
                d.metadata["file_name"] = file.name
            all_docs.extend(text_docs)
            category = "both"
        
        # 4) Python files
        elif ext.endswith(".py"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
            
            # Automatically display extracted code
            st.session_state["conversation_history"].append((file.name, f"```python\n{text}\n```"))
            
            text_docs = doc_service.process_text(text)
            for d in text_docs:
                d.metadata["file_name"] = file.name
            all_docs.extend(text_docs)
            category = "chatbot"
        
       # 5) Jupyter Notebook (ipynb)
        elif ext.endswith(".ipynb"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    nb_data = nbformat.read(f, as_version=4)
                
                extracted_code = []
                for cell in nb_data.cells:
                    if cell.cell_type == "code":
                        extracted_code.append("\n".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"])
                
                text = "\n\n".join(extracted_code)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
            
            # Automatically display extracted code
            st.session_state["conversation_history"].append((file.name, f"```python\n{text}\n```"))
            
            text_docs = doc_service.process_text(text)
            for d in text_docs:
                d.metadata["file_name"] = file.name
            all_docs.extend(text_docs)
            category = "chatbot"
            
        else:
            # If user uploads unsupported type (e.g., mp3, wav, etc.):
            st.warning(f"Unsupported file type for main chatbot: {file.name}. Use transcribe_translate.py for audio/video.")
            continue

        # Store file category
        st.session_state["file_categories"][file.name] = category

    if all_docs:
        doc_service.add_documents_to_vectorstore(all_docs)
        st.session_state["processed_files"].extend(new_files)
        return f"Processed {len(new_files)} new file(s) successfully!"
    return "No valid text extracted from these files."

def retrieve_relevant_chunks(query: str, top_k: int = 20):
    """
    Fetch chunks relevant to 'query' from the selected file in st.session_state['selected_file'].
    """
    f = st.session_state["selected_file"]
    if not f:
        return ["No file selected."]

    # Ensure chatbot-relevant files are included
    if f not in st.session_state["file_categories"]:
        return ["File category not found."]
    
    category = st.session_state["file_categories"][f]  # Get category

    if category in ["both", "chatbot"]:  # Include chatbot files
        try:
            results = doc_service.vectorstore.similarity_search(query, k=top_k, filter={"file_name": f})
            return [doc.page_content for doc in results] if results else ["No relevant chunks found."]
        except Exception as e:
            return [f"Error retrieving chunks: {e}"]

def estimate_confidence(llm_response: str, context_text: str) -> float:
    """
    Compute a 0-100 confidence score by embedding the LLM response + context, 
    then measuring cosine similarity.
    """
    try:
        resp_emb = embedding_model.embed_query(llm_response)
        ctx_emb = embedding_model.embed_query(context_text)
        sim = cosine_similarity([resp_emb], [ctx_emb])[0][0]
        return min(100.0, max(0.0, (sim + 1) * 50))
    except Exception:
        return 0.0

def process_user_query():
    user_input = st.session_state.user_query.strip()
    if not user_input:
        return
    
    # "summarise" command => summarise the selected file
    if user_input.lower() == "summarise":
        if not st.session_state["processed_files"]:
            st.session_state["conversation_history"].append(("summarise", "No files uploaded."))
            st.session_state.user_query = ""
            return
        fn = st.session_state["selected_file"]
        if not fn:
            st.session_state["conversation_history"].append(("summarise", "No file selected."))
            st.session_state.user_query = ""
            return
        chunks = retrieve_relevant_chunks(fn) if st.session_state["file_categories"].get(fn, "both") in ["both", "chatbot"] else ["This file is not available for chatbot queries."]
        context = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        msgs = [
            {"role": "system", "content": "You are a knowledgeable assistant..."},
            {"role": "user", "content": f"Provide a structured summary of **{fn}**.\n\nContext:\n{context}"}
        ]
        try:
            r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.3)
            ans = r.choices[0].message.content.strip()
            conf = estimate_confidence(ans, context)
            final = f"**Summary of {fn}:**\n{ans}\n\n**Confidence Score:** {conf:.2f}%\n---\n"
            st.session_state["conversation_history"].append(("summarize", final))
        except Exception as e:
            st.session_state["conversation_history"].append(("summarize", f"Error: {e}"))
        st.session_state["user_query"] = ""
        return
    
    # Normal Q&A
    chunks = retrieve_relevant_chunks(user_input)
    context_str = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    msgs = [
        {"role": "system", "content": "You are a knowledgeable assistant...Use only the 'Context' below."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}"}
    ]
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.7)
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str)
        final_ans = f"{answer}\n\n**Confidence Score:** {conf:.2f}%"
        st.session_state["conversation_history"].append((user_input, final_ans))
    except Exception as e:
        st.session_state["conversation_history"].append((user_input, f"Error: {e}"))
    st.session_state.user_query = ""

st.title("🤖 Your Academic Chatbot & Knowledge Hub")

# Sidebar for file upload
with st.sidebar:
    st.subheader("Upload Files")
    # Accept only PDF, DOC, DOCX, TXT 
    files = st.file_uploader(
        "Upload PDF, DOC, DOCX, TXT, PY, or IPYNB",
        type=["pdf","txt","doc", "docx", "py", "ipynb"],
        accept_multiple_files=True
    )
    if files:
        msg = preprocess_files(files)
        st.success(msg)

    if st.session_state["processed_files"]:
        st.subheader("Available Files")
        st.session_state["selected_file"] = st.selectbox(
            "Select a file",
            st.session_state["processed_files"]
        )

# Main Chat
st.header("Chat History")
for user_msg, bot_msg in st.session_state["conversation_history"]:
    st.markdown(f"**🧑‍💻 You:** {user_msg}")
    st.markdown(f"**🤖 ChatBot:**\n\n{bot_msg}")

st.text_input(
    "Enter your query or command ('quiz','summarize'):",
    key="user_query",
    on_change=process_user_query
)
