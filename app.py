import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import time

# Additional import for reading Word docs
import docx2txt

# Local imports
from services.language_service import LanguageService
from services.document_service import DocumentService

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Configure application constants
MAX_FILE_SIZE_MB = 10
MODEL_NAME = "gpt-3.5-turbo"  # Fallback to gpt-3.5-turbo for large contexts instead of gpt-4
MAX_CONTEXT_TOKENS = 4000  # Conservative limit to prevent context overflow
MAX_OUTPUT_TOKENS = 800  # Limit response size

# Instantiate Services
lang_service = LanguageService(client)
doc_service = DocumentService(client, max_file_size_mb=MAX_FILE_SIZE_MB)
embedding_model = doc_service.embedding_model

# Initialize session state
for key in ["processed_files", "conversation_history", "quiz_mode", "selected_file", "processing_status"]:
    if key not in st.session_state:
        # processed_files & conversation_history as lists; quiz_mode is bool; selected_file is None
        st.session_state[key] = (
            [] if key in ["processed_files", "conversation_history"] 
            else False if key == "quiz_mode"
            else None if key == "selected_file"
            else ""  # processing_status
        )

def preprocess_files(files):
    """
    Upload & parse PDF, DOC, DOCX, TXT files ONLY.
    Extract text, then embed into the vectorstore with 'file_name' metadata.
    Audio/Video are handled elsewhere (transcribe_translate.py).
    """
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Filter only new files
    new_files = [f for f in files if f.name not in st.session_state["processed_files"]]
    if not new_files:
        return "No new files to process."

    st.session_state["processing_status"] = "Processing files..."
    
    processed_count = 0
    skipped_files = []
    
    for file in new_files:
        # Check file size before saving
        file_size_mb = file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            skipped_files.append(f"{file.name} (Size: {file_size_mb:.2f} MB exceeds {MAX_FILE_SIZE_MB} MB limit)")
            continue
            
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as out_file:
            out_file.write(file.getbuffer())

        try:
            all_docs = []
            ext = file.name.lower()
            
            # 1) PDF
            if ext.endswith(".pdf"):
                pdf_docs = doc_service.process_pdf(file_path)
                for d in pdf_docs:
                    d.metadata["file_name"] = file.name
                all_docs.extend(pdf_docs)

            # 2) Word (doc/docx)
            elif ext.endswith((".docx", ".doc")):
                try:
                    text = docx2txt.process(file_path)
                    text_docs = doc_service.process_text(text)
                    for d in text_docs:
                        d.metadata["file_name"] = file.name
                    all_docs.extend(text_docs)
                except Exception as e:
                    skipped_files.append(f"{file.name} (Error: {str(e)})")
                    continue

            # 3) Plain text
            elif ext.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    text_docs = doc_service.process_text(text)
                    for d in text_docs:
                        d.metadata["file_name"] = file.name
                    all_docs.extend(text_docs)
                except Exception as e:
                    skipped_files.append(f"{file.name} (Error: {str(e)})")
                    continue
            else:
                # If user uploads unsupported type (e.g., mp3, wav, etc.):
                skipped_files.append(f"{file.name} (Unsupported file type)")
                continue
            
            # If we extracted some docs, add to vectorstore
            if all_docs:
                doc_service.add_documents_to_vectorstore(all_docs)
                st.session_state["processed_files"].append(file.name)
                processed_count += 1
                
        except Exception as e:
            skipped_files.append(f"{file.name} (Error: {str(e)})")
            continue

    st.session_state["processing_status"] = ""
    
    # Prepare result message
    result = f"Processed {processed_count} new file(s) successfully!"
    if skipped_files:
        result += f"\n\nSkipped files:\n" + "\n".join(skipped_files)
    
    return result

def retrieve_relevant_chunks(query: str, top_k: int = 5):
    """
    Fetch chunks relevant to 'query' from the selected file in st.session_state['selected_file'].
    """
    f = st.session_state["selected_file"]
    if not f:
        return ["No file selected."]
    try:
        # Truncate query to prevent batch size issues
        safe_query = query[:100] if len(query) > 100 else query
        
        # Use metadata filter so we only retrieve content from the chosen file
        results = doc_service.retrieve_relevant_chunks(
            safe_query, top_k=top_k, filter_metadata={"file_name": f}
        )
        return results if results else ["No relevant chunks found."]
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return [f"Error retrieving chunks: {e}"]

def estimate_confidence(llm_response: str, context_text: str) -> float:
    """
    Compute a 0-100 confidence score by embedding the LLM response + context.
    Ultra-conservative implementation to prevent batch size errors.
    """
    try:
        # Severely limit text size to prevent batch issues
        max_chars = 50  # Very conservative
        
        # Take only the beginning of each text
        response_sample = llm_response[:max_chars]
        context_sample = context_text[:max_chars]
        
        resp_emb = embedding_model.embed_query(response_sample)
        ctx_emb = embedding_model.embed_query(context_sample)
        
        sim = cosine_similarity([resp_emb], [ctx_emb])[0][0]
        return min(100.0, max(0.0, (sim + 1) * 50))
    except Exception as e:
        print(f"Confidence estimation error: {e}")
        return 50.0  # Return neutral score on error

def process_user_query():
    user_input = st.session_state.user_query.strip()
    if not user_input:
        return
    
    # Add user message to history immediately
    st.session_state["conversation_history"].append((user_input, "Processing your query..."))
    
    # "summarise" command => summarise the selected file
    if user_input.lower() == "summarise":
        if not st.session_state["processed_files"]:
            st.session_state["conversation_history"][-1] = (user_input, "No files uploaded.")
            st.session_state.user_query = ""
            return
        fn = st.session_state["selected_file"]
        if not fn:
            st.session_state["conversation_history"][-1] = (user_input, "No file selected.")
            st.session_state.user_query = ""
            return
        
        chunks = retrieve_relevant_chunks(fn, top_k=10)
        # Join only first few chunks to avoid context overflow
        context = "\n\n".join(chunks[:3]) if isinstance(chunks, list) else str(chunks)
        
        # Limit context size
        if len(context) > MAX_CONTEXT_TOKENS * 4:  # Rough char estimate
            context = context[:MAX_CONTEXT_TOKENS * 4]
        
        msgs = [
            {"role": "system", "content": "You are an academic summarizer focused on providing concise, accurate summaries of content."},
            {"role": "user", "content": f"Provide a structured summary of **{fn}**.\n\nContext:\n{context}"}
        ]
        try:
            # Use more performant model for large contexts
            r = client.chat.completions.create(
                model=MODEL_NAME, 
                messages=msgs, 
                max_tokens=MAX_OUTPUT_TOKENS, 
                temperature=0.3
            )
            ans = r.choices[0].message.content.strip()
            conf = estimate_confidence(ans, context)
            final = f"**Summary of {fn}:**\n{ans}\n\n**Confidence Score:** {conf:.2f}%\n---\n"
            st.session_state["conversation_history"][-1] = (user_input, final)
        except Exception as e:
            st.session_state["conversation_history"][-1] = (user_input, f"Error: {e}")
        st.session_state["user_query"] = ""
        return
    
    # Normal Q&A
    chunks = retrieve_relevant_chunks(user_input, top_k=5)
    context_str = "\n\n".join(chunks[:3]) if isinstance(chunks, list) else str(chunks)
    
    # Limit context size
    if len(context_str) > MAX_CONTEXT_TOKENS * 4:  # Rough char estimate
        context_str = context_str[:MAX_CONTEXT_TOKENS * 4]
    
    msgs = [
        {"role": "system", "content": "You are a knowledgeable assistant that provides accurate information based only on the context provided."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}"}
    ]
    try:
        r = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=msgs, 
            max_tokens=MAX_OUTPUT_TOKENS, 
            temperature=0.7
        )
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str)
        final_ans = f"{answer}\n\n**Confidence Score:** {conf:.2f}%"
        st.session_state["conversation_history"][-1] = (user_input, final_ans)
    except Exception as e:
        st.session_state["conversation_history"][-1] = (user_input, f"Error: {e}")
    st.session_state.user_query = ""

def main():
    st.title("🤖 Your Academic Chatbot")

    # Sidebar for file upload
    with st.sidebar:
        st.subheader("Upload Files")
        # Accept only PDF, DOC, DOCX, TXT 
        files = st.file_uploader(
            "Upload PDF, DOC, DOCX, or TXT (Max 10MB per file)",
            type=["pdf","txt","doc","docx"],
            accept_multiple_files=True
        )
        
        upload_button = st.button("Process Files")
        
        if files and upload_button:
            with st.spinner("Processing files..."):
                msg = preprocess_files(files)
                st.success(msg)

        # Display processing status if any
        if st.session_state["processing_status"]:
            st.info(st.session_state["processing_status"])

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
        st.divider()

    # User input
    st.text_input(
        "Enter your query:",
        key="user_query",
        on_change=process_user_query
    )
    
    # Add a clear chat button
    if st.button("Clear Chat History"):
        st.session_state["conversation_history"] = []
        st.experimental_rerun

if __name__ == "__main__":
    main()