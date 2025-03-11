import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt

from services.language_service import LanguageService
from services.document_service import DocumentService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

lang_service = LanguageService(client)
doc_service = DocumentService(client)
embedding_model = doc_service.embedding_model

for key in ["processed_files", "conversation_history", "quiz_mode", "selected_file"]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key in ["processed_files", "conversation_history"] 
            else False if key == "quiz_mode"
            else None
        )

def sanitize_collection_name(file_name: str) -> str:
    """Sanitize file name for Chroma collection: alphanumeric, underscore, hyphen only."""
    base_name = file_name.rsplit(".", 1)[0]
    sanitized = "".join(c for c in base_name if c.isalnum() or c in ["_", "-"])
    while "__" in sanitized or "--" in sanitized:
        sanitized = sanitized.replace("__", "_").replace("--", "-")
    sanitized = sanitized.strip("_-")
    sanitized = sanitized[:63]
    if len(sanitized) < 3:
        sanitized += "_doc"
    return sanitized

def preprocess_files(files):
    """Upload and parse PDF, DOC/DOCX, TXT files into collections."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    new_files = [f.name for f in files if f.name not in st.session_state["processed_files"]]
    if not new_files:
        return "No new files to process."

    processed_count = 0
    for file in files:
        if file.name not in new_files:
            continue
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as out_file:
            out_file.write(file.getbuffer())

        ext = file.name.lower()
        docs = []
        if ext.endswith(".pdf"):
            docs = doc_service.process_pdf(file_path)
        elif ext.endswith((".docx", ".doc")):
            try:
                text = docx2txt.process(file_path)
                docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        elif ext.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        if docs:
            collection_name = sanitize_collection_name(file.name)
            for d in docs:
                d.metadata["file_name"] = file.name
            doc_service.add_documents_to_vectorstore(docs, collection_name)
            st.session_state["processed_files"].append(file.name)
            processed_count += 1
        os.remove(file_path)

    return f"Processed {processed_count} new file(s) successfully!" if processed_count else "No valid text extracted."

def retrieve_relevant_chunks(query: str, top_k: int = 20):
    """Fetch chunks from the selected file’s collection."""
    f = st.session_state["selected_file"]
    if not f:
        return ["No file selected."]
    try:
        collection_name = sanitize_collection_name(f)
        results = doc_service.retrieve_relevant_chunks(query, collection_name, top_k)
        return results if results else ["No relevant chunks found."]
    except Exception as e:
        return [f"Error retrieving chunks: {e}"]

def estimate_confidence(llm_response: str, context_text: str) -> float:
    try:
        resp_emb = embedding_model.embed_query(llm_response)
        ctx_emb = embedding_model.embed_query(context_text)
        sim = cosine_similarity([resp_emb], [ctx_emb])[0][0]
        return min(100.0, max(0.0, (sim + 1) * 50))
    except Exception:
        return 0.0

# def process_user_query():
#     user_input = st.session_state.user_query.strip()
#     if not user_input:
#         return
    
#     if user_input.lower() == "summarise":
#         if not st.session_state["processed_files"]:
#             st.session_state["conversation_history"].append(("summarise", "No files uploaded."))
#             st.session_state.user_query = ""
#             return
#         fn = st.session_state["selected_file"]
#         if not fn:
#             st.session_state["conversation_history"].append(("summarise", "No file selected."))
#             st.session_state.user_query = ""
#             return
#         chunks = retrieve_relevant_chunks(fn)
#         context = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
#         msgs = [
#             {"role": "system", "content": "You are a knowledgeable assistant..."},
#             {"role": "user", "content": f"Provide a structured summary of **{fn}**.\n\nContext:\n{context}"}
#         ]
#         try:
#             r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.3)
#             ans = r.choices[0].message.content.strip()
#             conf = estimate_confidence(ans, context)
#             final = f"**Summary of {fn}:**\n{ans}\n\n**Confidence Score:** {conf:.2f}%\n---\n"
#             st.session_state["conversation_history"].append(("summarize", final))
#         except Exception as e:
#             st.session_state["conversation_history"].append(("summarize", f"Error: {e}"))
#         st.session_state["user_query"] = ""
#         return
    
#     chunks = retrieve_relevant_chunks(user_input)
#     context_str = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
#     msgs = [
#         {"role": "system", "content": "You are a knowledgeable assistant...Use only the 'Context' below."},
#         {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}"}
#     ]
#     try:
#         r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.7)
#         answer = r.choices[0].message.content.strip()
#         conf = estimate_confidence(answer, context_str)
#         final_ans = f"{answer}\n\n**Confidence Score:** {conf:.2f}%"
#         st.session_state["conversation_history"].append((user_input, final_ans))
#     except Exception as e:
#         st.session_state["conversation_history"].append((user_input, f"Error: {e}"))
#     st.session_state.user_query = ""

def process_user_query():
    user_input = st.session_state.user_query.strip()
    if not user_input:
        return
    
    # Summarization Mode
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
        chunks = retrieve_relevant_chunks(fn)
        context = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        
        # Preserve previous interactions
        conversation_history = [
            {"role": "system", "content": "You are a knowledgeable assistant..."}
        ]
        for user_msg, bot_msg in st.session_state["conversation_history"][-5:]:
            conversation_history.append({"role": "user", "content": user_msg})
            conversation_history.append({"role": "assistant", "content": bot_msg})
        
        conversation_history.append({"role": "user", "content": f"Provide a structured summary of **{fn}**.\n\nContext:\n{context}"})
        
        try:
            r = client.chat.completions.create(model="gpt-4", messages=conversation_history, max_tokens=1500, temperature=0)
            ans = r.choices[0].message.content.strip()
            conf = estimate_confidence(ans, context)
            final = f"**Summary of {fn}:**\n{ans}\n\n**Confidence Score:** {conf:.2f}%\n---\n"
            st.session_state["conversation_history"].append(("summarize", final))
        except Exception as e:
            st.session_state["conversation_history"].append(("summarize", f"Error: {e}"))
        
        st.session_state.user_query = ""
        return
    
    # Regular Query Processing
    chunks = retrieve_relevant_chunks(user_input)
    context_str = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    
    # Preserve conversation history
    conversation_history = [
        {"role": "system", "content": "You are a knowledgeable assistant...Use only the 'Context' below."}
    ]
    for user_msg, bot_msg in st.session_state["conversation_history"][-5:]:  # Store last 5 interactions
        conversation_history.append({"role": "user", "content": user_msg})
        conversation_history.append({"role": "assistant", "content": bot_msg})
    
    conversation_history.append({"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {user_input}"})
    
    try:
        r = client.chat.completions.create(model="gpt-4", messages=conversation_history, max_tokens=1500, temperature=0.7)
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str)
        final_ans = f"{answer}\n\n**Confidence Score:** {conf:.2f}%"
        st.session_state["conversation_history"].append((user_input, final_ans))
    except Exception as e:
        st.session_state["conversation_history"].append((user_input, f"Error: {e}"))
    
    st.session_state.user_query = ""

st.title("🤖 Your Academic Chatbot")

with st.sidebar:
    st.subheader("Upload Files")
    files = st.file_uploader(
        "Upload PDF, DOC, DOCX, or TXT",
        type=["pdf", "txt", "doc", "docx"],
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

st.header("Chat History")
for user_msg, bot_msg in st.session_state["conversation_history"]:
    st.markdown(f"**🧑‍💻 You:** {user_msg}")
    st.markdown(f"**🤖 ChatBot:**\n\n{bot_msg}")

st.text_input(
    "Enter your query:",
    key="user_query",
    on_change=process_user_query
)