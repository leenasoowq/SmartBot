import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import numpy as np
import re

from services.language_service import LanguageService
from services.document_service import DocumentService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

lang_service = LanguageService(client)
doc_service = DocumentService(client)
embedding_model = doc_service.embedding_model

# Initialize session state
for key in ["processed_files", "conversation_history", "quiz_mode", "selected_file", "last_bot_question", "last_expected_answer"]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key in ["processed_files", "conversation_history"] 
            else False if key == "quiz_mode"
            else None if key in ["selected_file", "last_expected_answer"]
            else ""  # For last_bot_question
        )

MAX_HISTORY = 5  # Store only the last 5 messages

def update_conversation_history(role, message):
    """Append new message to conversation history while maintaining a fixed size."""
    if role not in ["user", "assistant"]:
        print(f"⚠️ Invalid role detected: {role}, message: {message}")
        return
    st.session_state["conversation_history"].append((role, message))
    if len(st.session_state["conversation_history"]) > MAX_HISTORY:
        st.session_state["conversation_history"] = st.session_state["conversation_history"][-MAX_HISTORY:]

def get_expected_answer(question: str) -> str:
    """Retrieve document chunks and generate an answer using OpenAI."""
    chunks = retrieve_relevant_chunks(question, top_k=20)
    context_text = "\n\n".join(chunks) if chunks else ""
    msgs = [
        {"role": "system", "content": (
            "You are an expert AI assistant. Answer the question using the provided context as much as possible. "
            "If the context is insufficient, generate a well-reasoned answer based on general knowledge."
        )},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\nProvide only the answer."}
    ]
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=100, temperature=0.2)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return "Error retrieving expected answer."

def evaluate_user_answer(user_answer: str, expected_answer: str) -> str:
    """Evaluates user response against the expected answer."""
    if user_answer.lower() in ["i don't know", "idk", "not sure"]:
        return "Don't Know"
    msgs = [
        {"role": "system", "content": "You evaluate if a response is correct. Reply only with 'Correct', 'Close', or 'Incorrect'."},
        {"role": "user", "content": f"Expected: {expected_answer}\nUser: {user_answer}\nHow accurate is the user's response?"}
    ]
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=10, temperature=0.2)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return "Evaluation Error"

def sanitize_collection_name(file_name: str) -> str:
    """Sanitize file name for Chroma collection."""
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
    """Estimate confidence based on cosine similarity."""
    try:
        resp_emb = embedding_model.embed_query(llm_response)
        ctx_emb = embedding_model.embed_query(context_text)
        sim = cosine_similarity([resp_emb], [ctx_emb])[0][0]
        return min(100.0, max(0.0, (sim + 1) * 50))
    except Exception:
        return 0.0

def process_user_query(user_input):
    """Process the user's query and generate a response."""
    if not user_input.strip():
        return

    update_conversation_history("user", user_input)

    if user_input.lower() == "summarise":
        if not st.session_state["processed_files"]:
            update_conversation_history("assistant", "No files uploaded.")
            st.rerun()  # Force UI update
            return
        fn = st.session_state["selected_file"]
        if not fn:
            update_conversation_history("assistant", "No file selected.")
            st.rerun()  # Force UI update
            return
        chunks = retrieve_relevant_chunks(fn)
        context = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        msgs = [
            {"role": "system", "content": "You are a knowledgeable assistant."},
            {"role": "user", "content": f"Provide a structured summary of **{fn}**.\n\nContext:\n{context}"}
        ]
        try:
            r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.3)
            ans = r.choices[0].message.content.strip()
            conf = estimate_confidence(ans, context)
            final = f"**Summary of {fn}:**\n{ans}\n\n**Confidence Score:** {conf:.2f}%\n---\n"
            update_conversation_history("assistant", final)
        except Exception as e:
            update_conversation_history("assistant", f"Error: {e}")
        st.rerun()  # Force UI update
        return

    classification_msgs = [
        {"role": "system", "content": "You analyze user requests to determine if they are asking to be tested on a topic."},
        {"role": "user", "content": f"Analyze this request and respond only with 'yes' or 'no': {user_input}"}
    ]
    try:
        classification_response = client.chat.completions.create(
            model="gpt-4", messages=classification_msgs, max_tokens=10, temperature=0.2
        )
        is_test_request = classification_response.choices[0].message.content.strip().lower()
    except Exception as e:
        update_conversation_history("assistant", f"Error in classification: {e}")
        st.rerun()  # Force UI update
        return

    if is_test_request == "yes":
        file_name = st.session_state.get("selected_file", None)
        if not file_name:
            update_conversation_history("assistant", "No document is currently selected. Please upload or select a file first.")
            st.rerun()  # Force UI update
            return
        chunks = retrieve_relevant_chunks(file_name)
        document_text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        if not document_text.strip():
            update_conversation_history("assistant", "Could not extract relevant content from the document.")
            st.rerun()  # Force UI update
            return
        msgs = [
            {"role": "system", "content": (
                "You generate a single question based on the provided content. "
                "Ensure the question is relevant, fact-based, and answerable from the document."
            )},
            {"role": "user", "content": f"Generate exactly one question based on the following content:\n\n{document_text}"}
        ]
        try:
            r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=100, temperature=0.7)
            question = r.choices[0].message.content.strip().split("? ")[0] + "?"
            update_conversation_history("assistant", f"Here is your test question:\n\n**{question}**\n\nNote: I can only generate one question at a time.")
            st.session_state["last_bot_question"] = question
            st.session_state["last_expected_answer"] = get_expected_answer(question)
        except Exception as e:
            update_conversation_history("assistant", f"Error: {e}")
        st.rerun()  # Force UI update
        return

    if st.session_state["last_bot_question"] and st.session_state["last_expected_answer"]:
        evaluation = evaluate_user_answer(user_input, st.session_state["last_expected_answer"])
        if evaluation == "Don't Know":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"You didn't know the answer. Here is the correct answer:\n\n**{correct_answer}**"
        elif evaluation == "Incorrect":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"**Evaluation:** {evaluation}\n\nThe correct answer is: {correct_answer}"
        else:
            feedback = f"**Evaluation:** {evaluation}"
        update_conversation_history("assistant", feedback)
        st.session_state["last_bot_question"] = ""
        st.session_state["last_expected_answer"] = ""
        st.rerun()  # Force UI update
        return

    chunks = retrieve_relevant_chunks(user_input)
    context_str = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    msgs = [{"role": "system", "content": "You are a knowledgeable assistant. Use only the provided context."}]
    for role, message in st.session_state["conversation_history"][-5:]:
        if "Evaluation:" not in message and "Correct Answer:" not in message:
            msgs.append({"role": role, "content": message})
    msgs.append({"role": "user", "content": user_input})
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.7)
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str)
        final_ans = f"{answer}\n\n**Confidence Score:** {conf:.2f}%"
        update_conversation_history("assistant", final_ans)
        if answer.endswith("?"):
            st.session_state["last_bot_question"] = answer
            st.session_state["last_expected_answer"] = get_expected_answer(answer)
    except Exception as e:
        update_conversation_history("assistant", f"Error: {e}")
    st.rerun()  # Force UI update

# UI Setup
st.title("🤖 Your Academic Chatbot")

if "uploading" not in st.session_state:
    st.session_state["uploading"] = False
if "files_to_upload" not in st.session_state:
    st.session_state["files_to_upload"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if st.session_state["uploading"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], .block-container {
            pointer-events: none !important;
            opacity: 0.5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOC, DOCX, or TXT",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state['uploader_key']}"
    )
    if uploaded_files:
        st.session_state["files_to_upload"] = uploaded_files
    if st.session_state["files_to_upload"]:
        if st.button("Upload", disabled=st.session_state["uploading"]):
            st.session_state["uploading"] = True
            with st.spinner("Processing files... Please wait!"):
                msg = preprocess_files(st.session_state["files_to_upload"])
            st.success(msg)
            st.session_state["files_to_upload"] = None
            st.session_state["uploading"] = False
            st.session_state["uploader_key"] += 1
            st.rerun()

    if st.session_state["processed_files"]:
        st.subheader("Available Files")
        st.session_state["selected_file"] = st.selectbox(
            "Select a file",
            st.session_state["processed_files"],
            disabled=st.session_state["uploading"]
        )

if st.session_state["uploading"]:
    st.stop()

# Chat Interface
st.header("Chat")
chat_container = st.container()
with chat_container:
    for role, message in st.session_state.get("conversation_history", []):
        with st.chat_message(role):
            st.markdown(message)

# Dynamic Chat Input
prompt = st.chat_input("Enter your query:", disabled=st.session_state["uploading"])
if prompt:
    process_user_query(prompt)