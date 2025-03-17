import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import docx2txt
import numpy as np
import re
import json

from services.language_service import LanguageService
from services.document_service import DocumentService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

lang_service = LanguageService(client)
doc_service = DocumentService(client)
embedding_model = doc_service.embedding_model

# File to store persistent data
PROCESSED_FILES_STORAGE = "data/processed_files.json"
# Load processed files from disk if exists
def load_processed_files():
    if os.path.exists(PROCESSED_FILES_STORAGE):
        try:
            with open(PROCESSED_FILES_STORAGE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading processed files: {e}")
            return []
    return []

# Save processed files to disk
def save_processed_files(files):
    try:
        with open(PROCESSED_FILES_STORAGE, 'w') as f:
            json.dump(files, f)
    except Exception as e:
        st.error(f"Error saving processed files: {e}")


# Initialize session state
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = load_processed_files()

for key in ["conversation_history", "quiz_mode", "selected_file", "last_bot_question", "last_expected_answer", "pending_response"]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key in ["conversation_history"] 
            else False if key == "quiz_mode"
            else None if key in ["selected_file", "last_expected_answer", "pending_response"]
            else ""  # For last_bot_question
        )


MAX_HISTORY = 50

def update_conversation_history(role, message):
    """Append new message to conversation history and store last bot response."""
    if role not in ["user", "assistant"]:
        print(f"⚠️ Invalid role detected: {role}, message: {message}")
        return
    
    st.session_state["conversation_history"].append((role, message))
    
    # Store last bot response explicitly for follow-ups
    if role == "assistant":
        st.session_state["last_bot_answer"] = message  # Store last bot answer

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

def is_follow_up(user_input: str) -> bool:
    """Determine if the user input is a follow-up question based on previous context."""
    last_bot_response = st.session_state.get("last_bot_answer", None)
    
    if not last_bot_response:
        return False  # If there's no last bot response, it's not a follow-up.

    classification_msgs = [
        {"role": "system", "content": "You analyze user queries to determine if they are follow-up questions. "
                                      "A follow-up question refers to the previous response and asks for clarification, elaboration, or continuation. "
                                      "Respond with only 'yes' or 'no'."},
        {"role": "assistant", "content": last_bot_response},  # Pass the last response
        {"role": "user", "content": user_input}
    ]

    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=classification_msgs,
            max_tokens=5,
            temperature=0.2
        )
        return r.choices[0].message.content.strip().lower() == "yes"
    except Exception as e:
        print(f"Error in follow-up detection: {e}")
        return False  # Default to treating it as a new question

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
    processed_files = st.session_state["processed_files"]
    new_files = [f.name for f in files if f.name not in processed_files]

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
            processed_files.append(file.name)
            processed_count += 1
        os.remove(file_path)
     # Update the session state and save to disk
    st.session_state["processed_files"] = processed_files
    save_processed_files(processed_files)
    
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

def process_response(user_input):
    """Process the user's query and generate a response with source attribution."""
    if not user_input.strip():
        return
    
    # 🔥 Handle AI-Based Follow-Up Questions 🔥
    if is_follow_up(user_input):
        last_bot_response = st.session_state.get("last_bot_answer", "")

        if last_bot_response:
            follow_up_msgs = [
                {"role": "system", "content": "You provide follow-up answers based on the user's previous question and your last response. "
                                              "Ensure the follow-up answer expands on the previous response in a meaningful way."},
                {"role": "assistant", "content": last_bot_response},  # Pass last answer
                {"role": "user", "content": user_input}
            ]
            try:
                r = client.chat.completions.create(
                    model="gpt-4",
                    messages=follow_up_msgs,
                    max_tokens=200,
                    temperature=0.7
                )
                follow_up_response = r.choices[0].message.content.strip()
                update_conversation_history("assistant", follow_up_response)
                return  # Avoid treating it as a new question
            except Exception as e:
                update_conversation_history("assistant", f"Error in follow-up: {e}")
                return

    # General Q&A or Quiz Logic
    chunks = retrieve_relevant_chunks(user_input)
    context_str = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
    if not chunks or "No relevant chunks found" in context_str or "Error" in context_str:
        context_str = ""
        fallback_note = "\n\n**Note:** No relevant document content found; response based on general knowledge."
    else:
        fallback_note = ""

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
        return

    if is_test_request == "yes":
        file_name = st.session_state.get("selected_file", None)
        if not file_name:
            update_conversation_history("assistant", "No document is currently selected. Please upload or select a file first.")
            return
        chunks = retrieve_relevant_chunks(file_name)
        document_text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        if not document_text.strip():
            update_conversation_history("assistant", "Could not extract relevant content from the document.")
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
        return

    if st.session_state["last_bot_question"] and st.session_state["last_expected_answer"]:
        evaluation = evaluate_user_answer(user_input, st.session_state["last_expected_answer"])
        if evaluation == "Don't Know":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"You didn't know the answer. Here is the correct answer:\n\n**{correct_answer}**"
        elif evaluation == "Incorrect":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"**Evaluation:** {evaluation}\n\nThe correct answer is: {correct_answer}"
        elif evaluation == "Close":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"**Evaluation:** {evaluation}\n\nThe correct answer is: {correct_answer}"
        else:
            feedback = f"**Evaluation:** {evaluation}"
        update_conversation_history("assistant", feedback)
        st.session_state["last_bot_question"] = ""
        st.session_state["last_expected_answer"] = ""
        return

    # General Q&A
    msgs = [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. Use only the provided context to answer the query. "
            "If the context is empty or insufficient, indicate that the answer is based on general knowledge."
        )}
    ]
    for role, message in st.session_state["conversation_history"][-5:]:
        if "Evaluation:" not in message and "Correct Answer:" not in message:
            msgs.append({"role": role, "content": message})
    msgs.append({"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {user_input}"})
    
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.7)
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str) if context_str else 0.0
        source_note = (
            f"**Source:** Extracted from document chunks (Confidence: {conf:.2f}%)"
            if context_str else "**Source:** General knowledge (No relevant document content found)"
        )
        final_ans = f"{answer}\n\n{source_note}{fallback_note if not context_str else ''}"
        
        # Optional: Show retrieved chunks in an expander
        if context_str:
            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(chunks[:5]):  # Limit to 5 for brevity
                    st.write(f"Chunk {i+1}: {chunk[:200]}...")  # Truncate for display
        
        update_conversation_history("assistant", final_ans)
        if answer.endswith("?"):
            st.session_state["last_bot_question"] = answer
            st.session_state["last_expected_answer"] = get_expected_answer(answer)
    except Exception as e:
        update_conversation_history("assistant", f"Error: {e}")

def reset_session_state():
    """Reset session variables that store conversation history and other related data."""
    keys_to_reset = [
        "conversation_history", "quiz_mode", "selected_file", 
        "last_bot_question", "last_expected_answer", "pending_response"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]


def remove_file():
    """Remove a file from the processed files list."""
    if not st.session_state["selected_file"]:
        return
    
    file_to_remove = st.session_state["selected_file"]
    
    # Try to remove the collection from Chroma
    try:
        collection_name = sanitize_collection_name(file_to_remove)
        doc_service.delete_collection(collection_name)
    except Exception as e:
        st.error(f"Error removing collection: {e}")
    
    # Update processed files
    processed_files = st.session_state["processed_files"]
    if file_to_remove in processed_files:
        processed_files.remove(file_to_remove)
        st.session_state["selected_file"] = None
        
    # Save updated list to disk
    st.session_state["processed_files"] = processed_files
    save_processed_files(processed_files)
    st.success(f"Removed {file_to_remove} from your documents")
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
    update_conversation_history("user", prompt)
    st.session_state["pending_response"] = prompt
    st.rerun()

# Process the response if there's a pending input
if st.session_state["pending_response"]:
    with st.spinner("Thinking..."):
        process_response(st.session_state["pending_response"])
    st.session_state["pending_response"] = None
    st.rerun()

if st.button("Clear History"):
    reset_session_state()
    st.rerun()  # Re-run the app to refresh the UI
