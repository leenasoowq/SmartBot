import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
import docx2txt
import numpy as np
import re
import json
import shutil


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
    
    # Ensure session state is initialized
    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []

    # Append the new message
    st.session_state["conversation_history"].append((role, message))
    
    # Store the last bot response explicitly for follow-ups
    if role == "assistant":
        st.session_state["last_bot_answer"] = message  

    # Maintain history limit
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
        text_docs, image_docs = [], []
        
        if ext.endswith(".pdf"):
            text_docs, image_docs = doc_service.process_pdf(file_path)
        elif ext.endswith((".docx", ".doc")):
            try:
                text = docx2txt.process(file_path)
                text_docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        elif ext.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text_docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        if text_docs or image_docs:
            collection_name = sanitize_collection_name(file.name)
            
            # Add metadata to text documents
            for d in text_docs:
                if isinstance(d, Document):
                    d.metadata["file_name"] = file.name
            
            # Add metadata to image documents
            for d in image_docs:
                if isinstance(d, Document):
                    d.metadata["file_name"] = file.name
            
            # Add documents separately to vector store
            if text_docs:
                doc_service.add_documents_to_vectorstore(text_docs, collection_name)
            if image_docs:
                doc_service.add_documents_to_vectorstore(image_docs, f"{collection_name}_images")
            
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
    
def extract_page_number(query):
    match = re.search(r"page (\d+)", query, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None



def process_response(user_input):
    """Process the user's query and generate a response with source attribution."""
    if not user_input.strip():
        return
    
    if detect_image_request(user_input):
        return handle_image_request(user_input)
    
    if detect_summary_request(user_input):
        return handle_summary_request()
    
    if detect_quiz_request(user_input):
        return handle_quiz_request()
    
    if is_quiz_answering():
        return handle_quiz_answering(user_input)
    
    if detect_weakness_request(user_input):
        return handle_weakness_analysis()
    
    return handle_general_query(user_input)

def detect_image_request(user_input):
    return bool(re.search(r"image (?:at|on|in) page (\d+)", user_input.lower()))

def detect_summary_request(user_input):
    return "summarize" in user_input.lower() or "give a summary" in user_input.lower()

def detect_quiz_request(user_input):
    return "quiz" in user_input.lower() or "test me" in user_input.lower()

def detect_weakness_request(user_input):
    """Detects when the user asks about their weaknesses or what they should study."""
    return any(phrase in user_input.lower() for phrase in ["my weakness", "what should i study", "what am i weak at"])

def is_quiz_answering():
    return bool(st.session_state.get("last_bot_question")) and bool(st.session_state.get("last_expected_answer"))

def handle_image_request(user_input):
    match = re.search(r"image (?:at|on|in) page (\d+)", user_input.lower()) 
    page_num = int(match.group(1))
    image_data = doc_service.retrieve_image_summary(page_num)
    
    if image_data:
        response_text = f"**📷 Image Explanation (Page {image_data['page_number']})**\n\n📌 **Summary:** {image_data['summary']}\n\n🖼️ *(Stored Image Path: {image_data['image_path']})*"
    else:
        response_text = f"⚠️ No image metadata found for Page {page_num}."
    
    update_conversation_history("assistant", response_text)

def handle_summary_request():
    file_name = st.session_state.get("selected_file")
    
    if not file_name:
        update_conversation_history("assistant", "⚠️ No document is selected. Please upload or select a file first.")
        return
    
    # Retrieve content
    chunks = retrieve_relevant_chunks(file_name)
    
    # Debug: Log chunk contents
    if not chunks:
        update_conversation_history("assistant", "⚠️ Could not extract content from the document.")
        print("Debug: No content extracted from document.")
        return
    
    document_text = "\n\n".join(chunks)
    
    # Debug: Log document text length
    print(f"Debug: Extracted document text length - {len(document_text)} characters")

    if len(document_text.strip()) < 100:
        update_conversation_history("assistant", "⚠️ Document content is too short for summarization.")
        print("Debug: Extracted text is too short.")
        return
    
    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize the provided text concisely."},
                {"role": "user", "content": f"Summarize the following document:\n\n{document_text}"}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        summary = r.choices[0].message.content.strip()
        
        # Debug: Log summary output
        print(f"Debug: Generated summary - {summary}")

        update_conversation_history("assistant", f"📌 **Summary:**\n\n{summary}")
    except Exception as e:
        update_conversation_history("assistant", f"Error generating summary: {e}")
        print(f"Debug: Error in summary generation - {e}")

def handle_quiz_request():
    """Generate a quiz question based on document content."""
    file_name = st.session_state.get("selected_file")
    if not file_name:
        return update_conversation_history("assistant", "⚠️ No document selected. Upload or select a file first.")
    
    chunks = retrieve_relevant_chunks(file_name)
    if not chunks:
        return update_conversation_history("assistant", "⚠️ Could not extract content from the document.")
    
    document_text = "\n\n".join(chunks)
    if len(document_text.strip()) < 100:
        return update_conversation_history("assistant", "⚠️ Document content is too short for quiz generation.")
    
    try:
        r = client.chat.completions.create(model="gpt-4", messages=[
            {"role": "system", "content": "You generate a **single** question based on the provided content."},
            {"role": "user", "content": f"Generate exactly one question based on the following document:\n\n{document_text}"}
        ], max_tokens=100, temperature=0.3)
        
        question = r.choices[0].message.content.strip().split("? ")[0] + "?"
        update_conversation_history("assistant", f"📝 **Quiz Question:**\n\n**{question}**")
        
        st.session_state["last_bot_question"] = question
        st.session_state["last_expected_answer"] = get_expected_answer(question)
    except Exception as e:
        update_conversation_history("assistant", f"Error generating quiz question: {e}")

def handle_quiz_answering(user_input):
    """Evaluate the user's quiz answer and track incorrect responses."""
    correct_answer = st.session_state["last_expected_answer"]
    evaluation = evaluate_user_answer(user_input, correct_answer)

    if "quiz_history" not in st.session_state:
        st.session_state["quiz_history"] = []  # Initialize quiz memory
    
    quiz_result = {
        "question": st.session_state["last_bot_question"],
        "user_answer": user_input,
        "correct_answer": correct_answer,
        "evaluation": evaluation
    }
    
    st.session_state["quiz_history"].append(quiz_result)  # Store result

    # If the answer is incorrect or the user said "idk", store it in weaknesses
    if evaluation.lower() == "incorrect" or user_input.lower() in ["idk", "i don't know"]:
        if "weak_topics" not in st.session_state:
            st.session_state["weak_topics"] = []
        st.session_state["weak_topics"].append(quiz_result)

    feedback = f"✅ **Evaluation:** {evaluation}\n\n📌 Correct Answer: {correct_answer}" if evaluation.lower() != "incorrect" else f"❌ **Incorrect.**\n\n📌 Correct Answer: {correct_answer}"

    update_conversation_history("assistant", feedback)
    st.session_state["last_bot_question"] = ""
    st.session_state["last_expected_answer"] = ""

def handle_weakness_analysis():
    """Analyze user's past quiz results and determine weaknesses."""
    if "weak_topics" not in st.session_state or not st.session_state["weak_topics"]:
        update_conversation_history("assistant", "🎯 No major weaknesses detected! Keep learning and improving.")
        return

    weak_analysis = ["📊 **Weakness Analysis:**\n"]
    topic_counts = {}

    # Track weak topics based on incorrect answers
    for entry in st.session_state["weak_topics"]:
        topic = entry["question"].split(" ")[0]  # Extract the first word as topic
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Identify the most common weak topics
    if topic_counts:
        most_frequent_topic = max(topic_counts, key=topic_counts.get)
        weak_analysis.append(f"🚨 You frequently miss questions related to **{most_frequent_topic}**. Consider studying this area further.")

    # Show specific quiz mistakes
    weak_analysis.append("\n❌ **Questions You Got Wrong:**\n")
    for entry in st.session_state["weak_topics"]:
        weak_analysis.append(f"**Q:** {entry['question']}\n💡 **Correct Answer:** {entry['correct_answer']}\n")

    response_text = "\n".join(weak_analysis)
    update_conversation_history("assistant", response_text)

def handle_general_query(user_input):
    context_str = ""  # Load relevant context if available
    msgs = [
        {"role": "system", "content": "You answer queries using the provided context. If no context is found, use general knowledge."},
        {"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {user_input}"}
    ]
    
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.2)
        answer = r.choices[0].message.content.strip()
        source_note = "**Source:** Extracted from document chunks" if context_str else "**Source:** General Knowledge"
        update_conversation_history("assistant", f"{answer}\n\n{source_note}")
    except Exception as e:
        update_conversation_history("assistant", f"Error: {e}")




def reset_session_state():
    """Reset session variables that store conversation history and other related data."""
    keys_to_reset = [
        "conversation_history", "quiz_mode", "selected_file", 
        "last_bot_question", "last_expected_answer", "pending_response", "weak_topics", "quiz_history"  
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["conversation_history"] = []  # Ensure empty history is initialized
    st.session_state["weak_topics"] = []  # 🆕 Reinitialize empty weaknesses list
    st.session_state["quiz_history"] = [] 

    
    update_conversation_history("assistant", "✅ Chat history, quiz progress, and weaknesses have been cleared. Processed files remain intact.")

# def clear_selected_file():
#     """Clear metadata, images, and stored data of the selected file."""
#     if not st.session_state["selected_file"]:
#         st.warning("No file selected to clear.")
#         return

#     file_to_clear = st.session_state["selected_file"]

#     # Remove from processed files list
#     if file_to_clear in st.session_state["processed_files"]:
#         st.session_state["processed_files"].remove(file_to_clear)

#     # Delete associated metadata file
#     metadata_file = os.path.join("data", f"{file_to_clear}.json")
#     if os.path.exists(metadata_file):
#         os.remove(metadata_file)

#     # Delete images folder associated with the file
#     image_folder = os.path.join("images", file_to_clear)
#     if os.path.exists(image_folder):
#         shutil.rmtree(image_folder)

#     # Delete data folder associated with the file
#     data_folder = os.path.join("data", file_to_clear)
#     if os.path.exists(data_folder):
#         shutil.rmtree(data_folder)

#     # Update the session state and save to disk
#     st.session_state["selected_file"] = None
#     save_processed_files(st.session_state["processed_files"])

#     st.success(f"Cleared all data related to {file_to_clear}.")

def clear_selected_file():
    """Fully remove a selected file's metadata, images, vector data, and ChromaDB stored files."""
    if not st.session_state.get("selected_file"):
        st.warning("No file selected to clear.")
        return

    file_to_clear = st.session_state["selected_file"]
    collection_name = sanitize_collection_name(file_to_clear)
    image_collection_name = f"{collection_name}_images"

    # ✅ Remove from processed files list
    if file_to_clear in st.session_state["processed_files"]:
        st.session_state["processed_files"].remove(file_to_clear)

    # ✅ Delete metadata file (stored JSON file)
    metadata_file = os.path.join("data", f"{file_to_clear}.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    # ✅ Delete stored images
    image_folder = os.path.join("images", file_to_clear)
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    # ✅ Delete stored data (text/vector data)
    data_folder = os.path.join("data", file_to_clear)
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    # ✅ Delete ChromaDB stored data (Persistent Mode)
    chromadb_folder = "chroma_db"  # Change this to the actual ChromaDB storage path
    if os.path.exists(chromadb_folder):
        try:
            shutil.rmtree(chromadb_folder)
            print(f"✅ Successfully deleted ChromaDB folder: {chromadb_folder}")
        except Exception as e:
            print(f"⚠️ Error deleting ChromaDB folder: {e}")

    # ✅ Ensure all data is removed from the vector database
    try:
        # Step 1: Check existing data before deletion
        existing_data = doc_service.get_all_entries(collection_name)
        existing_images = doc_service.get_all_entries(image_collection_name)
        
        if existing_data:
            print(f"📌 Found {len(existing_data)} text entries in {collection_name}. Deleting...")
        if existing_images:
            print(f"📌 Found {len(existing_images)} image metadata entries in {image_collection_name}. Deleting...")

        # Step 2: Delete both text and image metadata collections
        doc_service.delete_collection(collection_name)
        doc_service.delete_collection(image_collection_name)

        # Step 3: Verify if collections are actually deleted
        remaining_text = doc_service.get_all_entries(collection_name)
        remaining_images = doc_service.get_all_entries(image_collection_name)

        if not remaining_text and not remaining_images:
            print(f"✅ Successfully deleted all data from vector store.")
        else:
            print(f"⚠️ Warning: Some data is still present. Manual intervention may be needed.")

    except Exception as e:
        st.error(f"⚠️ Error removing from vector store: {e}")

    # ✅ Clear session state metadata
    keys_to_clear = ["selected_file", "conversation_history", "last_bot_question", "last_expected_answer"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # ✅ Save updated list to disk
    save_processed_files(st.session_state["processed_files"])

    st.success(f"✅ Successfully cleared all data related to {file_to_clear}.")


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
    # Add the "Clear Selected File" button
        if st.button("Clear Selected File", key="clear_selected_file"):
            clear_selected_file()
            st.rerun()

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
