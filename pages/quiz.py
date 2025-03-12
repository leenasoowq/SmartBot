import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import random

from services.document_service import DocumentService
from services.quiz_service import QuizService

# Constants
LABEL_LETTERS = ["A", "B", "C", "D"]
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard"]
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

doc_service = DocumentService(client)
quiz_service = QuizService(client)

# Session state for quiz
if "quiz_step" not in st.session_state:
    st.session_state["quiz_step"] = "select_options"
if "num_questions" not in st.session_state:
    st.session_state["num_questions"] = 5
if "difficulty" not in st.session_state:
    st.session_state["difficulty"] = "Medium"
if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = []
if "current_q_index" not in st.session_state:
    st.session_state["current_q_index"] = 0
if "correct_count" not in st.session_state:
    st.session_state["correct_count"] = 0
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = []

def reset_quiz_state():
    """Reset the quiz state to initial values."""
    st.session_state.update({
        "quiz_step": "select_options",
        "quiz_data": [],
        "current_q_index": 0,
        "correct_count": 0,
    })

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
    """Process uploaded PDF files and store them in the vector store with collections."""
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
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        docs = doc_service.process_pdf(file_path)
        collection_name = sanitize_collection_name(file.name)
        doc_service.add_documents_to_vectorstore(docs, collection_name)
        st.session_state["processed_files"].append(file.name)
        processed_count += 1
        os.remove(file_path)
    return f"Processed {processed_count} new file(s) successfully!"

def load_context_for_file(file_name, top_k=10):
    """Retrieve relevant chunks from the vector store for a given file."""
    collection_name = sanitize_collection_name(file_name)
    print(f"Loading context from collection: {collection_name}")
    query = "key concepts, events, themes, or examples from the document"
    print(f"Retrieving chunks for query: {query}")
    chunks = doc_service.retrieve_relevant_chunks(query, collection_name, top_k=top_k)
    print(f"Retrieved {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: {chunk[:500]}")
    return "\n\n".join(chunks) if chunks else "No relevant chunks found."

def handle_answer_submission(q_index, labeled_options, correct_letter):
    """Handle user answer submission and update score."""
    user_letter_idx = labeled_options.index(st.session_state[f"quiz_q_{q_index}"])
    user_letter_str = LABEL_LETTERS[user_letter_idx]
    if user_letter_str == correct_letter:
        st.success("Correct!")
        if f"scored_{q_index}" not in st.session_state:
            st.session_state[f"scored_{q_index}"] = True
            st.session_state["correct_count"] += 1
    else:
        st.error(f"Incorrect. The correct answer is {correct_letter}.")

st.title("📝 Quiz Generator")

if st.session_state["quiz_step"] == "select_options":
    st.session_state["num_questions"] = st.slider("Number of questions", MIN_QUESTIONS, MAX_QUESTIONS, 5)
    st.session_state["difficulty"] = st.radio("Difficulty level", DIFFICULTY_LEVELS)
    
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        msg = preprocess_files(uploaded_files)
        st.success(msg)

    if st.session_state["processed_files"]:
        selected_file = st.selectbox("Select a file for quiz generation", st.session_state["processed_files"])
        
        if st.button("Generate Quiz"):
            reset_quiz_state()
            context_text = load_context_for_file(selected_file)
            if "Error" in context_text:
                st.error(context_text)
            else:
                try:
                    quiz_data = quiz_service.generate_quiz_questions(
                        context=context_text,
                        difficulty=st.session_state["difficulty"],
                        num_questions=st.session_state["num_questions"]
                    )
                    st.write("Quiz Data Generated:", quiz_data)  # Debugging output
                    if not quiz_data:
                        st.error("No quiz questions could be generated.")
                    else:
                        st.session_state["quiz_data"] = quiz_data
                        st.session_state["quiz_step"] = "in_progress"
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate quiz questions: {e}")
                    reset_quiz_state()
                    st.rerun()
    else:
        st.warning("No processed files found. Please upload a PDF first.")

elif st.session_state["quiz_step"] == "in_progress":
    q_index = st.session_state["current_q_index"]
    quiz_data = st.session_state["quiz_data"]

    if not quiz_data:
        st.write("No quiz questions available. Returning to main menu...")
        st.session_state["quiz_step"] = "select_options"
        st.rerun()

    if q_index < len(quiz_data):
        question_text, shuffled_options, correct_letter, explanation, confidence = quiz_data[q_index]
        
        st.subheader(f"Question {q_index + 1} of {len(quiz_data)}")
        st.write(question_text)
        
        labeled_options = [f"{letter}) {opt}" for letter, opt in zip(LABEL_LETTERS, shuffled_options)]
        chosen_option = st.radio("Select your answer:", labeled_options, key=f"quiz_q_{q_index}")
        
        if st.button("Submit Answer"):
            handle_answer_submission(q_index, labeled_options, correct_letter)
            st.info(f"**Explanation:** {explanation}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")

        if st.button("Next Question"):
            st.session_state["current_q_index"] += 1
            if st.session_state["current_q_index"] >= len(quiz_data):
                st.session_state["quiz_step"] = "score_report"
            st.rerun()

elif st.session_state["quiz_step"] == "score_report":
    total_questions = len(st.session_state["quiz_data"])
    correct = st.session_state["correct_count"]
    st.header("Quiz Finished!")
    st.subheader(f"Your score: {correct} / {total_questions}")

    if st.button("Back to Quiz Menu"):
        st.session_state["quiz_step"] = "select_options"
        st.rerun()
