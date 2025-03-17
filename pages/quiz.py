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
    st.session_state["quiz_step"] = "select_options" # Controls UI state (e.g., selecting options, taking quiz, or showing results)
if "num_questions" not in st.session_state:
    st.session_state["num_questions"] = 5 # Default number of questions
if "difficulty" not in st.session_state:
    st.session_state["difficulty"] = "Medium" # Default quiz difficulty
if "quiz_data" not in st.session_state:
    st.session_state["quiz_data"] = [] # Stores generated quiz questions
if "current_q_index" not in st.session_state:
    st.session_state["current_q_index"] = 0 # Tracks current quiz question index
if "correct_count" not in st.session_state:
    st.session_state["correct_count"] = 0 # Stores count of correct answers
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = [] # Stores uploaded & processed PDFs

# Resets the quiz state, clearing previous questions and answers.
def reset_quiz_state():
    """Reset the quiz state to initial values."""
    st.session_state.update({
        "quiz_step": "select_options",
        "quiz_data": [],
        "current_q_index": 0,
        "correct_count": 0,
    })

# Sanitizes file names for ChromaDB collection storage (alphanumeric, underscores, hyphens).
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

# Processes uploaded PDF files, extracts text, and stores them in the vector database.
def preprocess_files(files):
    """Process uploaded PDF files and store them in the vector store with collections."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Identify new files that haven't been processed yet
    new_files = [f.name for f in files if f.name not in st.session_state["processed_files"]]
    if not new_files:
        return "No new files to process."

    processed_count = 0
    for file in files:
        if file.name not in new_files:
            continue
        
        # Save the uploaded file
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        # Extract text from the PDF
        docs = doc_service.process_pdf(file_path)
        # Convert file name into a valid ChromaDB collection name
        collection_name = sanitize_collection_name(file.name)
        # Store extracted document chunks into the vector database
        doc_service.add_documents_to_vectorstore(docs, collection_name)
        # Mark the file as processed
        st.session_state["processed_files"].append(file.name)
        processed_count += 1
        # Remove the file after processing to save storage
        os.remove(file_path)
    return f"Processed {processed_count} new file(s) successfully!"

# Retrieves relevant document content from ChromaDB for quiz generation.
def load_context_for_file(file_name, top_k=10):
    """Retrieve relevant content from a document for quiz generation."""
    collection_name = sanitize_collection_name(file_name)
    print(f"Loading context from collection: {collection_name}")

    # Query for broad content extraction (ensures variety in quiz questions)
    query = "Summarize key concepts, definitions, and important details from this document."

    chunks = doc_service.retrieve_relevant_chunks(query, collection_name, top_k=top_k)

    if not chunks:
        print(f"ERROR: No relevant chunks found for {file_name}")
        return "No relevant chunks found."

    print(f"Retrieved {len(chunks)} chunks for quiz generation.")
    
    # Debugging: Print first 3 chunks to check retrieved content
    for i, chunk in enumerate(chunks[:3]):
        print(f"Chunk {i+1}: {chunk[:300]}")  # Print first 300 characters for preview

    return "\n\n".join(chunks)

# Handles user answer submission, updates the score, and provides feedback.
def handle_answer_submission(q_index, labeled_options, correct_letter):
    """Handle user answer submission and update score."""
    # Get the user's selected answer (by index)
    user_letter_idx = labeled_options.index(st.session_state[f"quiz_q_{q_index}"])
    user_letter_str = LABEL_LETTERS[user_letter_idx]
    if user_letter_str == correct_letter:
        st.success("Correct!")
        # Ensure the question is only scored once
        if f"scored_{q_index}" not in st.session_state:
            st.session_state[f"scored_{q_index}"] = True
            st.session_state["correct_count"] += 1
    else:
        st.error(f"Incorrect. The correct answer is {correct_letter}.")

st.title("📝 Quiz Generator")

if st.session_state["quiz_step"] == "select_options":
    st.session_state["num_questions"] = st.slider("Number of questions", MIN_QUESTIONS, MAX_QUESTIONS, 5) # User selects the number of questions for the quiz
    st.session_state["difficulty"] = st.radio("Difficulty level", DIFFICULTY_LEVELS) # User selects quiz difficulty level
    # File uploader allows users to upload PDFs for quiz generation
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    # If files are uploaded, process them
    if uploaded_files:
        msg = preprocess_files(uploaded_files)
        st.success(msg)
        
    # If files have been processed, allow the user to select one for quiz generation
    if st.session_state["processed_files"]:
        selected_file = st.selectbox("Select a file for quiz generation", st.session_state["processed_files"])
        # When user clicks "Generate Quiz"
        if st.button("Generate Quiz"):
            reset_quiz_state()
            # Retrieve relevant content from the selected document
            context_text = load_context_for_file(selected_file)
            if "Error" in context_text:
                st.error(context_text)
            else:
                try:
                    with st.spinner("Generating quiz... Please wait! 🕒"):
                        # Generate quiz questions using retrieved document content
                        quiz_data = quiz_service.generate_quiz_questions(
                            context=context_text,
                            difficulty=st.session_state["difficulty"],
                            num_questions=st.session_state["num_questions"]
                        )
                    st.write("Quiz Data Generated:", quiz_data)  # Debugging output
                    # If no quiz questions were generated, show an error message
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
# Quiz In Progress - Display Questions and Accept Answers
elif st.session_state["quiz_step"] == "in_progress":
    # Get the current question index
    q_index = st.session_state["current_q_index"]
    quiz_data = st.session_state["quiz_data"]
    # If no quiz data is available, return to main menu
    if not quiz_data:
        st.write("No quiz questions available. Returning to main menu...")
        st.session_state["quiz_step"] = "select_options"
        st.rerun()
    # If the quiz is still in progress, display the current question
    if q_index < len(quiz_data):
        question_text, shuffled_options, correct_letter, explanation, confidence = quiz_data[q_index]
        # Show question number and text
        st.subheader(f"Question {q_index + 1} of {len(quiz_data)}")
        st.write(question_text)
        # Display answer choices with labels (A, B, C, D)
        labeled_options = [f"{letter}) {opt}" for letter, opt in zip(LABEL_LETTERS, shuffled_options)]
        chosen_option = st.radio("Select your answer:", labeled_options, key=f"quiz_q_{q_index}")
        # When user submits an answer, evaluate it
        if st.button("Submit Answer"):
            handle_answer_submission(q_index, labeled_options, correct_letter)
            st.info(f"**Explanation:** {explanation}")
            st.write(f"**Confidence Score:** {confidence:.2f}%")
         # If "Next Question" is clicked, move to the next question
        if st.button("Next Question"):
            st.session_state["current_q_index"] += 1
            # If last question is answered, go to score report
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
