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

def reset_quiz_state():
    """Reset the quiz state to initial values."""
    st.session_state.update({
        "quiz_step": "select_options",
        "quiz_data": [],  # Clear existing quiz data
        "current_q_index": 0,  # Reset question index
        "correct_count": 0,  # Reset score
    })
    
def load_context_for_file(file_name, top_k=10):
    """Retrieve relevant chunks from the vector store for a given file."""
    chunks = doc_service.retrieve_relevant_chunks(file_name, top_k=top_k)
    return "\n\n".join(chunks)


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
    
    if "processed_files" in st.session_state and st.session_state["processed_files"]:
        selected_file = st.selectbox("Select a file for quiz generation", st.session_state["processed_files"])
        
        if st.button("Generate Quiz"):
            # Reset the user score
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
                        st.error("No quiz questions could be generated. Please try a different file or adjust the settings.")
                    else:
                        # Update session state with new quiz data
                        st.session_state["quiz_data"] = quiz_data
                        st.session_state["quiz_step"] = "in_progress"
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to generate quiz questions: {e}")
                    reset_quiz_state()
                    st.rerun()
    else:
        st.warning("No processed files found. Please go to the main page and upload/prepare files first.")

# The quiz is in progress
elif st.session_state["quiz_step"] == "in_progress":
    q_index = st.session_state["current_q_index"]
    quiz_data = st.session_state["quiz_data"]

    # Ensure we have valid quiz_data
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

# show final score
elif st.session_state["quiz_step"] == "score_report":
    total_questions = len(st.session_state["quiz_data"])
    correct = st.session_state["correct_count"]
    st.header("Quiz Finished!")
    st.subheader(f"Your score: {correct} / {total_questions}")

    if st.button("Back to Quiz Menu"):
        st.session_state["quiz_step"] = "select_options"
        st.rerun()
