import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)  

# Initialise session state
if "quiz_step" not in st.session_state:
    st.session_state.quiz_step = "select_options"
if "selected_difficulty" not in st.session_state:
    st.session_state.selected_difficulty = "Medium"
if "num_questions" not in st.session_state:
    st.session_state.num_questions = 5
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "selected_answer" not in st.session_state:
    st.session_state.selected_answer = None
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "detailed_explanation" not in st.session_state:
    st.session_state.detailed_explanation = ""
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "selected_file" not in st.session_state:
    st.session_state.selected_file = None

# Retrieve knowledge from ChromaDB before generating quiz
def retrieve_relevant_chunks(file_name, top_k=10):
    """Fetch relevant information from the selected file stored in ChromaDB."""
    try:
        results = vectorstore.similarity_search(file_name, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving documents: {e}"]

# Generate quiz questions
def generate_quiz_questions(file_name, difficulty, num_questions=5):
    retrieved_chunks = retrieve_relevant_chunks(file_name)
    print(f"Retrieved Chunks from ChromaDB for {file_name}: {retrieved_chunks}")

    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": f"You are an academic quiz assistant generating {difficulty}-level questions based on provided knowledge."},
        {"role": "user", "content": f"""
            Generate exactly {num_questions} {difficulty}-level multiple-choice quiz questions using the provided knowledge.

            ### Knowledge Context:
            {context}

            ### Format:
            Question: <question_text>
            A) <option_1>
            B) <option_2>
            C) <option_3>
            D) <option_4>
            Correct Answer: <correct_option_letter>
            Explanation: <explanation_text>
            Explanation Confidence Score: <confidence_score>

            ### Instructions:
            - The explanation should be detailed and directly reference the provided context.
            - The confidence score should reflect how well the explanation aligns with the context.
        """}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            temperature=0.3,
        )
        quiz_text = response.choices[0].message.content.strip()

        print("Quiz Response:\n", quiz_text)

        questions = quiz_text.split("\n\n")
        parsed_questions = []

        for question in questions:
            if "Correct Answer:" in question and "Explanation:" in question:
                try:
                    lines = question.split("\n")
                    question_text = lines[0].replace("Question: ", "").strip()
                    options = [line.split(") ")[1].strip() for line in lines[1:5]]

                    correct_answer = lines[5].replace("Correct Answer: ", "").strip().upper()
                    explanation = lines[6].replace("Explanation: ", "").strip()
                    # Calculate confidence score
                    explanation_confidence_score = calculate_explanation_confidence(explanation, context)
                    # Avoid bias answer
                    shuffled_options = options[:]
                    random.shuffle(shuffled_options)

                    correct_index = shuffled_options.index(options[["A", "B", "C", "D"].index(correct_answer)])
                    correct_answer_letter = ["A", "B", "C", "D"][correct_index]

                    parsed_questions.append((question_text, shuffled_options, correct_answer_letter, explanation, explanation_confidence_score))
                except Exception as e:
                    print("Error processing question:", e)
                    continue  # Skip problematic questions

        # Ensure we got valid questions
        if len(parsed_questions) < num_questions:
            return [("Error: Failed to generate valid quiz questions.", [], "N/A", "No explanation available.", 0.0)]

        return parsed_questions[:num_questions]

    except Exception as e:
        print("Quiz generation error:", e)
        return [("Error: Failed to generate quiz.", [], "N/A", "No explanation available.", 0.0)]

def calculate_explanation_confidence(explanation, context):
    explanation_embedding = embedding_model.embed_query(explanation)
    context_embedding = embedding_model.embed_query(context)
    
    similarity = cosine_similarity([explanation_embedding], [context_embedding])[0][0]
    
    # Convert similarity to a confidence score (0-100)
    confidence_score = min(100.0, max(0.0, (similarity + 1) * 50))  # Scale from -1 to 1 to 0-100
    return confidence_score

# Quiz mode UI
st.title("🎓 Interactive Quiz Mode")

# Sidebar for file selection and upload
st.sidebar.title("File Selection")

# Use the processed files from app.py if available
if "processed_files" in st.session_state:
    st.session_state.uploaded_files = st.session_state.processed_files

# Allow user to upload additional files
additional_files = st.sidebar.file_uploader("Upload additional files", accept_multiple_files=True, type=["pdf", "mp3", "wav", "mp4", "avi"])
if additional_files:
    st.session_state.uploaded_files.extend([file.name for file in additional_files])

# Select file for quiz generation
if st.session_state.uploaded_files:
    st.session_state.selected_file = st.sidebar.selectbox("Select file for quiz generation", st.session_state.uploaded_files)

if st.session_state.quiz_step == "select_options":
    st.subheader("Select Quiz Options")
    st.session_state.num_questions = st.slider("How many questions?", 1, 10, 5)
    st.session_state.selected_difficulty = st.radio("Select difficulty:", ["Easy", "Medium", "Hard"])

    if st.button("Start Quiz"):
        if st.session_state.selected_file:
            st.session_state.quiz_data = generate_quiz_questions(st.session_state.selected_file, st.session_state.selected_difficulty, st.session_state.num_questions)
            st.session_state.quiz_step = "in_progress"
            st.session_state.current_question = 0
            st.session_state.feedback = ""
            st.session_state.detailed_explanation = ""
            st.rerun()
        else:
            st.error("Please select a file to generate the quiz.")

elif st.session_state.quiz_step == "in_progress":
    st.subheader(f"Question {st.session_state.current_question + 1} of {st.session_state.num_questions}")
    question, options, correct_answer, explanation, explanation_confidence_score = st.session_state.quiz_data[st.session_state.current_question]
    st.write(f"**{question}**")

    option_labels = ["A", "B", "C", "D"]
    labeled_options = [f"{label}) {option}" for label, option in zip(option_labels, options)]
    selected_answer = st.radio("Select your answer:", labeled_options, key=f"question_{st.session_state.current_question}")

    if st.button("Submit Answer"):
        # Convert selected answer to corresponding letter 
        selected_letter = selected_answer.split(") ")[0]

        st.session_state.selected_answer = selected_letter
        if st.session_state.selected_answer == correct_answer:
            st.session_state.feedback = "Correct!"
            st.session_state.detailed_explanation = f"Well done! {correct_answer} is the right choice because: {explanation}\n\n**Explanation Confidence Score:** {explanation_confidence_score:.2f}%"
        else:
            st.session_state.feedback = f"Incorrect. The correct answer is: {correct_answer}."
            st.session_state.detailed_explanation = f"Your choice ({selected_answer}) is incorrect because it does not align with the key concept. The correct answer ({correct_answer}) is correct because: {explanation}\n\n**Explanation Confidence Score:** {explanation_confidence_score:.2f}%"
        st.rerun()

    if st.session_state.feedback:
        st.write(st.session_state.feedback)
        st.write(st.session_state.detailed_explanation)
        if st.session_state.current_question < st.session_state.num_questions - 1:
            if st.button("Next Question"):
                st.session_state.current_question += 1
                st.session_state.feedback = ""
                st.session_state.detailed_explanation = ""
                st.rerun()

        if st.button("Exit Quiz"):
            st.session_state.quiz_step = "select_options"
            st.rerun()