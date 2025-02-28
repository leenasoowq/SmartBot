import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
import os
import random
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

# Retrieve knowledge from ChromaDB before generating quiz
def retrieve_relevant_chunks(query, top_k=3):
    """Fetch relevant information from ChromaDB before generating quiz questions."""
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving documents: {e}"]

# Generate quiz questions
def generate_quiz_questions(query, difficulty, num_questions=5):
    retrieved_chunks = retrieve_relevant_chunks(query)
    print(f"Retrieved Chunks from ChromaDB: {retrieved_chunks}")

    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": f"You are an academic quiz assistant generating {difficulty}-level multiple-choice quiz questions based on provided knowledge."},
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
        """}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1500,
            temperature=0.7,
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

                    # Avoid bias answer
                    shuffled_options = options[:]
                    random.shuffle(shuffled_options)

                    correct_index = shuffled_options.index(options[["A", "B", "C", "D"].index(correct_answer)])
                    correct_answer_letter = ["A", "B", "C", "D"][correct_index]

                    parsed_questions.append((question_text, shuffled_options, correct_answer_letter, explanation))

                except Exception as e:
                    print("Error processing question:", e)
                    continue  # Skip problematic questions

        # Ensure we got valid questions
        if len(parsed_questions) < num_questions:
            return [("Error: Failed to generate valid quiz questions.", [], "N/A", "No explanation available.")]

        return parsed_questions[:num_questions]

    except Exception as e:
        print("Quiz generation error:", e)
        return [("Error: Failed to generate quiz.", [], "N/A", "No explanation available.")]
    
# Quiz mode UI
st.title("🎓 Interactive Quiz Mode")

if st.session_state.quiz_step == "select_options":
    st.subheader("🎯 Select Quiz Options")
    st.session_state.num_questions = st.slider("How many questions?", 1, 10, 5)
    st.session_state.selected_difficulty = st.radio("Select difficulty:", ["Easy", "Medium", "Hard"])

    if st.button("Start Quiz"):
        st.session_state.quiz_data = generate_quiz_questions("firewall", st.session_state.selected_difficulty, st.session_state.num_questions)
        st.session_state.quiz_step = "in_progress"
        st.session_state.current_question = 0
        st.session_state.feedback = ""
        st.rerun()

elif st.session_state.quiz_step == "in_progress":
    st.subheader(f"Question {st.session_state.current_question + 1} of {st.session_state.num_questions}")
    question, options, correct_answer, explanation = st.session_state.quiz_data[st.session_state.current_question]
    st.write(f"**{question}**")

    selected_answer = st.radio("Select your answer:", options, key=f"question_{st.session_state.current_question}")

    if st.button("Submit Answer"):
        # Convert selected answer to corresponding letter 
        selected_index = options.index(selected_answer)
        selected_letter = ["A", "B", "C", "D"][selected_index]

        st.session_state.selected_answer = selected_letter
        if st.session_state.selected_answer == correct_answer:
            st.session_state.feedback = "Correct!"
        else:
            st.session_state.feedback = f"Incorrect. The correct answer is: {correct_answer}\nExplanation: {explanation}"
        st.rerun()

    if st.session_state.feedback:
        st.write(st.session_state.feedback)
        if st.session_state.current_question < st.session_state.num_questions - 1:
            if st.button("Next Question"):
                st.session_state.current_question += 1
                st.session_state.feedback = ""
                st.rerun()

        if st.button("Exit Quiz"):
            st.session_state.quiz_step = "select_options"
            st.rerun()
