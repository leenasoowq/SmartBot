import os
import pdfplumber
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client

# Initialize SentenceTransformer and FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')  # Pretrained embedding model
index = faiss.IndexFlatL2(384)  # FAISS index for vector storage
knowledge_base = []  # Store text chunks for reference

# Global Variables
quiz_data = []  # Store quiz questions
current_question = 0  # Track the current question
answer_submitted = False  # Prevent skipping ahead before submitting


# Function to preprocess PDF and add to knowledge base
def preprocess_pdf_to_knowledge_base(pdf_path):
    global knowledge_base, index
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_chunks.append(text)

        # Generate embeddings and add to FAISS index
        embeddings = model.encode(text_chunks, convert_to_tensor=False)
        index.add(embeddings)
        knowledge_base = text_chunks  # Save text chunks for retrieval
    except Exception as e:
        print(f"Error processing PDF: {e}")


# Function to retrieve relevant chunks from the knowledge base
def retrieve_relevant_chunks(query, top_k=3):
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(query_embedding, top_k)
    
    results = [knowledge_base[idx] for idx in indices[0] if idx < len(knowledge_base)]

    if any(keyword in query.lower() for keyword in ["quiz", "exam", "test", "mcq", "multiple-choice", "questions"]):
        results = results[:top_k]  # Get only the most relevant chunks for a quiz

    return results


# Function to generate quiz questions dynamically
def generate_quiz_questions_with_rag(query, num_questions=5):
    retrieved_chunks = retrieve_relevant_chunks(query)
    context = "\n\n".join(retrieved_chunks)

    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates multiple-choice quiz questions."},
        {"role": "user", "content": f"""
            Generate **exactly {num_questions}** multiple-choice quiz questions based on the following context.
            
            ### Instructions:
            - Each question must have:
              1. A clear question statement.
              2. Four answer choices labeled A, B, C, and D.
              3. The correct answer labeled as **Correct Answer: X**.
              4. A short explanation labeled as **Explanation: ...**.
            
            Context:
            {context}
        """}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000,  # Increased to allow more content
            temperature=0.7,
        )
        quiz_text = response.choices[0].message.content.strip()

        # Split questions into a list
        questions = quiz_text.split("\n\n")

        parsed_questions = []
        for question in questions:
            if "Correct Answer: " in question and "Explanation:" in question:
                question_text = question.split("Correct Answer: ")[0].strip()
                correct_answer = question.split("Correct Answer: ")[-1].split("Explanation:")[0].strip().replace("**", "").strip()
                explanation = question.split("Explanation:")[-1].strip()
                options = ["A", "B", "C", "D"]
                parsed_questions.append((question_text, options, correct_answer, explanation))
        
        # Return exactly the number of requested questions
        return parsed_questions[:num_questions]

    except Exception as e:
        return [("Error: Failed to generate quiz.", [], "N/A", "No explanation available.")]


# Gradio Chatbot and Quiz Interface
def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Knowledge Base Chatbot with Interactive Quiz Mode")

        chatbot = gr.Chatbot(label="Knowledge Base Bot")

        # Upload Section
        with gr.Row():
            file_upload = gr.File(label="Upload File (PDF or Text)", file_types=[".pdf", ".txt"], visible=True)
            upload_status = gr.Textbox(label="Upload Status", interactive=False)

        def process_upload(file):
            global index, knowledge_base
            index.reset()
            knowledge_base = []
            if file:
                preprocess_pdf_to_knowledge_base(file.name)
            return "Knowledge base loaded successfully!"

        upload_button = gr.Button("Load Knowledge Base")
        upload_button.click(process_upload, inputs=[file_upload], outputs=upload_status)

        # Chat Section
        with gr.Row():
            user_input = gr.Textbox(label="Your Query", placeholder="Ask a question or request a quiz.")
            submit_button = gr.Button("Submit")

        # Quiz Interaction Section
        quiz_question = gr.Textbox(label="Current Question", interactive=False)
        answer_choices = gr.Radio(label="Your Answer", choices=["A", "B", "C", "D"], interactive=True)
        submit_answer_btn = gr.Button("Submit Answer", interactive=True)
        feedback = gr.Textbox(label="Feedback", interactive=False)
        next_btn = gr.Button("Next Question", interactive=False)

        # Function to handle answer submission 
        def submit_answer(user_answer):
            global current_question, quiz_data, answer_submitted
            if current_question < len(quiz_data):
                question_text, options, correct_answer, explanation = quiz_data[current_question]

                # Ensure comparison is case-insensitive and formatted correctly
                is_correct = user_answer.strip().upper() == correct_answer.strip().upper()
                feedback_msg = "✅ Correct!" if is_correct else f"❌ Incorrect. The correct answer is: {correct_answer}\n\n💡 Explanation: {explanation}"

                answer_submitted = True  # Enable next question button
                return feedback_msg, gr.update(interactive=False), gr.update(interactive=True)
            else:
                return "No more questions!", gr.update(interactive=False), gr.update(interactive=False)

        submit_answer_btn.click(submit_answer, inputs=[answer_choices], outputs=[feedback, submit_answer_btn, next_btn])

        # Function to handle quiz progression
        def next_question():
            global current_question, quiz_data, answer_submitted

            if answer_submitted and current_question < len(quiz_data) - 1:
                current_question += 1
                question_text, options, correct_answer, explanation = quiz_data[current_question]
                answer_submitted = False  # Reset for new question
                return question_text, options, "", gr.update(interactive=True), gr.update(interactive=False)
            
            elif len(quiz_data) == 1:  # Stop if only one question exists
                return "No more questions available!", [], "Quiz finished!", gr.update(interactive=False), gr.update(interactive=False)
            
            else:
                return "Quiz completed!", [], "Quiz finished!", gr.update(interactive=False), gr.update(interactive=False)

        next_btn.click(next_question, outputs=[quiz_question, answer_choices, feedback, submit_answer_btn, next_btn])

        def handle_prompt(query):
            retrieved_chunks = retrieve_relevant_chunks(query)
            context = "\n\n".join(retrieved_chunks)

            # Check if the user requested a summary
            if "summarize" in query.lower() or "summarise" in query.lower():
                messages = [
                    {"role": "system", "content": "You are an AI assistant that summarizes documents."},
                    {"role": "user", "content": f"Summarize the following text:\n\n{context}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers questions."},
                    {"role": "user", "content": f"""
                        Based on the following context, answer the query:

                        Context:
                        {context}

                        Query:
                        {query}
                    """}
                ]

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=300,  # Set a reasonable limit for summaries
                    temperature=0.7,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error: {e}"


        # Function to handle user queries with OpenAI
        def chat_response(history, query):
            global quiz_data, current_question, answer_submitted

            if "quiz" in query.lower():
                num_questions = int(query.split(" ")[2]) if query.split(" ")[2].isdigit() else 5
                quiz_data.clear()
                quiz_data.extend(generate_quiz_questions_with_rag(query, num_questions))
                current_question = 0
                answer_submitted = False

                if quiz_data:
                    return history + [(query, "Quiz started!")], quiz_data[0][0], quiz_data[0][1], "", gr.update(interactive=True), gr.update(interactive=False)
                else:
                    return history + [(query, "Failed to generate quiz.")], "No questions available.", [], "", gr.update(interactive=False), gr.update(interactive=False)

            elif "summarize" in query.lower() or "summarise" in query.lower():
                summary = handle_prompt(query)
                return history + [(query, summary)], "", [], "", gr.update(interactive=False), gr.update(interactive=False)

            else:
                response = handle_prompt(query)
                return history + [(query, response)], "", [], "", gr.update(interactive=False), gr.update(interactive=False)


        submit_button.click(chat_response, inputs=[chatbot, user_input], outputs=[chatbot, quiz_question, answer_choices, feedback, submit_answer_btn, next_btn])

    demo.launch()

if __name__ == "__main__":
    main()
