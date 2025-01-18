import os
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv
import gradio as gr
import time

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client

# Function to extract text from a PDF file
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    return text

# Function to extract text from a plain text file
def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""

# Function to extract text from a URL
def extract_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text(separator="\n")
    except Exception as e:
        print(f"Error fetching URL content: {e}")
        return ""

# Function to generate quiz questions using GPT API
def generate_quiz_questions(text, num_questions=5, difficulty="medium"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates quiz questions."},
        {"role": "user", "content": f"""
            Create {num_questions} multiple-choice quiz questions based on the following text.
            The difficulty level should be '{difficulty}'.
            Each question should have:
            1. A clearly stated question.
            2. Four answer choices labeled A, B, C, and D.
            3. A correct answer labeled as 'Correct Answer'.
            
            Text:
            {text}
        """}
    ]
    for _ in range(3):  # Retry up to 3 times
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0,
            )
            # Access response content correctly
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print("Rate limit exceeded. Retrying in 60 seconds...")
                time.sleep(60)  # Wait for 60 seconds before retrying
            else:
                print(f"Error generating questions: {e}")
                return "Error: Failed to generate quiz questions."
    return "Error: Failed to generate quiz questions due to repeated rate limit issues."

# Parse generated quiz into questions, options, and correct answers
def parse_quiz(quiz_text):
    questions = []
    quiz_items = quiz_text.split("\n\n")
    for item in quiz_items:
        if "**Correct Answer:**" in item:
            question_part, correct_answer = item.split("**Correct Answer:**")
            question_lines = question_part.strip().split("\n")
            question = question_lines[0]
            options = question_lines[1:]
            correct_answer = correct_answer.strip()
            questions.append({"question": question, "options": options, "correct": correct_answer})
    return questions

# Function to handle input and generate quiz
def process_input(input_type, input_data, num_questions, difficulty):
    text = ""

    if input_type == "PDF File":
        text = extract_text_from_pdf(input_data)
    elif input_type == "Text File":
        text = extract_text_from_txt(input_data)
    elif input_type == "URL":
        text = extract_text_from_url(input_data)

    if not text.strip():
        return "Error: No text could be extracted or provided."

    try:
        num_questions = int(num_questions)
    except ValueError:
        return "Error: Please enter a valid number of questions."

    quiz_text = generate_quiz_questions(text, num_questions, difficulty)

    if "Error:" in quiz_text:
        return quiz_text

    return parse_quiz(quiz_text)

# Set up Gradio interface
def main():
    def interface(input_type, input_data, url_input, num_questions, difficulty):
        input_value = input_data if input_type in ["PDF File", "Text File"] else url_input
        quiz_data = process_input(input_type, input_value, num_questions, difficulty)
        if isinstance(quiz_data, str):
            return quiz_data, None  # Return error message
        return "Quiz Generated!", quiz_data

    def check_answer(user_answer, correct_answer):
        if user_answer == correct_answer:
            return "Correct!"
        else:
            return f"Incorrect! The correct answer is: {correct_answer}"

    with gr.Blocks() as demo:
        gr.Markdown("## Quiz Generator")

        input_type = gr.Radio(["PDF File", "Text File", "URL"], label="Select Input Type")

        input_data = gr.File(label="Upload File (for PDF or Text File)", file_types=[".pdf", ".txt"], visible=False)
        url_input = gr.Textbox(label="Enter URL", visible=False)

        num_questions = gr.Number(label="Number of Questions", value=5, precision=0)

        difficulty = gr.Radio(
            ["easy", "medium", "hard"], label="Select Difficulty Level", value="easy"
        )

        quiz_status = gr.Textbox(label="Quiz Status", interactive=False)
        question_box = gr.Textbox(label="Question", interactive=False, visible=False)
        options_box = gr.Radio(choices=[], label="Options", visible=False)
        feedback_box = gr.Textbox(label="Feedback", interactive=False, visible=False)
        next_button = gr.Button("Next Question", visible=False)

        quiz_data = []
        question_index = 0

        def toggle_input_visibility(choice):
            # Toggle visibility based on selected input type
            return gr.update(visible=choice in ["PDF File", "Text File"]), gr.update(visible=choice == "URL")

        input_type.change(toggle_input_visibility, input_type, [input_data, url_input])

        def load_quiz(input_type, input_data, url_input, num_questions, difficulty):
            nonlocal quiz_data, question_index
            quiz_status, quiz_data = interface(input_type, input_data, url_input, num_questions, difficulty)
            question_index = 0
            if isinstance(quiz_data, list) and quiz_data:
                question = quiz_data[0]["question"]
                options = quiz_data[0]["options"]
                return (
                    quiz_status,
                    question,
                    gr.update(choices=options, visible=True),  # Populate options dynamically
                    gr.update(visible=True),  # Show feedback
                    gr.update(visible=True),  # Show next button
                )
            return (
                quiz_status,
                "",
                gr.update(choices=[], visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )


        def handle_next_question(selected_option):
            nonlocal question_index
            if question_index < len(quiz_data):
                correct_answer = quiz_data[question_index]["correct"]
                feedback = check_answer(selected_option, correct_answer)
                question_index += 1
                if question_index < len(quiz_data):
                    next_question = quiz_data[question_index]["question"]
                    next_options = quiz_data[question_index]["options"]
                    return feedback, next_question, gr.update(choices=next_options, visible=True)
                else:
                    return feedback, "Quiz Completed!", gr.update(choices=[], visible=False)
            return "", "", gr.update(choices=[], visible=False)

        generate_button = gr.Button("Generate Quiz")
        generate_button.click(
            load_quiz,
            [input_type, input_data, url_input, num_questions, difficulty],
            [quiz_status, question_box, options_box, feedback_box, next_button],
        )

        options_box.change(handle_next_question, inputs=[options_box], outputs=[feedback_box, question_box, options_box])

    demo.launch()

if __name__ == "__main__":
    main()