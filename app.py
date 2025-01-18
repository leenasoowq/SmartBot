import os
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import pdfplumber
from dotenv import load_dotenv
import gradio as gr

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
# Function to generate quiz questions using GPT API
def generate_quiz_questions(text, num_questions=5):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that generates quiz questions."},
        {"role": "user", "content": f"""
            Create {num_questions} multiple-choice quiz questions based on the following text. 
            Each question should have:
            1. A clearly stated question.
            2. Four answer choices labeled A, B, C, and D.
            3. A correct answer labeled as 'Correct Answer'.
            
            Text:
            {text}
        """}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
            max_tokens=500,
            temperature=0.7,
        )
        # Access response content correctly
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating questions: {e}")
        return "Error: Failed to generate quiz questions."


# Function to handle input and generate quiz
def process_input(input_type, input_data, num_questions):
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

    questions = generate_quiz_questions(text, num_questions)

    return questions if questions else "Error: Failed to generate quiz questions."

# Set up Gradio interface
def main():
    def interface(input_type, input_data, num_questions):
        if input_type in ["PDF File", "Text File"]:
            return process_input(input_type, input_data, num_questions)
        elif input_type == "URL":
            return process_input(input_type, input_data, num_questions)
        else:
            return "Error: Invalid input type."

    with gr.Blocks() as demo:
        gr.Markdown("## Quiz Generator")

        input_type = gr.Radio(["PDF File", "Text File", "URL"], label="Select Input Type")

        input_data = gr.File(label="Upload File (for PDF or Text File)", file_types=[".pdf", ".txt"], visible=False)
        url_input = gr.Textbox(label="Enter URL", visible=False)

        num_questions = gr.Number(label="Number of Questions", value=5, precision=0)

        result = gr.Textbox(label="Generated Quiz Questions", lines=10)

        def toggle_input_visibility(choice):
            return gr.update(visible=choice in ["PDF File", "Text File"]), gr.update(visible=choice == "URL")

        input_type.change(toggle_input_visibility, input_type, [input_data, url_input])

        generate_button = gr.Button("Generate Quiz")
        generate_button.click(interface, [input_type, input_data, num_questions], result)

    demo.launch()

if __name__ == "__main__":
    main()