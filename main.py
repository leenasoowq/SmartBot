import gradio as gr
from processing import extract_text_from_pdf, extract_text_from_txt, extract_text_from_url
from generate_quiz import generate_quiz_questions
from utils import initialize_database, store_content, retrieve_content

# initialize_database()  

def process_input(input_type, input_data, url_input, num_questions, difficulty):
    text = ""

    if input_type == "PDF File":
        text = extract_text_from_pdf(input_data)
    elif input_type == "Text File":
        text = extract_text_from_txt(input_data)
    elif input_type == "URL":
        text = extract_text_from_url(url_input)

    if not text.strip():
        return "Error: No text could be extracted or provided."

    quiz_text = generate_quiz_questions(text, num_questions, difficulty)

    if "Error:" in quiz_text:
        return quiz_text

    return quiz_text

def main():
    with gr.Blocks() as demo:
        gr.Markdown("## Quiz Generator")

        input_type = gr.Radio(["PDF File", "Text File", "URL"], label="Select Input Type")
        input_data = gr.File(label="Upload File", file_types=[".pdf", ".txt"], visible=False)
        url_input = gr.Textbox(label="Enter URL", visible=False)
        num_questions = gr.Number(label="Number of Questions", value=5, precision=0)
        difficulty = gr.Radio(["easy", "medium", "hard"], label="Select Difficulty", value="medium")
        result = gr.Textbox(label="Generated Quiz", lines=10)

        def toggle_visibility(choice):
            return gr.update(visible=choice in ["PDF File", "Text File"]), gr.update(visible=choice == "URL")

        input_type.change(toggle_visibility, input_type, [input_data, url_input])

        generate_button = gr.Button("Generate Quiz")
        generate_button.click(
            process_input,
            inputs=[input_type, input_data, url_input, num_questions, difficulty],
            outputs=result,
        )

        demo.launch()

if __name__ == "__main__":
    main()
