from openai import OpenAI
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("The OPENAI_API_KEY environment variable is not set.")

client = OpenAI(api_key=api_key)

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
    for _ in range(3):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=500,
                temperature=0.7,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print("Rate limit exceeded. Retrying in 60 seconds...")
                time.sleep(60)
            else:
                print(f"Error generating questions: {e}")
                return "Error: Failed to generate quiz questions."
    return "Error: Failed to generate quiz questions after retries."
