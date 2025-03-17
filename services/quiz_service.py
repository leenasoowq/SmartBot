import random
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings

class QuizService:
    # Initializes the quiz service with an OpenAI client and an embedding model for similarity scoring.
    def __init__(self, client: OpenAI, embedding_model_name: str = "text-embedding-3-large"):
        self.client = client
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    # Generates multiple-choice quiz questions based on the provided context and difficulty level.
    def generate_quiz_questions(self, context: str, difficulty: str = "Medium", num_questions: int = 5) -> list:
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an academic quiz assistant generating {difficulty}-level multiple-choice questions "
                    f"based solely on the provided context. Analyze the context to identify its primary subject matter "
                    f"(e.g., programming, history), then create specific, detailed questions about that subject alone. "
                    f"Use exact details from the context (e.g., terms, examples) and avoid introducing topics not present. "
                    f"Ensure variety: factual (e.g., 'What is X?'), conceptual (e.g., 'What does Y mean?'), and analytical "
                    f"(e.g., 'Why is Z used?'). Ignore metadata unless part of the subject."
                ),
            },
            {
                "role": "user",
                "content": f"""
Please generate exactly {num_questions} {difficulty}-level multiple-choice questions using only the context below. 
Analyze the context to determine its main subject, then create specific, engaging questions directly tied to it using 
exact details. Ensure a mix of factual, conceptual, and analytical questions, and avoid vague or external topics.

### Context:
{context}

### Format:
Question: <question_text>
A) <option_1>
B) <option_2>
C) <option_3>
D) <option_4>
Correct Answer: <letter>
Explanation: <explanation_text>
"""
            }
        ]

        try:
            # Send request to GPT model to generate quiz questions
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1500,
                temperature=0.5,
            )
            # Retrieve and clean the response
            quiz_text = response.choices[0].message.content.strip()
            print(f"GPT Response:\n{quiz_text}")  # Debug output
            # Split the response into separate questions
            question_blocks = quiz_text.split("\n\n")
            parsed_questions = []
            # Process each question block
            for block in question_blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                # Ensure the block contains a valid question, answer choices, and explanation
                if ("Correct Answer:" in block) and ("Explanation:" in block) and lines:
                    try:
                        question_line = lines[0].replace("Question: ", "").strip()
                        option_a = lines[1].split(") ", 1)[1].strip()
                        option_b = lines[2].split(") ", 1)[1].strip()
                        option_c = lines[3].split(") ", 1)[1].strip()
                        option_d = lines[4].split(") ", 1)[1].strip()
                        # Extract correct answer letter and explanation
                        correct_letter_line = lines[5].replace("Correct Answer:", "").strip().upper()
                        explanation_line = lines[6].replace("Explanation:", "").strip()
                        # Prepare options and shuffle them randomly
                        options = [option_a, option_b, option_c, option_d]
                        shuffled = options[:]
                        random.shuffle(shuffled)
                        # Adjust the correct answer position after shuffling
                        original_index = ["A", "B", "C", "D"].index(correct_letter_line)
                        correct_option_text = options[original_index]
                        new_correct_index = shuffled.index(correct_option_text)
                        new_correct_letter = ["A", "B", "C", "D"][new_correct_index]
                        # Compute confidence score based on similarity between explanation and context
                        confidence_score = self._compute_confidence(explanation_line, context)
                        # Store the parsed question data
                        parsed_questions.append((
                            question_line,
                            shuffled,
                            new_correct_letter,
                            explanation_line,
                            confidence_score
                        ))
                    except Exception:
                        continue
            # Return the parsed questions, limited to the requested number
            return parsed_questions[:num_questions]

        except Exception as e:
            # Return error message if the GPT request fails
            return [(
                "Error: Failed to generate quiz questions.", 
                [], 
                "N/A", 
                str(e), 
                0.0
            )]
            
    # Computes a confidence score for the quiz question explanation by comparing it to the context.
    def _compute_confidence(self, explanation: str, context: str) -> float:
        explanation_embedding = self.embedding_model.embed_query(explanation)
        context_embedding = self.embedding_model.embed_query(context)
        similarity = cosine_similarity([explanation_embedding], [context_embedding])[0][0]
        return float(min(100.0, max(0.0, (similarity + 1) * 50)))
