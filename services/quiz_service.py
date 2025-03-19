import random
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings

class QuizService:
    def __init__(self, client: OpenAI, embedding_model_name: str = "text-embedding-3-large"):
        self.client = client
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

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
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1500,
                temperature=0.5,
            )
            quiz_text = response.choices[0].message.content.strip()
            print(f"DEBUG: Raw GPT Response:\n{quiz_text}\n")  # Log raw GPT output

            question_blocks = quiz_text.split("\n\n")
            parsed_questions = []

            def parse_quiz_block(block):
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if len(lines) < 6:  # Minimum lines for question + 4 options + answer/explanation
                    print(f"DEBUG: Skipping malformed block (too few lines): {block}")
                    return None
                try:
                    question = lines[0].replace("Question: ", "").strip()
                    options = [lines[i].split(") ", 1)[1].strip() for i in range(1, 5)]
                    correct_letter = lines[5].replace("Correct Answer:", "").strip().upper()
                    explanation = lines[6].replace("Explanation:", "").strip()
                    return (question, options, correct_letter, explanation)
                except IndexError as e:
                    print(f"DEBUG: Parsing error (IndexError) in block: {block}\nError: {e}")
                    return None
                except Exception as e:
                    print(f"DEBUG: Parsing error in block: {block}\nError: {e}")
                    return None

            for block in question_blocks:
                parsed = parse_quiz_block(block)
                if parsed:
                    question, options, correct_letter, explanation = parsed
                    shuffled = options[:]
                    random.shuffle(shuffled)
                    original_index = ["A", "B", "C", "D"].index(correct_letter)
                    correct_option_text = options[original_index]
                    new_correct_index = shuffled.index(correct_option_text)
                    new_correct_letter = ["A", "B", "C", "D"][new_correct_index]
                    confidence_score = self._compute_confidence(explanation, context)
                    parsed_questions.append((
                        question,
                        shuffled,
                        new_correct_letter,
                        explanation,
                        confidence_score
                    ))
                    print(f"DEBUG: Successfully parsed question: {question}")  # Log successful parsing
                else:
                    print(f"DEBUG: Failed to parse block: {block}")

            if not parsed_questions:
                print("DEBUG: No valid questions parsed from GPT response.")
            else:
                print(f"DEBUG: Parsed {len(parsed_questions)} questions successfully.")

            return parsed_questions[:num_questions]

        except Exception as e:
            print(f"DEBUG: API Error in generate_quiz_questions: {e}")
            return [(
                "Error: Failed to generate quiz questions.", 
                [], 
                "N/A", 
                str(e), 
                0.0
            )]

    def _compute_confidence(self, explanation: str, context: str) -> float:
        explanation_embedding = self.embedding_model.embed_query(explanation)
        context_embedding = self.embedding_model.embed_query(context)
        similarity = cosine_similarity([explanation_embedding], [context_embedding])[0][0]
        return float(min(100.0, max(0.0, (similarity + 1) * 50)))