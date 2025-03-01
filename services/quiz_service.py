# services/quiz_service.py

import random
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings import OpenAIEmbeddings

class QuizService:
    """
    Generates multiple-choice quiz questions from text context using GPT, and can compute
    explanation-confidence scores based on embeddings.
    """

    def __init__(self,
                 client: OpenAI,
                 embedding_model_name: str = "text-embedding-3-large"):
        """
        :param client: An instance of openai.OpenAI for GPT calls.
        :param embedding_model_name: The OpenAI model name for generating text embeddings.
        """
        self.client = client
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)

    def generate_quiz_questions(self, context: str,
                                difficulty: str = "Medium",
                                num_questions: int = 5) -> list:
        """
        Uses the provided 'context' to generate multiple-choice quiz questions with GPT.

        :param context: Text context from which to generate quiz questions.
        :param difficulty: Difficulty level of the quiz ('Easy', 'Medium', or 'Hard').
        :param num_questions: Number of MCQ questions to produce.
        :return: A list of tuples (question_text, [shuffled_options], correct_answer_letter,
                                  explanation, confidence_score).
        """
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an academic quiz assistant generating {difficulty}-level questions "
                    f"based on the provided context."
                ),
            },
            {
                "role": "user",
                "content": f"""
Please generate exactly {num_questions} {difficulty}-level multiple-choice quiz questions using the context below.

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
                temperature=0.3,
            )
            quiz_text = response.choices[0].message.content.strip()

            # Split the GPT output by double newlines for separate questions
            question_blocks = quiz_text.split("\n\n")
            parsed_questions = []

            for block in question_blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if ("Correct Answer:" in block) and ("Explanation:" in block) and lines:
                    try:
                        # Basic text parsing for the expected format
                        question_line = lines[0].replace("Question: ", "").strip()
                        option_a = lines[1].split(") ", 1)[1].strip()
                        option_b = lines[2].split(") ", 1)[1].strip()
                        option_c = lines[3].split(") ", 1)[1].strip()
                        option_d = lines[4].split(") ", 1)[1].strip()

                        correct_letter_line = lines[5].replace("Correct Answer:", "").strip().upper()
                        explanation_line = lines[6].replace("Explanation:", "").strip()

                        options = [option_a, option_b, option_c, option_d]
                        # Shuffle to avoid the correct one always being in the same position
                        shuffled = options[:]
                        random.shuffle(shuffled)
                        
                        # Convert original correct_letter_line (A/B/C/D) to new shuffled index
                        original_index = ["A", "B", "C", "D"].index(correct_letter_line)
                        correct_option_text = options[original_index]
                        new_correct_index = shuffled.index(correct_option_text)
                        new_correct_letter = ["A", "B", "C", "D"][new_correct_index]

                        # Compute optional confidence score
                        confidence_score = self._compute_confidence(explanation_line, context)

                        parsed_questions.append((
                            question_line,
                            shuffled,
                            new_correct_letter,
                            explanation_line,
                            confidence_score
                        ))
                    except Exception:
                        # Skip block if parsing fails
                        continue

            # Return only up to 'num_questions'
            return parsed_questions[:num_questions]

        except Exception as e:
            return [(
                "Error: Failed to generate quiz questions.", 
                [], 
                "N/A", 
                str(e), 
                0.0
            )]

    def _compute_confidence(self, explanation: str, context: str) -> float:
        """
        Computes a rough "confidence score" by measuring cosine similarity
        between the explanation and the entire context, scaled to [0..100].

        :param explanation: Explanation text from the question block.
        :param context: The overall context text from which questions were created.
        :return: A confidence score in the range [0..100].
        """
        explanation_embedding = self.embedding_model.embed_query(explanation)
        context_embedding = self.embedding_model.embed_query(context)
        similarity = cosine_similarity([explanation_embedding], [context_embedding])[0][0]
        # Scale from [-1..1] range to [0..100]
        return float(min(100.0, max(0.0, (similarity + 1) * 50)))
