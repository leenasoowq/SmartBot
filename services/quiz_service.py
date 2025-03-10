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
                 embedding_model_name: str = "text-embedding-3-small"):
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
        
        With improved handling for large documents, focusing on content rather than metadata.

        :param context: Text context from which to generate quiz questions.
        :param difficulty: Difficulty level of the quiz ('Easy', 'Medium', or 'Hard').
        :param num_questions: Number of MCQ questions to produce.
        :return: A list of tuples (question_text, [shuffled_options], correct_answer_letter,
                                  explanation, confidence_score).
        """
        # Enhanced prompt to focus on content-based questions, not metadata
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are an academic quiz assistant generating {difficulty}-level questions "
                    f"based on the provided context. Focus ONLY on the actual content and concepts "
                    f"presented in the text. DO NOT create questions about metadata "
                    f"unless the text specifically "
                    f"discusses these as important topics. Create conceptual questions that test understanding "
                    f"of the material's core ideas, theories, and applications."
                ),
            },
            {
                "role": "user",
                "content": f"""
Please generate exactly {num_questions} {difficulty}-level multiple-choice quiz questions using the context below.

IMPORTANT INSTRUCTIONS:
1. Focus ONLY on the actual content, not on metadata
2. Do not create questions about the author, publication date, or edition information
3. Questions should test understanding of key concepts and ideas
4. All answers should be findable in the provided context, not requiring external knowledge

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
                        
                        # Skip questions about metadata (additional filter)
                        metadata_keywords = ["author", "published", "edition", "publisher", "publication date", 
                                           "book was written", "writer", "when was this book", "who wrote",
                                           "written by", "publishing company", "copyright"]
                        if any(keyword in question_line.lower() for keyword in metadata_keywords):
                            continue
                            
                        option_a = lines[1].split(") ", 1)[1].strip()
                        option_b = lines[2].split(") ", 1)[1].strip()
                        option_c = lines[3].split(") ", 1)[1].strip()
                        option_d = lines[4].split(") ", 1)[1].strip()

                        correct_letter_line = lines[5].replace("Correct Answer:", "").strip().upper()
                        
                        # Improved handling of explanation text that might span multiple lines
                        explanation_start_idx = next((i for i, line in enumerate(lines) if "Explanation:" in line), -1)
                        if explanation_start_idx != -1:
                            explanation_parts = [lines[explanation_start_idx].replace("Explanation:", "").strip()]
                            for i in range(explanation_start_idx + 1, len(lines)):
                                if not any(x in lines[i] for x in ["Question:", "A)", "B)", "C)", "D)", "Correct Answer:"]):
                                    explanation_parts.append(lines[i])
                            explanation_line = " ".join(explanation_parts)
                        else:
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

                        # Check relevance to context more strictly
                        question_relevance = self._compute_confidence(question_line, context)
                        
                        # Only include questions with high relevance to the context
                        if question_relevance > 60:
                            # Compute confidence score
                            confidence_score = self._compute_confidence(explanation_line, context)
    
                            parsed_questions.append((
                                question_line,
                                shuffled,
                                new_correct_letter,
                                explanation_line,
                                confidence_score
                            ))
                    except Exception as e:
                        # Skip block if parsing fails
                        print(f"Error parsing question block: {e}")
                        continue

            # If we don't have enough questions after filtering, regenerate with stricter constraints
            if len(parsed_questions) < min(2, num_questions):
                # Make a more focused prompt for a second attempt
                return self._fallback_question_generation(context, difficulty, num_questions)
                
            # Return only up to 'num_questions'
            return parsed_questions[:num_questions]

        except Exception as e:
            print(f"Error generating quiz questions: {e}")
            return [(
                "Error: Failed to generate quiz questions.", 
                [], 
                "N/A", 
                str(e), 
                0.0
            )]
            
    def _fallback_question_generation(self, context, difficulty, num_questions):
        """
        Fallback method for when the initial quiz generation produces too many metadata questions.
        Uses a more restrictive prompt to focus exclusively on content.
        """
        # Truncate context if it's too large
        max_context_chars = 12000
        if len(context) > max_context_chars:
            # Extract the most informative segments
            sentences = context.split('. ')
            filtered_sentences = [s for s in sentences if len(s) > 50 and not any(w in s.lower() for w in ["copyright", "published", "edition", "author"])]
            context = '. '.join(filtered_sentences[:30])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an academic quiz creator focused EXCLUSIVELY on content understanding. "
                    "You must NEVER create questions about authors, publication information, "
                    "publication dates, or edition numbers. Focus only on the key concepts, theories, "
                    "methods, and ideas in the provided text."
                ),
            },
            {
                "role": "user",
                "content": f"""
Generate {num_questions} conceptual multiple-choice questions that test understanding of the core ideas in the text.

STRICT RULES:
- NO questions about who wrote the material, when it was written, or where it was published
- Focus on the ACTUAL CONTENT only - concepts, ideas, definitions, processes
- All questions must be answerable from the content provided
- Each question should test understanding, not just recall

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

            # Parse as before
            question_blocks = quiz_text.split("\n\n")
            parsed_questions = []

            for block in question_blocks:
                lines = [line.strip() for line in block.split("\n") if line.strip()]
                if ("Correct Answer:" in block) and ("Explanation:" in block) and lines:
                    try:
                        question_line = lines[0].replace("Question: ", "").strip()
                        
                        # Even stricter filter for metadata questions
                        metadata_keywords = ["author", "published", "edition", "publisher", "copyright", 
                                           "book was written", "writer", "publication date", "publishing", 
                                           "published by", "wrote", "written by"]
                        if any(keyword in question_line.lower() for keyword in metadata_keywords):
                            continue
                            
                        # Rest of parsing logic same as original method
                        option_a = lines[1].split(") ", 1)[1].strip()
                        option_b = lines[2].split(") ", 1)[1].strip()
                        option_c = lines[3].split(") ", 1)[1].strip()
                        option_d = lines[4].split(") ", 1)[1].strip()

                        correct_letter_line = lines[5].replace("Correct Answer:", "").strip().upper()
                        
                        # Handle multi-line explanations
                        explanation_start_idx = next((i for i, line in enumerate(lines) if "Explanation:" in line), -1)
                        if explanation_start_idx != -1:
                            explanation_parts = [lines[explanation_start_idx].replace("Explanation:", "").strip()]
                            for i in range(explanation_start_idx + 1, len(lines)):
                                if not any(x in lines[i] for x in ["Question:", "A)", "B)", "C)", "D)", "Correct Answer:"]):
                                    explanation_parts.append(lines[i])
                            explanation_line = " ".join(explanation_parts)
                        else:
                            explanation_line = lines[6].replace("Explanation:", "").strip()
                        
                        options = [option_a, option_b, option_c, option_d]
                        shuffled = options[:]
                        random.shuffle(shuffled)
                        
                        original_index = ["A", "B", "C", "D"].index(correct_letter_line)
                        correct_option_text = options[original_index]
                        new_correct_index = shuffled.index(correct_option_text)
                        new_correct_letter = ["A", "B", "C", "D"][new_correct_index]
                        
                        confidence_score = self._compute_confidence(explanation_line, context)
                        
                        parsed_questions.append((
                            question_line,
                            shuffled,
                            new_correct_letter,
                            explanation_line,
                            confidence_score
                        ))
                    except Exception as e:
                        print(f"Error parsing fallback question: {e}")
                        continue
                        
            return parsed_questions[:num_questions]
        except Exception as e:
            print(f"Error in fallback generation: {e}")
            return [(
                "Error: Failed to generate quiz questions even with fallback.", 
                [], 
                "N/A", 
                str(e), 
                0.0
            )]

    def _compute_confidence(self, explanation: str, context: str) -> float:
        """
        Computes a rough "confidence score" by measuring cosine similarity
        between the explanation and the context.
        
        Ultra-safe implementation to prevent batch size errors.
        """
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Function to safely get embeddings with extreme truncation if needed
            def get_safe_embedding(text):
                # Try with progressively smaller text until it works
                text_sizes = [100, 75, 50, 25]
                for size in text_sizes:
                    try:
                        truncated = text[:size]
                        return self.embedding_model.embed_query(truncated)
                    except Exception as e:
                        print(f"Embedding failed with size {size}: {e}")
                        continue
                
                # If all else fails, use a minimal text sample
                return self.embedding_model.embed_query(text[:10])
            
            # Get embeddings safely
            expl_embedding = get_safe_embedding(explanation)
            
            # For context, we'll just take several very small samples and average
            context_samples = []
            if len(context) > 500:
                # Take samples from different parts of the context
                step = len(context) // 5
                for i in range(0, len(context), step):
                    if i < len(context):
                        context_samples.append(context[i:i+50])
            else:
                context_samples = [context[:50]]
            
            # Get embeddings for each context sample
            ctx_embeddings = []
            for sample in context_samples[:5]:  # Limit to 5 samples
                try:
                    ctx_embeddings.append(get_safe_embedding(sample))
                except Exception:
                    continue
            
            # Average the context embeddings
            if ctx_embeddings:
                ctx_embedding = np.mean(ctx_embeddings, axis=0)
            else:
                # If all context embeddings failed, try with a minimal sample
                ctx_embedding = get_safe_embedding(context[:10])
            
            # Calculate similarity
            similarity = cosine_similarity([expl_embedding], [ctx_embedding])[0][0]
            
            # Scale to 0-100
            return float(min(100.0, max(0.0, (similarity + 1) * 50)))
        except Exception as e:
            print(f"Complete confidence calculation failure: {e}")
            return 50.0  # Return a neutral score on error