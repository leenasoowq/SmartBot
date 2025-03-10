# services/language_service.py

from openai import OpenAI
from typing import List, Dict, Any, Optional

class LanguageService:
    """
    Handles interactions with language models for various tasks like
    question answering, summarization, and content generation.
    """

    def __init__(self, client: OpenAI, model: str = "gpt-3.5-turbo"):
        """
        :param client: An instance of openai.OpenAI
        :param model: The default model to use for completions
        """
        self.client = client
        self.default_model = model

    def generate_answer(
        self, 
        query: str, 
        context: str, 
        model: Optional[str] = None,
        max_tokens: int = 800, 
        temperature: float = 0.7
    ) -> str:
        """
        Generate an answer to a query based on given context.
        
        :param query: The user's question
        :param context: Context information to help answer the question
        :param model: Language model to use (defaults to self.default_model)
        :param max_tokens: Maximum tokens in the response
        :param temperature: Creativity temperature (0-1)
        :return: Generated answer
        """
        model = model or self.default_model
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"

    def summarize_content(
        self, 
        content: str, 
        title: str = "Document",
        model: Optional[str] = None,
        max_tokens: int = 800, 
        temperature: float = 0.3
    ) -> str:
        """
        Generate a summary of the provided content.
        
        :param content: Text content to summarize
        :param title: Title of the content being summarized
        :param model: Language model to use (defaults to self.default_model)
        :param max_tokens: Maximum tokens in the response
        :param temperature: Creativity temperature (0-1)
        :return: Generated summary
        """
        model = model or self.default_model
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes content accurately and concisely."},
            {"role": "user", "content": f"Provide a structured summary of **{title}**.\n\nContent:\n{content}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def generate_quiz(
        self, 
        content: str, 
        num_questions: int = 5,
        model: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.7
    ) -> str:
        """
        Generate a quiz based on the provided content.
        
        :param content: Text content to base the quiz on
        :param num_questions: Number of questions to generate
        :param model: Language model to use (defaults to self.default_model)
        :param max_tokens: Maximum tokens in the response
        :param temperature: Creativity temperature (0-1)
        :return: Generated quiz
        """
        model = model or self.default_model
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates educational quizzes."},
            {"role": "user", "content": f"Generate a quiz with {num_questions} questions based on the following content. Include answers.\n\nContent:\n{content}"}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating quiz: {str(e)}"