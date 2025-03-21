from openai import OpenAI

class LanguageService:
    """
    Performs language detection and translation to English using OpenAI GPT.
    """

    def __init__(self, client: OpenAI):
        """
        :param client: An instance of openai.OpenAI for making GPT calls.
        """
        self.client = client

    def detect_language(self, text: str) -> str:
        """
        Detects the language of 'text' using GPT.
        Returns a string representing the detected language.

        :param text: Input text for language detection.
        :return: Detected language name (e.g., "Chinese", "Malay", "Tamil", or an error string).
        """
        try:
            # For large text, you could chunk it, but this example keeps it simple.
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Detect the language of the following text. "
                            "Return only the language name, for example: 'Chinese' or 'Malay' or 'Tamil'."
                        )
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=10,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error detecting language: {e}"

    def translate_text(self, text: str, source_language: str = "Unknown") -> str:
        """
        Translates the given text into English, optionally referencing the source language.

        :param text: Text to be translated.
        :param source_language: A string describing the detected language (for the prompt).
        :return: Translated English text, or an error message.
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the following text from {source_language} to English. "
                            "Preserve meaning and context."
                        )
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=1000,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error translating text: {e}"
