# utils.py
import os
import logging
from pydub import AudioSegment
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def split_audio(file_path, chunk_size_mb=24):
    """Split an audio file into smaller chunks (each under chunk_size_mb MB)."""
    try:
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = chunk_size_mb * 60 * 1000  # Convert MB to milliseconds (approx.)
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        return chunks
    except Exception as e:
        logging.error(f"Error splitting audio file {file_path}: {e}")
        raise

def transcribe_audio(file_path):
    """Transcribe audio using OpenAI Whisper."""
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 25:
            logging.warning(f"File size ({file_size_mb:.2f} MB) exceeds 25 MB limit. Splitting into smaller chunks...")
            audio_chunks = split_audio(file_path)
            transcriptions = []
            for i, chunk in enumerate(audio_chunks):
                chunk_path = f"{file_path}_chunk_{i}.mp3"
                chunk.export(chunk_path, format="mp3")
                with open(chunk_path, "rb") as chunk_file:
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=chunk_file,
                        response_format="text"
                    )
                transcriptions.append(transcription)
                os.remove(chunk_path)  # Clean up temporary chunk file
            return " ".join(transcriptions)
        else:
            with open(file_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            return transcription
    except Exception as e:
        logging.error(f"Error transcribing audio file {file_path}: {e}")
        raise

def split_text_into_chunks(text, max_tokens=4000):
    """Split text into smaller chunks that fit within the token limit."""
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_token_count = 0

        for word in words:
            word_token_count = len(word) // 4 + 1  # Approximate token count
            if current_token_count + word_token_count > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_token_count = word_token_count
            else:
                current_chunk.append(word)
                current_token_count += word_token_count

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
        raise