import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from pydub import AudioSegment
from collections import Counter  # For language detection

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Function to split audio files into smaller chunks
def split_audio(file_path, chunk_size_mb=24):
    """Split an audio file into smaller chunks (each under chunk_size_mb MB)."""
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = chunk_size_mb * 60 * 1000  # Convert MB to milliseconds (approx.)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

def split_text_into_chunks(text, max_tokens=8000):
    """Split text into smaller chunks that fit within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_tokens:
            chunks.append(" ".join(current_chunk[:-1]))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
def split_text_into_chunks(text, max_tokens=4000):
    """Split text into smaller chunks that fit within the token limit."""
    words = text.split()
    chunks = []
    current_chunk = []

    current_token_count = 0
    for word in words:
        # Approximate token count: 1 token ~= 4 characters
        word_token_count = len(word) // 4 + 1
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

# Function to transcribe audio using OpenAI Whisper
def transcribe_audio(file_path):
    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > 25:
            st.warning(f"File size ({file_size_mb:.2f} MB) exceeds 25 MB limit. Splitting into smaller chunks...")
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
        return f"Error transcribing audio: {e}"

# Function to translate text to English
def translate_to_english(text, source_language):
    """Translate text from a source language to English."""
    try:
        # Split text into chunks if it's too long
        chunks = split_text_into_chunks(text, max_tokens=4000)  # Reduced max_tokens
        translated_texts = []

        for chunk in chunks:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": f"Translate the following text from {source_language} to English. Preserve the meaning and context."},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=1000,  # Reduced max_tokens for completion
                temperature=0.3,
            )
            translated_texts.append(response.choices[0].message.content.strip())

        return " ".join(translated_texts)
    except Exception as e:
        return f"Error translating text: {e}"

# Function to detect the source language
def detect_language(text):
    """Detect the source language of the input text."""
    try:
        # Split text into chunks if it's too long
        chunks = split_text_into_chunks(text)
        detected_languages = []

        for chunk in chunks:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Detect the language of the following text. Return only the language name (e.g., Chinese, Malay, Tamil)."},
                    {"role": "user", "content": chunk}
                ],
                max_tokens=10,
                temperature=0.1,
            )
            detected_languages.append(response.choices[0].message.content.strip())

        # Return the most common language detected
        return Counter(detected_languages).most_common(1)[0][0]
    except Exception as e:
        return f"Error detecting language: {e}"

# UI for the new page
st.title("🎙️ Transcription & Translation")

# Upload Files
uploaded_files = st.file_uploader("Upload Files (PDF, Audio, Video)", type=["pdf", "mp3", "wav", "mp4", "avi"], accept_multiple_files=True)
if uploaded_files:
    upload_dir = "./uploads"
    translated_dir = "./translated_texts"
    
    # Ensure the "uploads" and "translated_texts" folders exist before saving files
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
    if not os.path.exists(translated_dir):
        os.makedirs(translated_dir)

    for file in uploaded_files:
        file_path = os.path.join(upload_dir, file.name)  # Store in uploads/ folder

        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        # Handle audio files
        if file.name.endswith((".mp3", ".wav", ".mp4", ".avi")):
            st.warning(f"Processing audio file: {file.name}")
            transcription = transcribe_audio(file_path)
            if transcription.startswith("Error"):
                st.error(transcription)
                continue
            st.info(f"Transcription: {transcription}")  # Log the transcription

            # Detect the language of the transcription
            source_language = detect_language(transcription)
            if source_language.startswith("Error"):
                st.error(source_language)
                continue
            st.info(f"Detected Language: {source_language}")  # Log the detected language

            # Translate the transcription to English
            translated_text = translate_to_english(transcription, source_language)
            if translated_text.startswith("Error"):
                st.error(translated_text)
                continue
            st.info(f"Translated Text: {translated_text}")  # Log the translated text

            # Save the translated text to a file
            translated_file_path = os.path.join(translated_dir, f"{os.path.splitext(file.name)[0]}.txt")
            with open(translated_file_path, "w", encoding="utf-8") as translated_file:
                translated_file.write(translated_text)
            st.success(f"Translated text saved to: {translated_file_path}")

    st.success("Finished processing files!")