# transcribe_translate.py (or pages/transcribe_translate.py)

import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Services
from services.audio_service import AudioService
from services.language_service import LanguageService

# Load environment variables (e.g., API keys)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize services
audio_service = AudioService(client)
lang_service = LanguageService(client)

st.title("🎙️ Transcription & Translation")

# Let user upload audio directly on this page
uploaded_audio = st.file_uploader(
    "Upload Audio/Video for transcription (mp3, wav, mp4, avi)",
    type=["mp3", "wav", "mp4", "avi"]
)
# If a file is uploaded, process it
if uploaded_audio:
    file_path = os.path.join("uploads", uploaded_audio.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    # Display a warning while transcribing
    st.warning(f"Transcribing {uploaded_audio.name} ...")
    # Perform audio transcription
    transcription = audio_service.transcribe_audio(file_path)
    # If transcription failed, display an error
    if transcription.startswith("Error"):
        st.error(transcription)
    else:
        # Show transcribed text
        st.info(f"**Transcription:**\n{transcription}")
        # Detect the language of the transcribed text
        detected_lang = lang_service.detect_language(transcription)
        # If language detection failed, show an error
        if detected_lang.startswith("Error"):
            st.error(detected_lang)
        else:
            # Show detected language
            st.info(f"Detected language: {detected_lang}")
            # Translate the transcribed text
            translated_text = lang_service.translate_text(transcription, detected_lang)
            # If translation failed, display an error
            if translated_text.startswith("Error"):
                st.error(translated_text)
            else:
                # Display the translated text
                st.success("Translation complete!")
                st.write(f"**Translated Text:**\n{translated_text}")

                # Optionally save to disk
                output_file = os.path.join("translated_texts", f"{uploaded_audio.name}.txt")
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(translated_text)
                st.write(f"Saved to: {output_file}")
