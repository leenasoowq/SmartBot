import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Services
from services.audio_service import AudioService
from services.language_service import LanguageService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

audio_service = AudioService(client)
lang_service = LanguageService(client)

st.title("🎙️ Transcription & Translation")

# Let user upload audio directly on this page
uploaded_audio = st.file_uploader(
    "Upload Audio/Video for transcription (mp3, wav, mp4, avi)",
    type=["mp3", "wav", "mp4", "avi"]
)

if uploaded_audio:
    file_path = os.path.join("uploads", uploaded_audio.name)
    os.makedirs("uploads", exist_ok=True)  # Ensure uploads directory exists
    
    with open(file_path, "wb") as f:
        f.write(uploaded_audio.getbuffer())
    
    st.warning(f"Transcribing {uploaded_audio.name} ...")
    transcription = audio_service.transcribe_audio(file_path)
    if transcription.startswith("Error"):
        st.error(transcription)
    else:
        st.info(f"**Transcription:**\n{transcription}")

        detected_lang = lang_service.detect_language(transcription)
        if detected_lang.startswith("Error"):
            st.error(detected_lang)
        else:
            st.info(f"Detected language: {detected_lang}")
            translated_text = lang_service.translate_text(transcription, detected_lang)
            if translated_text.startswith("Error"):
                st.error(translated_text)
            else:
                st.success("Translation complete!")
                st.write(f"**Translated Text:**\n{translated_text}")

                # Ensure translated_texts directory exists before saving
                output_dir = "translated_texts"
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{uploaded_audio.name}.txt")
                with open(output_file, "w", encoding="utf-8") as out:
                    out.write(translated_text)
                st.write(f"Saved to: {output_file}")