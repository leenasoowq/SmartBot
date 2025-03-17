# services/audio_service.py
import os
from pydub import AudioSegment
from openai import OpenAI

class AudioService:
    """
    Provides audio-related utilities: chunking audio and transcribing via Whisper.
    """
    # Initializes the audio service with OpenAI client and a max file size limit.
    def __init__(self, client: OpenAI, max_size_mb: float = 25.0):
        self.client = client
        self.max_size_mb = max_size_mb
    # Splits an audio file into smaller chunks if it exceeds a specified size.
    def split_audio(self, file_path: str, chunk_size_mb: int = 24) -> list[AudioSegment]:
        """
        Splits an audio file into smaller chunks if over chunk_size_mb MB.
        Returns a list of AudioSegments.
        """
        audio = AudioSegment.from_file(file_path)
        chunk_length_ms = chunk_size_mb * 60 * 1000  # MB -> approx. minutes -> ms
        return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    # Transcribes an audio file using OpenAI's Whisper model, chunking if necessary.
    def transcribe_audio(self, file_path: str) -> str:
        """
        Transcribes an audio file using OpenAI Whisper (model='whisper-1').
        Auto-chunks if file > self.max_size_mb.
        """
        try:
            # Get the file size in MB
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
             # If file is too large, split into smaller chunks
            if file_size_mb > self.max_size_mb:
                audio_chunks = self.split_audio(file_path)
                partials = []
                # Process each chunk separately
                for i, chunk in enumerate(audio_chunks):
                    tmp_path = f"{file_path}_chunk_{i}.mp3"
                    chunk.export(tmp_path, format="mp3")
                    # Open and transcribe the chunk using OpenAI's Whisper
                    with open(tmp_path, "rb") as f:
                        result = self.client.audio.transcriptions.create(
                            model="whisper-1",
                            file=f,
                            response_format="text"
                        )
                    # Store the transcribed text and delete temporary chunk
                    partials.append(result)
                    os.remove(tmp_path)
                # Merge all transcribed parts into a single text response
                return " ".join(partials)
            else:
                # Transcribe directly
                with open(file_path, "rb") as f:
                    result = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        response_format="text"
                    )
                return result
        except Exception as e:
            return f"Error transcribing audio: {e}"
