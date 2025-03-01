import os
import numpy as np
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment  

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialise OpenAI Embeddings & ChromaDB
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

# Initialise session state
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to split audio files into smaller chunks
def split_audio(file_path, chunk_size_mb=24):
    """Split an audio file into smaller chunks (each under chunk_size_mb MB)."""
    audio = AudioSegment.from_file(file_path)
    chunk_length_ms = chunk_size_mb * 60 * 1000  # Convert MB to milliseconds (approx.)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
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
    """Translate text from a source language to English using OpenAI's GPT model."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",  # Use GPT-4 for better translation quality
            messages=[
                {"role": "system", "content": f"Translate the following text from {source_language} to English. The text is about Chinese literature. Preserve the meaning and context."},
                {"role": "user", "content": text}
            ],
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error translating text: {e}"

# Function to detect the source language
def detect_language(text):
    """Detect the source language of the input text."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Detect the language of the following text. The text is likely in Chinese, Malay, or Tamil. Return only the language name (e.g., Chinese, Malay, Tamil)."},
                {"role": "user", "content": text}
            ],
            max_tokens=10,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error detecting language: {e}"

# Process Multiple Files
def preprocess_files(files):
    upload_dir = "./uploads"
    
    # Ensure the "uploads" folder exists before saving files
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    new_files = []

    # Check if files were uploaded
    for file in files:
        if file.name not in st.session_state.processed_files:
            new_files.append(file.name)

    if not new_files:
        return "⚠️ No new files detected. Using existing knowledge base."

    try:
        all_docs = []
        for file in files:
            file_path = os.path.join(upload_dir, file.name)  # Store in uploads/ folder

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            # Handle PDF files
            if file.name.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)
                all_docs.extend(docs)
            # Handle audio files
            elif file.name.endswith((".mp3", ".wav", ".mp4", ".avi")):
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

                # Create a document from the translated text
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = text_splitter.create_documents([translated_text])
                all_docs.extend(docs)

        # Store embeddings in ChromaDB
        if all_docs:  # Ensure there are documents to process
            vectorstore.add_documents(all_docs)
            # vectorstore.persist()

            # Update processed files in session state
            st.session_state.processed_files.extend(new_files)
            return f"{len(new_files)} new files processed successfully!"
        else:
            return "⚠️ No valid text extracted from files."
    except Exception as e:
        return f"Error processing files: {e}"

# Retrieve relevant text from ChromaDB
def retrieve_relevant_chunks(query, top_k=20):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        st.info(f"Retrieved chunks: {results}")  # Log the retrieved chunks
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving documents: {e}"]
        
# Process user query using structured ToT reasoning
def process_query():
    user_input = st.session_state.user_query.strip()

    # Translate non-English queries to English
    if not user_input.isascii():  # Check if the input contains non-ASCII characters (e.g., Chinese, Malay, Tamil)
        st.warning("Detected non-English input. Translating to English...")
        source_language = detect_language(user_input)  # Detect the source language
        translated_input = translate_to_english(user_input, source_language)
        if translated_input.startswith("Error"):
            st.error(translated_input)
            return
        st.success(f"Translated input: {translated_input}")
        user_input = translated_input  # Use the translated input for further processing

    if user_input.lower() == "quiz":
        st.session_state.quiz_mode = True  # Enable quiz mode
        st.switch_page("pages/quiz.py")
        return  # Stop further processing

    elif user_input.lower() == "summarize":
        if not st.session_state.processed_files:
            st.session_state.conversation_history.append(("summarize", "No files have been uploaded. Please upload a file first."))
            return

        summaries = []
        for file in st.session_state.processed_files:
            retrieved_chunks = retrieve_relevant_chunks(file)  # Retrieve relevant content per file
            context = "\n\n".join(retrieved_chunks)

            messages = [
                {"role": "system", "content": "You are a knowledgeable assistant who explains concepts using structured academic reasoning. Always base your answers on the provided context and avoid generating generic or off-topic responses."},
                {"role": "user", "content": f"""
                    Answer the question using a structured academic approach.

                    ### 📌 Introduction
                    - **Briefly introduce** the key concept related to the question.

                    ### 📖 Detailed Explanation
                    - **Break down** the concept step by step.

                    ### 🌍 Real-World Analogy
                    - **Provide an analogy** or example.

                    ### 🎯 Key Takeaways
                    - **Summarize** the most important points.

                    **Question:** {user_input}  
                    **Context:** {context}  
                """}
            ]

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=1500,
                    temperature=0.3,
                )
                summary = response.choices[0].message.content.strip()
                confidence_score = estimate_confidence(response, context)  # Pass context here
                summaries.append(f"### Summary of {file}\n{summary}\n\n**Confidence Score:** {confidence_score:.2f}%")
            except Exception as e:
                summaries.append(f"Error summarizing {file}: {e}")

        full_summary = "\n\n---\n\n".join(summaries)
        st.session_state.conversation_history.append(("summarize", full_summary))
        st.session_state.user_query = ""  # Clear input after processing
        return

    else:
        # Standard processing for other queries
        retrieved_chunks = retrieve_relevant_chunks(user_input)
        context = "\n\n".join(retrieved_chunks)

        messages = [
            {"role": "system", "content": "You are a knowledgeable assistant who explains concepts using structured academic reasoning."},
            {"role": "user", "content": f"""
                Answer the question using a structured academic approach.

                ### 📌 Introduction
                - **Briefly introduce** the key concept related to the question.

                ### 📖 Detailed Explanation
                - **Break down** the concept step by step.

                ### 🌍 Real-World Analogy
                - **Provide an analogy** or example.

                ### 🎯 Key Takeaways
                - **Summarize** the most important points.

                **Question:** {user_input}  
                **Context:** {context}  
            """}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1500,
                temperature=0.7,
            )
            bot_response = response.choices[0].message.content.strip()
            confidence_score = estimate_confidence(response, context)  
            st.session_state.conversation_history.append((user_input, f"{bot_response}\n\n**Confidence Score:** {confidence_score:.2f}%"))
            st.session_state.user_query = ""
        except Exception as e:
            st.session_state.conversation_history.append((user_input, f"Error: {e}"))

def estimate_confidence(response, context):
    response_embedding = embedding_model.embed_query(response.choices[0].message.content)
    context_embedding = embedding_model.embed_query(context)
    
    similarity = cosine_similarity([response_embedding], [context_embedding])[0][0]
    
    # Adjust scaling for better confidence scores
    confidence_score = min(100.0, max(0.0, (similarity + 1) * 50)) 
    return confidence_score

# UI
st.title("🤖 Your academic weapon: SmartBot")

# Sidebar for file upload and selection
with st.sidebar:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Upload Files (PDF, Audio, Video)", type=["pdf", "mp3", "wav", "mp4", "avi"], accept_multiple_files=True)
    if uploaded_files:
        upload_status = preprocess_files(uploaded_files)
        st.success(upload_status)

    if st.session_state.processed_files:
        st.subheader("Uploaded Files")
        selected_file = st.selectbox("Select a file:", st.session_state.processed_files)

# Display chat history
st.subheader("Chat History")
for user_msg, bot_msg in st.session_state.conversation_history:
    st.markdown(f"**🧑‍💻 You:** {user_msg}")
    st.markdown(f"🤖 **ChatBot:**\n\n{bot_msg}", unsafe_allow_html=True)

st.subheader("Enter your query")

# User input field (stores in session state)
st.text_input(
    "Type your question:",
    key="user_query",
    on_change=process_query  # Calls process_query() when Enter is pressed
)

# Check if the user wants to switch to the quiz
if "user_query" in st.session_state and st.session_state.user_query.lower() == "quiz":
    st.page_link("pages/quiz.py")
