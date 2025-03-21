import base64
import io
import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
import docx2txt
import numpy as np
import re
import json
import shutil
import openai
from PIL import Image  
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Importing custom service modules
from services.language_service import LanguageService
from services.document_service import DocumentService
from services.helpers import (  
    update_conversation_history, reset_session_state, handle_image_query,
    sanitize_collection_name, load_processed_files, save_processed_files, 
    preprocess_files, handle_summary_request, clear_selected_file, 
    detect_image_request, detect_weakness_request, handle_weakness_analysis, 
    handle_image_request, detect_reference_request, fetch_online_references, 
    is_follow_up, retrieve_relevant_chunks, get_expected_answer, 
    evaluate_user_answer, estimate_confidence, process_response
)

# Load environment variables (API keys, configurations)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize services for language processing and document management
lang_service = LanguageService(client)
doc_service = DocumentService(client)

# Initialize session state variables for chatbot functionality
if "processed_files" not in st.session_state:
    st.session_state["processed_files"] = load_processed_files()

# Initialize conversation-related session variables
for key in ["conversation_history", "quiz_mode", "selected_file", "last_bot_question", "last_expected_answer", "pending_response"]:
    if key not in st.session_state:
        st.session_state[key] = (
            [] if key in ["conversation_history"] 
            else False if key == "quiz_mode"
            else None if key in ["selected_file", "last_expected_answer", "pending_response"]
            else ""  # For last_bot_question
        )

# Maximum allowed conversation history length
MAX_HISTORY = 50

# UI Setup
st.title("🤖 Your Academic Chatbot")

# Upload Image
st.subheader("Upload an Image for Analysis")
image_file = st.file_uploader("Choose an image to be explained", type=["png", "jpg", "jpeg"],  key=f"image_uploader_{st.session_state.get('uploader_key', 0)}")

if image_file:
    if "image_uploaded" not in st.session_state or st.session_state["image_uploaded"] != image_file:
        st.session_state["conversation_history"].append(("user", "📷 [Image Uploaded]"))
        st.session_state["image_uploaded"] = image_file  # Store uploaded image reference
        st.session_state["pending_response"] = {"text": None, "image": image_file}

        # Prevent infinite rerun after clearing history
        if image_file is not None:
            st.rerun()  # Rerun only when a new image is uploaded

if "uploading" not in st.session_state:
    st.session_state["uploading"] = False
if "files_to_upload" not in st.session_state:
    st.session_state["files_to_upload"] = None
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

if st.session_state["uploading"]:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"], .block-container {
            pointer-events: none !important;
            opacity: 0.5 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

with st.sidebar:
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOC, DOCX, or TXT",
        type=["pdf", "txt", "doc", "docx"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state['uploader_key']}"
    )
    if uploaded_files:
        st.session_state["files_to_upload"] = uploaded_files
    if st.session_state["files_to_upload"]:
        if st.button("Upload", disabled=st.session_state["uploading"]):
            st.session_state["uploading"] = True
            with st.spinner("Processing files... Please wait!"):
                msg = preprocess_files(st.session_state["files_to_upload"])
            st.success(msg)
            st.session_state["files_to_upload"] = None
            st.session_state["uploading"] = False
            st.session_state["uploader_key"] += 1
            st.rerun()

    if st.session_state["processed_files"]:
        st.subheader("Available Files")
        st.session_state["selected_file"] = st.selectbox(
            "Select a file",
            st.session_state["processed_files"],
            disabled=st.session_state["uploading"]
        )
    # Add the "Clear Selected File" button
        if st.button("Clear Selected File", key="clear_selected_file"):
            clear_selected_file()
            st.rerun()

if st.session_state["uploading"]:
    st.stop()

st.header("Chat")
chat_container = st.container()


if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []
if "pending_response" not in st.session_state:
    st.session_state["pending_response"] = None
if "image_uploaded" not in st.session_state:
    st.session_state["image_uploaded"] = None  # Track uploaded image to avoid infinite loop

with chat_container:
    for role, message in st.session_state["conversation_history"]:
        with st.chat_message(role):
            st.markdown(message)


# Dynamic Chat Input
prompt = st.chat_input("Enter your query:", disabled=st.session_state.get("uploading", False))

if prompt:
    st.session_state["conversation_history"].append(("user", f"📝 {prompt}"))
    st.session_state["pending_response"] = {"text": prompt, "image": None}
    st.rerun()

# Process the response if there's a pending input
if st.session_state["pending_response"]:
    with st.spinner("Thinking..."):
        process_response(st.session_state["pending_response"])
    st.session_state["pending_response"] = None
    st.rerun()

# Clear Chat History
if st.button("Clear History"):
    reset_session_state()
    st.rerun()