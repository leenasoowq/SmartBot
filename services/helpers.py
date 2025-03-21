import os
import json
import re
import shutil
import docx2txt
import streamlit as st
from typing import List
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from openai import OpenAI
import base64
import io
from PIL import Image  
from services.language_service import LanguageService
from services.document_service import DocumentService

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

lang_service = LanguageService(client)
doc_service = DocumentService(client)
embedding_model = doc_service.embedding_model

# --- File Name Sanitizer ---
def sanitize_collection_name(file_name: str) -> str:
    base_name = file_name.rsplit(".", 1)[0]
    sanitized = "".join(c for c in base_name if c.isalnum() or c in ["_", "-"])
    sanitized = sanitized.replace("__", "_").replace("--", "-").strip("_-")[:63]
    return sanitized if len(sanitized) >= 3 else sanitized + "_doc"

# --- Load & Save Processed Files ---
def load_processed_files(file_path="data/processed_files.json"):
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading processed files: {e}")
    return []

def save_processed_files(files, file_path="data/processed_files.json"):
    try:
        with open(file_path, 'w') as f:
            json.dump(files, f)
    except Exception as e:
        st.error(f"Error saving processed files: {e}")

# --- Conversation History ---
def update_conversation_history(role: str, message: str) -> None:
    if role in ["user", "assistant"]:
        st.session_state["conversation_history"].append((role, message))
        if role == "assistant":
            st.session_state["last_bot_answer"] = message
        if len(st.session_state["conversation_history"]) > 50:
            st.session_state["conversation_history"] = st.session_state["conversation_history"][-50:]

# --- Session Reset ---
def reset_session_state():
    for key in [
        "conversation_history", "quiz_mode", "selected_file", "last_bot_question",
        "last_expected_answer", "pending_response", "weak_topics", "quiz_history",
        "image_uploaded", "files_to_upload"
    ]:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["conversation_history"] = []
    st.session_state["weak_topics"] = []
    st.session_state["quiz_history"] = [] 
    st.session_state["image_uploaded"] = None 
    st.session_state["pending_response"] = None 
    st.session_state["files_to_upload"] = None 
    st.session_state["uploader_key"] += 1
    st.rerun()
    update_conversation_history("assistant", "Chat history, quiz progress, and weaknesses have been cleared. Processed files remain intact.")

# Processes uploaded files (PDF, DOCX, TXT) and extracts text into collections for retrieval.
def preprocess_files(files):
    """Upload and parse PDF, DOC/DOCX, TXT files into collections."""
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    processed_files = st.session_state["processed_files"]
    new_files = [f.name for f in files if f.name not in processed_files]

    if not new_files:
        return "No new files to process."
    
    processed_count = 0
    for file in files:
        if file.name not in new_files:
            continue
        file_path = os.path.join(upload_dir, file.name)
        with open(file_path, "wb") as out_file:
            out_file.write(file.getbuffer())
        ext = file.name.lower()
        text_docs, image_docs = [], []
        
        if ext.endswith(".pdf"):
            text_docs, image_docs = doc_service.process_pdf(file_path)
        elif ext.endswith((".docx", ".doc")):
            try:
                text = docx2txt.process(file_path)
                text_docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        elif ext.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text_docs = doc_service.process_text(text)
            except Exception as e:
                st.error(f"Error reading {file.name}: {e}")
                continue
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue

        if text_docs or image_docs:
            collection_name = sanitize_collection_name(file.name)
            
            # Add metadata to text documents
            for d in text_docs:
                if isinstance(d, Document):
                    d.metadata["file_name"] = file.name
            
            # Add metadata to image documents
            for d in image_docs:
                if isinstance(d, Document):
                    d.metadata["file_name"] = file.name
            
            # Add documents separately to vector store
            if text_docs:
                doc_service.add_documents_to_vectorstore(text_docs, collection_name)
            if image_docs:
                doc_service.add_documents_to_vectorstore(image_docs, f"{collection_name}_images")
            
            processed_files.append(file.name)
            processed_count += 1

        os.remove(file_path)

    # Update the session state and save to disk
    st.session_state["processed_files"] = processed_files
    save_processed_files(processed_files)

    return f"Processed {processed_count} new file(s) successfully!" if processed_count else "No valid text extracted."





def handle_summary_request():
    file_name = st.session_state.get("selected_file")
    
    if not file_name:
        update_conversation_history("assistant", "No document is selected. Please upload or select a file first.")
        return
    
    # Retrieve content
    chunks = retrieve_relevant_chunks(file_name)
    
    # Debug: Log chunk contents
    if not chunks:
        update_conversation_history("assistant", "Could not extract content from the document.")
        print("Debug: No content extracted from document.")
        return
    
    document_text = "\n\n".join(chunks)
    
    # Debug: Log document text length
    print(f"Debug: Extracted document text length - {len(document_text)} characters")

    if len(document_text.strip()) < 100:
        update_conversation_history("assistant", "Document content is too short for summarization.")
        print("Debug: Extracted text is too short.")
        return
    
    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You summarize the provided text concisely."},
                {"role": "user", "content": f"Summarize the following document:\n\n{document_text}"}
            ],
            max_tokens=150,
            temperature=0.3
        )
        
        summary = r.choices[0].message.content.strip()
        
        # Debug: Log summary output
        print(f"Debug: Generated summary - {summary}")

        update_conversation_history("assistant", f"**Summary:**\n\n{summary}")
    except Exception as e:
        update_conversation_history("assistant", f"Error generating summary: {e}")
        print(f"Debug: Error in summary generation - {e}")


# Function to clear chatbot session state
def reset_session_state():
    """Reset session variables that store conversation history and other related data."""
    keys_to_reset = [
        "conversation_history", "quiz_mode", "selected_file", 
        "last_bot_question", "last_expected_answer", "pending_response", "weak_topics", "quiz_history", "image_uploaded","files_to_upload"
    ]
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state["conversation_history"] = []  # Ensure empty history is initialized
    st.session_state["weak_topics"] = []  # Reinitialize empty weaknesses list
    st.session_state["quiz_history"] = [] 
    st.session_state["image_uploaded"] = None 
    st.session_state["pending_response"] = None 
    st.session_state["files_to_upload"] = None 

    st.session_state["uploader_key"] += 1  
    st.rerun()
   
    update_conversation_history("assistant", "hat history, quiz progress, and weaknesses have been cleared. Processed files remain intact.")

# Function to remove a selected file from the processed files list
def clear_selected_file():
    """Fully remove a selected file's metadata, images, vector data, and ChromaDB stored files."""
    if not st.session_state.get("selected_file"):
        st.warning("No file selected to clear.")
        return

    file_to_clear = st.session_state["selected_file"]
    collection_name = sanitize_collection_name(file_to_clear)
    image_collection_name = f"{collection_name}_images"

    # Remove from processed files list
    if file_to_clear in st.session_state["processed_files"]:
        st.session_state["processed_files"].remove(file_to_clear)

    # Delete metadata file (stored JSON file)
    metadata_file = os.path.join("data", f"{file_to_clear}.json")
    if os.path.exists(metadata_file):
        os.remove(metadata_file)

    # Delete stored images
    image_folder = os.path.join("images", file_to_clear)
    if os.path.exists(image_folder):
        shutil.rmtree(image_folder)

    # Delete stored data (text/vector data)
    data_folder = os.path.join("data", file_to_clear)
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)

    # Delete ChromaDB stored data (Persistent Mode)
    chromadb_folder = "chroma_db"  # Change this to the actual ChromaDB storage path
    if os.path.exists(chromadb_folder):
        try:
            shutil.rmtree(chromadb_folder)
            print(f"Successfully deleted ChromaDB folder: {chromadb_folder}")
        except Exception as e:
            print(f"Error deleting ChromaDB folder: {e}")

    """
    # Ensure all data is removed from the vector database
    try:
        # Step 1: Check existing data before deletion
        existing_data = doc_service.get_all_entries(collection_name)
        existing_images = doc_service.get_all_entries(image_collection_name)
        
        if existing_data:
            print(f"Found {len(existing_data)} text entries in {collection_name}. Deleting...")
        if existing_images:
            print(f"Found {len(existing_images)} image metadata entries in {image_collection_name}. Deleting...")

        # Step 2: Delete both text and image metadata collections
        doc_service.delete_collection(collection_name)
        doc_service.delete_collection(image_collection_name)

        # Step 3: Verify if collections are actually deleted
        remaining_text = doc_service.get_all_entries(collection_name)
        remaining_images = doc_service.get_all_entries(image_collection_name)

        if not remaining_text and not remaining_images:
            print(f"Successfully deleted all data from vector store.")
        else:
            print(f"Warning: Some data is still present. Manual intervention may be needed.")

    except Exception as e:
        st.error(f"Error removing from vector store: {e}")
    """
    # Clear session state metadata
    keys_to_clear = ["selected_file", "conversation_history", "last_bot_question", "last_expected_answer"]
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    # Save updated list to disk
    save_processed_files(st.session_state["processed_files"])

    st.success(f"Successfully cleared all data related to {file_to_clear}.")

# --- Process Image Queries ---
def handle_image_query(image_file):
    try:
        # Open image and convert to bytes
        image = Image.open(image_file)
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        image_bytes = img_bytes.getvalue()

        # Encode image to Base64
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")

        # Initialize GPT-4o model
        chain_gpt = ChatOpenAI(model="gpt-4o")

        # Create prompt
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(content="Describe the contents of this image."),
            HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}])
        ]

        # Send image to OpenAI for description
        response = chain_gpt.invoke(prompt)

        # Extract response content
        image_description = response.content if hasattr(response, "content") else "No description available"

        # Update chat history and return result
        update_conversation_history("assistant", image_description)
        return image_description

    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        update_conversation_history("assistant", error_message)
        return error_message

def detect_image_request(user_input):
    return bool(re.search(r"image (?:at|on|in) page (\d+)", user_input.lower()))

def detect_weakness_request(user_input):
    """Detects when the user asks about their weaknesses or what they should study."""
    return any(phrase in user_input.lower() for phrase in ["my weakness", "what should i study", "what am i weak at"])

def handle_image_query(image_file):
    """Processes the uploaded image and retrieves an explanation from GPT-4o Vision."""
    
def handle_weakness_analysis():
    """Analyze user's past quiz results and determine weaknesses."""
    if "weak_topics" not in st.session_state or not st.session_state["weak_topics"]:
        update_conversation_history("assistant", "No major weaknesses detected! Keep learning and improving.")
        return

    weak_analysis = ["**Weakness Analysis:**\n"]
    topic_counts = {}

    # Track weak topics based on incorrect answers
    for entry in st.session_state["weak_topics"]:
        topic = entry["question"].split(" ")[0]  # Extract the first word as topic
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
    
    # Identify the most common weak topics
    if topic_counts:
        most_frequent_topic = max(topic_counts, key=topic_counts.get)
        weak_analysis.append(f"You frequently miss questions related to **{most_frequent_topic}**. Consider studying this area further.")

    # Show specific quiz mistakes
    weak_analysis.append("\n**Questions You Got Wrong:**\n")
    for entry in st.session_state["weak_topics"]:
        weak_analysis.append(f"**Q:** {entry['question']}\n💡 **Correct Answer:** {entry['correct_answer']}\n")

    response_text = "\n".join(weak_analysis)
    update_conversation_history("assistant", response_text)

def handle_image_request(user_input):
    match = re.search(r"image (?:at|on|in) page (\d+)", user_input.lower()) 
    page_num = int(match.group(1))
    image_data = doc_service.retrieve_image_summary(page_num)
    
    if image_data:
        response_text = f"**📷 Image Explanation (Page {image_data['page_number']})**\n\n **Summary:** {image_data['summary']}\n\n *(Stored Image Path: {image_data['image_path']})*"
    else:
        response_text = f"No image metadata found for Page {page_num}."
    
    update_conversation_history("assistant", response_text)

def detect_reference_request(user_input):
    """Dynamically determines if the user is asking for references using GPT-4."""
    classification_msgs = [
        {"role": "system", "content": "Determine if the user wants references or external sources. "
                                      "If yes, reply with 'yes'. If no, reply with 'no'."},
        {"role": "user", "content": user_input}
    ]
    
    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=classification_msgs,
            max_tokens=3,
            temperature=0
        )
        return r.choices[0].message.content.strip().lower() == "yes"
    except Exception as e:
        print(f"Error in reference detection: {e}")
        return False  # Default to false if there's an error
    
# fetch_online_references function
def fetch_online_references(query):
    """Fetch relevant online references using GPT-4 dynamically."""
    msgs = [
        {"role": "system", "content": (
            "You are an AI assistant that provides reputable online references. "
            "Find sources for the given topic and list them with their URLs. "
            "Format the response **strictly** as:\n"
            "- [Title](URL)\n"
            "- [Title](URL)\n"
            "Do not add extra explanations, introductions, or conclusions. Just return the clean list."
        )},
        {"role": "user", "content": f"Find relevant online references for: {query}"}
    ]
    
    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=msgs,
            max_tokens=400,
            temperature=0.3
        )
        
        response_text = r.choices[0].message.content.strip()
        references = []

        # Extract references using regex
        for line in response_text.split("\n"):
            match = re.match(r"- \[(.*?)\]\((.*?)\)", line)  # Matches Markdown format links
            if match:
                title, link = match.groups()
                references.append((title, link))

        return references  # Returns as many sources as found
    except Exception as e:
        print(f"Error fetching references: {e}")
        return []

# Checks if user input is a follow-up question based on the last chatbot response.
def is_follow_up(user_input: str) -> bool:
    """Determine if the user input is a follow-up question based on previous context."""
    last_bot_response = st.session_state.get("last_bot_answer", None)
    
    if not last_bot_response:
        return False  # If there's no last bot response, it's not a follow-up.

    classification_msgs = [
        {"role": "system", "content": "You analyze user queries to determine if they are follow-up questions. "
                                      "A follow-up question refers to the previous response and asks for clarification, elaboration, or continuation. "
                                      "Respond with only 'yes' or 'no'."},
        {"role": "assistant", "content": last_bot_response},  # Pass the last response
        {"role": "user", "content": user_input}
    ]

    try:
        r = client.chat.completions.create(
            model="gpt-4",
            messages=classification_msgs,
            max_tokens=5,
            temperature=0.2
        )
        return r.choices[0].message.content.strip().lower() == "yes"
    except Exception as e:
        print(f"Error in follow-up detection: {e}")
        return False  # Default to treating it as a new question

# Retrieves the most relevant document chunks from the selected file based on the query.
def retrieve_relevant_chunks(query: str, top_k: int = 20):
    """Fetch chunks from the selected file’s collection."""
    f = st.session_state["selected_file"]
    if not f:
        return ["No file selected."]
    try:
        collection_name = sanitize_collection_name(f)
        results = doc_service.retrieve_relevant_chunks(query, collection_name, top_k)
        return results if results else ["No relevant chunks found."]
    except Exception as e:
        return [f"Error retrieving chunks: {e}"]

# Function to retrieve an expected answer based on a document's content
def get_expected_answer(question: str) -> str:
    """Retrieve document chunks and generate an answer using OpenAI."""
    chunks = retrieve_relevant_chunks(question, top_k=20)
    context_text = "\n\n".join(chunks) if chunks else ""
    msgs = [
        {"role": "system", "content": (
            "You are an expert AI assistant. Answer the question using the provided context as much as possible. "
            "If the context is insufficient, generate a well-reasoned answer based on general knowledge."
        )},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {question}\nProvide only the answer."}
    ]
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=100, temperature=0.2)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return "Error retrieving expected answer."

# Function to evaluate a user's response compared to the expected answer
def evaluate_user_answer(user_answer, expected_answer: str) -> str:
    """Evaluates user response against the expected answer."""
    if isinstance(user_answer, dict):  # Ensure it's a string
        user_answer = user_answer.get("text", "").strip()  # Extract text safely

    if user_answer.lower() in ["i don't know", "idk", "not sure"]:
        return "Don't Know"
    msgs = [
        {"role": "system", "content": "You evaluate if a response is correct. Reply only with 'Correct', 'Close', or 'Incorrect'."},
        {"role": "user", "content": f"Expected: {expected_answer}\nUser: {user_answer}\nHow accurate is the user's response?"}
    ]
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=10, temperature=0.2)
        return r.choices[0].message.content.strip()
    except Exception as e:
        return "Evaluation Error"

# Estimates the confidence level of the AI-generated response using cosine similarity.
def estimate_confidence(llm_response: str, context_text: str) -> float:
    """Estimate confidence based on cosine similarity."""
    try:
        resp_emb = embedding_model.embed_query(llm_response)
        ctx_emb = embedding_model.embed_query(context_text)
        sim = cosine_similarity([resp_emb], [ctx_emb])[0][0]
        return min(100.0, max(0.0, (sim + 1) * 50))
    except Exception:
        return 0.0
    
# Function to process the user's query and generate a response
def process_response(user_input):
    """Process the user's query and generate a response with source attribution."""
    if isinstance(user_input, str):
        user_text = user_input.strip()
        user_image = None
    elif isinstance(user_input, dict):
        user_text = user_input.get("text", "").strip() if user_input.get("text") else ""
        user_image = user_input.get("image")
    else:
        return  # Unknown input format, do nothing

    if not user_text and not user_image:
        return
    
     # Process Image Query
    if user_image:
        return handle_image_query(user_image)
    
     # Process Text Query
    if user_text:
        if detect_image_request(user_text):
            return handle_image_request(user_text)

        if detect_weakness_request(user_text):
            return handle_weakness_analysis()
    
    if detect_reference_request(user_text):  
        # Determine the correct query: use last response if it's a follow-up, else use the user query.
        if is_follow_up(user_text):
            reference_query = st.session_state.get("last_bot_answer", user_text)
        else:
            reference_query = user_text  # Use the user's direct query

        # Fetch online references based on the determined query
        search_results = fetch_online_references(reference_query)

        if search_results:
            if "response_text" not in locals():
                response_text = ""  # Ensure variable exists
            response_text += "\n\n**Here are some references for further reading:**\n" + "\n".join(
                [f"- [{title}]({link})" for title, link in search_results]
            )
        else:
            response_text += "\n\nI couldn't find relevant sources online."
    
    # Check if the user's query is a follow-up question
    if is_follow_up(user_input):
        last_bot_response = st.session_state.get("last_bot_answer", "")
        if last_bot_response:
            # Construct message history to generate a follow-up response
            follow_up_msgs = [
                {"role": "system", "content": "You provide follow-up answers based on the user's previous question and your last response. "
                                              "Ensure the follow-up answer expands on the previous response in a meaningful way."},
                {"role": "assistant", "content": last_bot_response},  # Pass last answer
                {"role": "user", "content": user_input}
            ]
            try:
                # Generate follow-up response
                r = client.chat.completions.create(
                    model="gpt-4",
                    messages=follow_up_msgs,
                    max_tokens=200,
                    temperature=0.7
                )
                follow_up_response = r.choices[0].message.content.strip()
                update_conversation_history("assistant", follow_up_response)
                return  # Avoid treating it as a new question
            except Exception as e:
                update_conversation_history("assistant", f"Error in follow-up: {e}")
                return

    # General Q&A or Quiz Logic
    # Retrieve relevant document chunks to use as context for answering
    chunks = retrieve_relevant_chunks(st.session_state["selected_file"])

    # Ensure document_text is initialized
    document_text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks) if chunks else ""

    # Ensure context_str is initialized before using it
    context_str = document_text  

    if not chunks or "No relevant chunks found" in context_str or "Error" in context_str:
        context_str = ""  # Prevents errors when context is empty
        fallback_note = "\n\n**Note:** No relevant document content found; response based on general knowledge."
    else:
        fallback_note = ""
        
    # Determine if the user is requesting a quiz/test question
    classification_msgs = [
        {"role": "system", "content": "You analyze user requests to determine if they are asking to be tested on a topic."},
        {"role": "user", "content": f"Analyze this request and respond only with 'yes' or 'no': {user_input}"}
    ]
    try:
        classification_response = client.chat.completions.create(
            model="gpt-4", messages=classification_msgs, max_tokens=10, temperature=0.2
        )
        is_test_request = classification_response.choices[0].message.content.strip().lower()
    except Exception as e:
        update_conversation_history("assistant", f"Error in classification: {e}")
        return
    
    # If the user is requesting a test question, generate a quiz question
    if is_test_request == "yes":
        file_name = st.session_state.get("selected_file", None)
        if not file_name:
            update_conversation_history("assistant", "No document is currently selected. Please upload or select a file first.")
            return
        chunks = retrieve_relevant_chunks(file_name)
        document_text = "\n\n".join(chunks) if isinstance(chunks, list) else str(chunks)
        if not document_text.strip():
            update_conversation_history("assistant", "Could not extract relevant content from the document.")
            return
        msgs = [
            {"role": "system", "content": (
                "You generate a single **short-answer** question based on the provided content. "
                "Ensure the question is relevant, fact-based, and answerable from the document."
                "Do not ask about metadata such as author names, file details, document structure, or any non-content-related information."
            )},
            {"role": "user", "content": f"Generate exactly one question based on the following content:\n\n{document_text}"}
        ]
        try:
            r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=100, temperature=0.7)
            question = r.choices[0].message.content.strip().split("? ")[0] + "?"
            update_conversation_history("assistant", f"Here is your test question:\n\n**{question}**\n\nNote: I can only generate one short-answer question at a time.")
            
            # Store the generated question and its expected answer
            st.session_state["last_bot_question"] = question
            st.session_state["last_expected_answer"] = get_expected_answer(question)
        except Exception as e:
            update_conversation_history("assistant", f"Error: {e}")
        return

    # If the user is answering a quiz question, evaluate their response
    if st.session_state["last_bot_question"] and st.session_state["last_expected_answer"]:
        evaluation = evaluate_user_answer(user_text, st.session_state["last_expected_answer"])  # Use user_text here
        
        # Store incorrect answers or "idk" in weaknesses
        if evaluation.lower() == "incorrect" or user_text.lower() in ["idk", "i don't know"]:
            if "weak_topics" not in st.session_state:
                st.session_state["weak_topics"] = []
            st.session_state["weak_topics"].append({
                "question": st.session_state["last_bot_question"],
                "user_answer": user_text,  # Use user_text here
                "correct_answer": st.session_state["last_expected_answer"]
            })
        
        # Generate appropriate feedback based on evaluation result
        if evaluation == "Don't Know":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"You didn't know the answer. Here is the correct answer:\n\n**{correct_answer}**"
        elif evaluation == "Incorrect":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"**Evaluation:** {evaluation}\n\nThe correct answer is: {correct_answer}"
        elif evaluation == "Close":
            correct_answer = st.session_state["last_expected_answer"]
            feedback = f"**Evaluation:** {evaluation}\n\nThe correct answer is: {correct_answer}"
        else:
            feedback = f"**Evaluation:** {evaluation}"
        update_conversation_history("assistant", feedback)
        st.session_state["last_bot_question"] = ""
        st.session_state["last_expected_answer"] = ""
        return

    # General Q&A
    msgs = [
        {"role": "system", "content": (
            "You are a knowledgeable assistant. Use only the provided context to answer the query. "
            "If the context is empty or insufficient, indicate that the answer is based on general knowledge."
        )}
    ]
    
    # Append recent conversation history for better context
    for role, message in st.session_state["conversation_history"][-5:]:
        if "Evaluation:" not in message and "Correct Answer:" not in message:
            msgs.append({"role": role, "content": message})
    msgs.append({"role": "user", "content": f"Context:\n{context_str}\n\nQuery: {user_input}"})
    
    # Attach source attribution note
    try:
        r = client.chat.completions.create(model="gpt-4", messages=msgs, max_tokens=1500, temperature=0.7)
        answer = r.choices[0].message.content.strip()
        conf = estimate_confidence(answer, context_str) if context_str else 0.0
        
        source_note = (
            f"**Source:** Extracted from document chunks (Confidence: {conf:.2f}%)"
            if context_str else "**Source:** General knowledge (No relevant document content found)"
        )
        final_ans = f"{answer}\n\n{source_note}{fallback_note if not context_str else ''}"
        
        #Show retrieved chunks in an expander
        if context_str:
            with st.expander("View Retrieved Chunks"):
                for i, chunk in enumerate(chunks[:5]):  # Limit to 5 for brevity
                    st.write(f"Chunk {i+1}: {chunk[:200]}...")  # Truncate for display
        update_conversation_history("assistant", final_ans)
        if answer.endswith("?"):
            st.session_state["last_bot_question"] = answer
            st.session_state["last_expected_answer"] = get_expected_answer(answer)
    except Exception as e:
        update_conversation_history("assistant", f"Error: {e}")

