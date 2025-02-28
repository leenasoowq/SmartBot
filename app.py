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

# Process Multiple PDFs
def preprocess_pdfs(files):
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
        return "⚠️ No new PDFs detected. Using existing knowledge base."

    try:
        all_docs = []
        for file in files:
            file_path = os.path.join(upload_dir, file.name)  # Store in uploads/ folder

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Ensure text is split properly
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)

        # Store embeddings in ChromaDB
        vectorstore.add_documents(all_docs)
        # vectorstore.persist()

        # Update processed files in session state
        st.session_state.processed_files.extend(new_files)

        return f"{len(new_files)} new PDFs processed successfully!"
    except Exception as e:
        return f"Error processing PDFs: {e}"

# Retrieve relevant text from ChromaDB
def retrieve_relevant_chunks(query, top_k=20):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"Error retrieving documents: {e}"]

# Process user query using structured ToT reasoning
def process_query():
    user_input = st.session_state.user_query.strip()

    if user_input.lower() == "quiz":
        st.session_state.quiz_mode = True  # Enable quiz mode
        st.switch_page("pages/quiz.py")
        return  # Stop further processing

    elif user_input.lower() == "summarize":
        if not st.session_state.processed_files:
            st.session_state.conversation_history.append(("summarize", "No PDFs have been uploaded. Please upload a file first."))
            return

        summaries = []
        for file in st.session_state.processed_files:
            retrieved_chunks = retrieve_relevant_chunks(file)  # Retrieve relevant content per PDF
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

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs (Multiple Supported)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    upload_status = preprocess_pdfs(uploaded_files)
    st.success(upload_status)

# Dropdown to show uploaded PDFs
if st.session_state.processed_files:
    selected_pdf = st.selectbox("Uploaded PDFs:", st.session_state.processed_files)
    # st.write(f"**Currently selected:** {selected_pdf}")

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