import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
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
    new_files = [file for file in files if file.name not in st.session_state.processed_files]

    if not new_files:
        return "⚠️ No new PDFs detected. Using existing knowledge base."

    try:
        all_docs = []
        for file in new_files:
            with open(f"./{file.name}", "wb") as f:
                f.write(file.getbuffer())  

            loader = PyPDFLoader(f"./{file.name}")
            documents = loader.load()

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            docs = text_splitter.split_documents(documents)
            all_docs.extend(docs)

        # Store embeddings in ChromaDB
        vectorstore.add_documents(all_docs)
        print(f"✅ {len(all_docs)} document chunks added to ChromaDB")
        vectorstore.persist()

        # Remember processed files
        st.session_state.processed_files.extend([file.name for file in new_files])

        return f"✅ {len(new_files)} new PDFs processed successfully!"
    except Exception as e:
        return f"❌ Error processing PDFs: {e}"

# Retrieve relevant text from ChromaDB
def retrieve_relevant_chunks(query, top_k=3):
    try:
        results = vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    except Exception as e:
        return [f"❌ Error retrieving documents: {e}"]

# Process user query
def process_query():
    user_input = st.session_state.user_query.strip()
    if user_input and user_input.lower() != "quiz":
        retrieved_chunks = retrieve_relevant_chunks(user_input)
        context = "\n\n".join(retrieved_chunks)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that answers questions."},
            {"role": "user", "content": f"Based on the following context, answer the query:\n\n{context}\n\nQuery: {user_input}"}
        ]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=300,
                temperature=0.7,
            )
            bot_response = response.choices[0].message.content.strip()
            
            # Maintain conversation history
            st.session_state.conversation_history.append((user_input, bot_response))
            st.session_state.user_query = ""  # Clear input after processing
        except Exception as e:
            st.session_state.conversation_history.append((user_input, f"❌ Error: {e}"))

# UI
st.title("🤖 Your academic weapon: SmartBot")

# Upload PDFs
uploaded_files = st.file_uploader("Upload PDFs (Multiple Supported)", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    upload_status = preprocess_pdfs(uploaded_files)
    st.success(upload_status)

# Display chat history
st.subheader("🗨️ Chat History")
for user_msg, bot_msg in st.session_state.conversation_history:
    st.write(f"**🧑‍💻 You:** {user_msg}")
    st.write(f"🤖 **ChatBot:** {bot_msg}")

st.subheader("💬 Enter your query")

# User input field (stores in session state)
st.text_input(
    "Type your question or type 'quiz' to start:",
    key="user_query",
    on_change=process_query  # Calls process_query() when Enter is pressed
)

# Check if the user wants to switch to the quiz
if "user_query" in st.session_state and st.session_state.user_query.lower() == "quiz":
    st.page_link("pages/quiz.py")

