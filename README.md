# 🧠 SmartBot: Because Googling Everything is Too Mainstream! 🤖

Welcome to **SmartBot**, your **AI-powered** academic assistant! This app helps you **upload PDFs, transcript and translation videos, query relevant information**, and **generate interactive quizzes** based on your study materials.  

---

## Features  
- Upload multiple PDFs and extract knowledge  
- Ask questions and get AI-generated answers  
- Generate quizzes based on uploaded documents  
- Interactive quiz mode with multiple-choice questions  
- Transcript and translate videos/audios
---
## Motivation 
With the rise of ChatGPT, many students have adopted it as a study companion. However, through deeper research, I identified several limitations:

- ChatGPT relies solely on **pretrained knowledge** rather than **Retrieval-Augmented Generation (RAG)**.
- It lacks **interactive learning features** to help students test their understanding on specific lecture materials.
  
To address these gaps, I developed **SmartBot** — an AI-powered learning assistant designed to enhance self-directed learning through *context-aware explanations*, *interactive quizzes*, and *lecture video support*.
---
## Key Features

### Context-Aware Q&A with RAG

SmartBot delivers accurate answers by grounding its responses on **user-uploaded PDFs** using RAG. This includes:

- **OpenAI GPT-4o** for advanced natural language generation.
- **ChromaDB** as the vector store, chosen over FAISS for:
  - Persistent storage
  - Metadata support
  - Optimization for RAG
- **LangChain** pipeline with:
  - `PyPDFLoader` to extract content
  - `RecursiveCharacterTextSplitter` to chunk text for better search
- **Confidence Scoring** based on cosine similarity between embeddings

---

### 🌳 Tree of Thought (ToT) Reasoning

Unlike ChatGPT's one-liner responses, SmartBot applies **Tree of Thought-style reasoning** for detailed answers:

1. **Introduction**: Introduces the key concept
2. **Detailed Explanation**: Step-by-step breakdown
3. **Real-World Analogy**: For intuitive understanding
4. **Key Takeaways**: Summarized bullet points for recall

---

## Interactive Quiz Generator

SmartBot lets you test your understanding with quizzes generated **from your own materials**.

- Adjustable **difficulty** and **number of questions**
- Randomized **MCQ options**
- Detailed feedback:
  - ✅ Why the **correct answer** is correct
  - ❌ Why **other answers** are incorrect
- Real-time scoring and final performance report

This transforms passive reading into **active learning**.

---

## Whisper-Powered Transcription & Translation

SmartBot supports **audio and video lectures** by integrating OpenAI’s **Whisper API** to:

- **Transcribe** spoken content into text
- **Translate** to support multilingual understanding

---

## 🛠️ Tech Stack

| Component         | Technology                 |
|------------------|----------------------------|
| Language Model    | OpenAI GPT-4o              |
| Embeddings        | text-embedding-3-large     |
| Speech-to-Text    | OpenAI Whisper             |
| Vector Store      | ChromaDB                   |
| Framework         | LangChain + Streamlit      |
| Deployment        | Localhost / Cloud-ready    |

---

## 🙏 Acknowledgements

This project was built with great passion and commitment to creating a helpful, real-world learning tool. It has received encouraging feedback from users.

I hope my efforts are evident in the design and implementation of **SmartBot** — and I respectfully request that this dedication be reflected in my evaluation.

---

## 🛠️ Installation  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/leenasoowq/SmartBot.git
cd SmartBot
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # MacOS/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 🔑 Environment Variables
Create a .env file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key
```

### 🏃 Running the Application
```bash
streamlit run app.py
```





---


