Hi people 

1. Install Dependencies: Run the following command to install the required libraries:
pip install -r requirements.txt

2. Set up the openai api key (probably later part of the project we can start sharing api accounts so we can utilise to $300)
   1. Get Your OpenAI API Key:
      
      Sign up or log in to OpenAI at https://platform.openai.com/.
      Go to the API Keys section and generate a new API key.
      Create a .env File:

      In your project folder, create a file named .env.
      Add the following content to the file:
      OPENAI_API_KEY=your_openai_api_key
      Replace your_openai_api_key with your actual OpenAI API key.

3. To run the code in terminal:
   python app.py

4. Brief Summary:
   1. Input: PDF file, text file, URL
   2. set number of quiz questions
   3. set difficulty level









## **Objective of the Project**
The project aims to apply classroom knowledge to real-world problems. Through this project, you will:

- Practice analyzing problems, selecting appropriate models, and evaluating their performance.
- Enhance programming skills by implementing solutions.
- Develop communication skills by presenting your work clearly and effectively.
- Learn to manage challenges encountered in projects, such as discrepancies between plans and reality or coordinating with teammates.

---

## **Timeline and Deliverables**

| **Deliverable**       | **Due Date**           |
|-----------------------|------------------------|
| Project Proposal      | Week 3 Sunday Midnight |
| Progressive Report    | Week 8 Sunday Midnight |
| Source Code           | Week 11 Sunday Midnight |
| Presentation/Demo     | Week 11 Friday Lab Time |
| Peer Review           | Week 12 Sunday Midnight |

---

## **Report Preparation/Submission Instruction**

### **How to Submit:**
- Submit your final presentation slides and code in **xSITE Dropbox**.

### **Project Proposal:**
- You are required to submit a project proposal to help the evaluators understand your project and provide feedback if necessary.
- Your proposal should clearly define the project, outline the data to be used, and describe the expected outcomes.
- **There is no template.** The proposal should be concise.
- The proposal must contain task allocation and a plan. **Each student must contribute to the coding of the project.**
- The proposal accounts for **10% of the total score**. It is a binary evaluation: either submitted or not submitted.

### **Progress Report:**
- A progress report helps you assess the current status of your project and provides a clear overview to the evaluators, who may offer feedback if necessary.
- Your progress report should include both your project plan and the actual progress made so far. Allow modifications to your plan.
- **There is no template.** The report should be concise.
- You may present the information in a **table format**.
- The progress report accounts for **10% of the total score**. It is a binary evaluation: either submitted or not submitted.

---

## **Late Submission Policy**
- A **penalty of 20% per day** for each deliverable will be imposed for late submission unless an extension has been granted prior to the submission date.
- Requests for extensions will be granted on a **case-by-case basis**.
- Any work submitted **more than 4 days after the submission date will not be accepted** and no mark will be awarded.

---

## **Rubrics**

### **Project Quality (40%)**

| Weightage (%)  | Excellent (10-9)                                           | Good (8.9-7)                                       | Average (6.9-6)                                   | Fail to Meet Expectations (5.9-0)               |
|----------------|------------------------------------------------------------|---------------------------------------------------|--------------------------------------------------|-------------------------------------------------|
| Functionality   | All features are fully implemented and work correctly without any bugs or errors. | Most features are implemented, with minor bugs that do not affect core functionality. | Basic features are implemented but some are incomplete or buggy. | The project fails to achieve its intended purpose. |
| Innovation and Creativity | Demonstrates exceptional creativity and originality by presenting innovative ideas and solutions. | Presents strong and creative ideas that offer valuable insights or improvements. | Provides some new ideas or solutions, but they may be limited in scope or impact. | Lacks originality and creativity in ideas or solutions. |

---

### **Code Quality (20%)**

| Weightage (%) | Excellent (10-9)                                      | Good (8.9-7)                                     | Average (6.9-6)                                   | Fail to Meet Expectations (5.9-0)               |
|---------------|-------------------------------------------------------|-------------------------------------------------|--------------------------------------------------|-------------------------------------------------|
| Data Structures, Modularization, Comments, Basic Control Flow, Variables | Well-structured, efficient code with clear comments and proper use of data structures and control flow. | Organized and mostly efficient code with minor readability or structure issues. | Functional but disorganized code with minimal comments and some control flow issues. | Poorly organized code with frequent errors and inadequate comments. |

---

### **Presentation (20%)**

| Weightage (%)  | Excellent (10-9)                                           | Good (8.9-7)                                      | Average (6.9-6)                                   | Fail to Meet Expectations (5.9-0)               |
|----------------|------------------------------------------------------------|--------------------------------------------------|--------------------------------------------------|-------------------------------------------------|
| Presentation Quality | Clear, engaging, and well-structured presentation; all key points covered. | Clear presentation with minor gaps or redundancies. | Adequate presentation but lacks depth or clarity in some areas. | Presentation is incomplete or difficult to follow. |
| Use of Advanced Tools like ChatGPT | Provides a detailed and insightful explanation of how ChatGPT or similar tools were used. Clearly identifies the tool's biggest advantages and demonstrates a strong understanding of its application. | Offers a solid explanation of the tool's use, with good identification of its advantages. | Describes basic usage of the tool with limited discussion of its advantages. | Provides an inadequate or unclear description of how the tool was used. |

---

## **Project Idea Samples**

Here are some project ideas for your teams to select. You are welcome to propose your own project ideas.

### **Idea 1: Speech De-identification using ASR + NER**

Speech de-identification (speech de-id) is the process of removing or masking sensitive information, such as names, dates, or locations, from spoken audio. This technology is crucial in industries like healthcare, customer service, and legal services, where conversations often contain personally identifiable information (PII). By ensuring privacy and compliance with regulations such as GDPR and HIPAA, speech de-id plays a vital role in handling sensitive audio data securely.

**Steps:**
1. **Transcription**: Use Automatic Speech Recognition (ASR) to transcribe the spoken audio into text.
2. **Entity Identification**: Apply Named Entity Recognition (NER) to the transcribed text to identify sensitive entities.
3. **Mapping to Timestamps**: Leverage the word-level timestamps provided by ASR to map the identified entities back to their corresponding locations in the original audio.
4. **Audio Modification**: Modify the audio to mask or replace the sensitive content, resulting in a de-identified output that ensures privacy while preserving the overall context.

---

### **Idea 2: Video Speech Translation**

Have you ever struggled to understand a lecture because of the lecturer’s accent? Or come across a fantastic educational video, only to find it’s in a language you don’t understand? This project aims to make learning more accessible and personalized by translating videos into your preferred language.

**Steps:**
1. **ASR (Automatic Speech Recognition)**: Recognize and transcribe the spoken content from the teaching video.
2. **LLM (Large Language Model)**: Translate the content into the target language.
3. **TTS (Text-to-Speech)**: Synthesize the translated text into natural-sounding speech in the target language.
4. **Video Integration**: Align the new audio with the visuals and adapt or replace slides if necessary.

---

### **Idea 3: Personalized NLP Learning Assistant using RAG**

This project aims to build a personalized NLP learning assistant using Retrieval-Augmented Generation (RAG) technology.

**Steps:**
1. **Content Preparation**: Store course materials, lecture notes, and references in a structured database.
2. **RAG Framework**:
   - Use retrieval techniques to search for relevant content based on a query.
   - Combine the retrieved information with LLM reasoning and generation capabilities.
3. **Dialog System**: Enable interactive learning, allowing students to ask follow-up questions.
4. **Personalization**: Tailor recommendations based on individual learning preferences.



