import json
import os
import shutil
import fitz  # PyMuPDF
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from PIL import Image
#import pytesseract
from langchain.schema import Document
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
import base64


class DocumentService:
    def __init__(
        self,
        client: OpenAI,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "text-embedding-3-large",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.client = client
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_vectorstore(self, collection_name: str):
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def process_pdf(self, file_path: str, collection_name: str = "default_collection"):
        """Processes a PDF file, extracting text and images, and sending images to OpenAI for descriptions."""
        
        # Create an images directory
        path = "./images"
        os.makedirs(path, exist_ok=True)
        
        # Open the PDF
        doc = fitz.open(file_path)
        text_docs = []  # Holds extracted text
        image_docs = []  # Holds extracted image descriptions
        combined_docs = []  # Links text and image metadata

        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Extract text from the page
            text_content = page.get_text("text").strip()
            if text_content:
                split_text = self.splitter.split_text(text_content)
                for chunk in split_text:
                    doc_entry = Document(
                        page_content=chunk,
                        metadata={"page_number": page_num + 1}
                    )
                    text_docs.append(doc_entry)
                    combined_docs.append(doc_entry)

            # Extract and process images
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_filename = os.path.join(path, f"page{page_num + 1}_img{img_index + 1}.{image_ext}")

                # Save image locally
                with open(image_filename, "wb") as img_file:
                    img_file.write(image_bytes)

                # Encode image to Base64
                encoded_image = base64.b64encode(image_bytes).decode("utf-8")
                chain_gpt = ChatOpenAI(model="gpt-4o")

                # Send image to OpenAI for description
                prompt = [
                    SystemMessage(content="You are a bot that is good at analyzing images."),
                    HumanMessage(content="Describe the contents of this image."),
                    HumanMessage(content=[{"type": "image_url", "image_url": {"url": f"data:image/{image_ext};base64,{encoded_image}"}}])
                ]
                response = chain_gpt.invoke(prompt)

                # Extract response content
                image_description = response.content if hasattr(response, "content") else "No description available"

                # Store image description as a document
                image_doc = Document(
                    page_content=f"Page {page_num + 1} Image {img_index + 1}: {image_description}",
                    metadata={
                        "image_path": image_filename,
                        "summary": image_description,
                        "page_number": page_num + 1
                    }
                )
                image_docs.append(image_doc)
        
        # Add text documents to vector store
        self.add_documents_to_vectorstore(text_docs, collection_name)
        
        # Add image descriptions to a separate vector store collection
        self.add_documents_to_vectorstore(image_docs, f"{collection_name}_images")
        
        return text_docs, image_docs

    def process_text(self, text: str):
        return self.splitter.create_documents([text])


    def add_documents_to_vectorstore(self, docs, collection_name: str):
        vectorstore = self.get_vectorstore(collection_name)
        vectorstore.add_documents(docs)

    def retrieve_relevant_chunks(self, query: str, collection_name: str, top_k: int = 10):
        try:
            vectorstore = self.get_vectorstore(collection_name)
            results = vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            return [f"Error retrieving documents: {e}"]
        
  
    def retrieve_image_summary(self, page_num, collection_name="default_collection"):
        """Retrieves image summaries from the vector store for a given page number."""
        
        image_collection_name = f"{collection_name}_images"  # Query only the image metadata collection
        vectorstore = self.get_vectorstore(image_collection_name)
        
        query = f"Page {page_num} Image 2"
        results = vectorstore.similarity_search(query, k=25) 
        print(f"results {results} \n")
        print(f"Retrieved {len(results)} results from vector store.")

         # Print all available metadata for debugging
        if results:
            print("**Available Metadata from Vector Store:**")
            for idx, result in enumerate(results):
                print(f"**Result {idx + 1} Metadata:** {json.dumps(result.metadata, indent=4)}")

        # Filter results to keep only relevant image documents
        filtered_results = [
            result for result in results
            if result.metadata
            and isinstance(result.metadata, dict)  # Ensure metadata exists
            and "image_path" in result.metadata 
            and "summary" in result.metadata 
            and result.metadata["summary"].strip()  # Ensure summary is not empty
            and "page_number" in result.metadata  # Ensure page number exists
            and result.metadata["page_number"] == page_num

        ]

        if not filtered_results:
            print(f"No valid image found for Page {page_num}.")
            return None

        # Select the most relevant image result
        first_result = filtered_results[0] #using the second result cuz all the first will be logos ;-;

        # Format the file path to use forward slashes for Streamlit compatibility
        formatted_path = first_result.metadata["image_path"].replace("\\", "/")

        # Clean and format metadata before returning
        formatted_metadata = {
            "image_path": formatted_path,
            "summary": first_result.metadata["summary"].strip(),  # Clean summary
            "page_number": first_result.metadata["page_number"]
        }
        print(formatted_metadata)

        return formatted_metadata

    def delete_collection(self, collection_name: str):
        try:
            collection_path = os.path.join(self.persist_directory, collection_name)
            if os.path.exists(collection_path):
                shutil.rmtree(collection_path)
                return True
            vectorstore = self.get_vectorstore(collection_name)
            vectorstore.delete_collection()
            return True
        except Exception as e:
            return False

    def list_collections(self):
        try:
            return [item for item in os.listdir(self.persist_directory) if os.path.isdir(os.path.join(self.persist_directory, item))]
        except Exception as e:
            return []
