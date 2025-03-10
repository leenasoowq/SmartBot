# services/document_service.py

import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
import numpy as np
from tiktoken import get_encoding

class DocumentService:
    """
    Handles loading PDF documents, splitting text into chunks, and
    storing/retrieving embeddings from Chroma.
    """

    def __init__(
        self,
        client: OpenAI,
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "text-embedding-3-large",
        chunk_size: int = 500,
        chunk_overlap: int = 200,
        max_file_size_mb: int = 10
    ):
        """
        :param client: An instance of openai.OpenAI (unused directly, but can be extended).
        :param persist_directory: Directory for ChromaDB to store embeddings.
        :param embedding_model_name: The model name for generating embeddings.
        :param chunk_size: Character size for each chunk of text.
        :param chunk_overlap: Overlap between consecutive chunks for better context.
        :param max_file_size_mb: Maximum file size in MB to process.
        """
        self.client = client
        self.max_file_size_mb = max_file_size_mb

        if chunk_overlap >= chunk_size:
            print(f"Warning: chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}). Adjusting to {chunk_size // 4}.")
            chunk_overlap = chunk_size // 4  # Set overlap to 1/4 of chunk size
        
        # For embedding model, use a separate, conservative chunk size
        embedding_chunk_size = 100  # Fixed small value for embeddings
        
        self.embedding_model = OpenAIEmbeddings(
            model=embedding_model_name,
            chunk_size=embedding_chunk_size)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
        # Count token instead of characters for large files uploaded
        self.tokenizer = get_encoding("cl100k_base") 

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda text: len(self.tokenizer.encode(text))
        )

    def process_pdf(self, file_path: str):
        """
        Loads a PDF from the given path and splits it into smaller chunks (Documents).
        :param file_path: Path to the PDF file.
        :return: List of Documents.
        """
        # Check file size before processing
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({self.max_file_size_mb} MB)")
        
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.splitter.split_documents(documents)

    def process_text(self, text: str):
        """
        Splits a raw text string into smaller chunked Documents.
        :param text: The text to split.
        :return: List of Documents.
        """
        # Limit text size to prevent memory issues
        max_text_size = 1000000  # ~1MB of text
        if len(text) > max_text_size:
            print(f"Warning: Text size ({len(text)} chars) exceeds recommended size. Truncating to {max_text_size} chars.")
            text = text[:max_text_size]
            
        return self.splitter.create_documents([text])

    def add_documents_to_vectorstore(self, docs):
        """
        Adds the given documents to the Chroma vector store, generating embeddings.
        Handles batch size errors by recursively splitting documents.
        :param docs: A list of Document objects (from LangChain).
        """
        if not docs:
            return
            
        # Process documents in smaller batches
        batch_size = 10
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i+batch_size]
            retry_docs = []
            
            for doc in batch:
                try:
                    # Try to add document with original content
                    self.vectorstore.add_documents([doc])
                except ValueError as e:
                    if "batch size" in str(e).lower():
                        # If batch size error, split the content further
                        print(f"Batch size error for document, splitting further: {e}")
                        # Split the content to make it smaller
                        content = doc.page_content
                        if len(content) > 500:
                            # Create two smaller documents
                            mid = len(content) // 2
                            doc1 = doc.copy()
                            doc1.page_content = content[:mid]
                            doc2 = doc.copy()
                            doc2.page_content = content[mid:]
                            retry_docs.extend([doc1, doc2])
                        else:
                            # If content is already small but still causing issues
                            # Just take the first part that's likely to work
                            doc.page_content = content[:250]
                            retry_docs.append(doc)
                    else:
                        # For other errors, print and continue
                        print(f"Error adding document: {e}")
            
            # Process any documents that needed to be retried with smaller content
            if retry_docs:
                self._process_retry_docs(retry_docs)

    def _process_retry_docs(self, retry_docs, max_depth=3, current_depth=0):
        """
        Helper method to recursively process retry documents with smaller content.
        :param retry_docs: List of documents to retry.
        :param max_depth: Maximum recursion depth to prevent infinite loops.
        :param current_depth: Current recursion depth.
        """
        if not retry_docs or current_depth >= max_depth:
            return
            
        new_retry_docs = []
        for doc in retry_docs:
            try:
                self.vectorstore.add_documents([doc])
            except Exception as e:
                print(f"Failed with smaller content: {e}")
                # Split again if content is still too large
                content = doc.page_content
                if len(content) > 100:
                    mid = len(content) // 2
                    doc1 = doc.copy()
                    doc1.page_content = content[:mid]
                    doc2 = doc.copy()
                    doc2.page_content = content[mid:]
                    new_retry_docs.extend([doc1, doc2])
        
        # Recursively process new retry documents
        if new_retry_docs:
            self._process_retry_docs(new_retry_docs, max_depth, current_depth + 1)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 10, filter_metadata=None):
        """
        Searches the vector store for the chunks most similar to the query.
        :param query: The text query.
        :param top_k: Number of top matching results to return.
        :param filter_metadata: Metadata filter to apply (e.g., {"file_name": "example.pdf"}).
        :return: List of chunk strings that are most relevant to the query.
        """
        try:
            # Truncate query to prevent batch size issues
            safe_query = query[:100] if len(query) > 100 else query
            
            results = self.vectorstore.similarity_search(
                safe_query, k=top_k, filter=filter_metadata
            )
            return [doc.page_content for doc in results]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return [f"Error retrieving documents: {e}"]

    def safe_embed_text(self, text, max_tokens=100):
        """
        Safely embed text regardless of length by chunking and averaging embeddings.
        
        :param text: Text to embed
        :param max_tokens: Maximum tokens per chunk (conservative estimate)
        :return: Embedding vector
        """
        # For very short text, just embed directly with a safety check
        if len(text) < 100:
            return self.embedding_model.embed_query(text[:100])
        
        # For longer text, chunk it and average the embeddings
        # Estimate ~4 chars per token for English text
        max_chars = max_tokens * 4
        
        # Create chunks with reasonable overlap
        chunks = []
        for i in range(0, min(len(text), 3000), max_chars // 2):
            chunk = text[i:i + max_chars]
            if len(chunk) > 20:  # Avoid tiny chunks
                chunks.append(chunk)
        
        # If we have no valid chunks, use a small portion of the text
        if not chunks:
            chunks = [text[:100]]
        
        # Embed each chunk
        embeddings = []
        for chunk in chunks[:5]:  # Limit to first 5 chunks
            try:
                embedding = self.embedding_model.embed_query(chunk)
                embeddings.append(embedding)
            except Exception as e:
                print(f"Error embedding chunk: {e}")
                continue
        
        # Average the embeddings
        if not embeddings:
            # Last resort - try with a tiny piece of text
            return self.embedding_model.embed_query(text[:50])
        
        return np.mean(embeddings, axis=0).tolist()