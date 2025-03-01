# services/document_service.py

import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings


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
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        :param client: An instance of openai.OpenAI (unused directly, but can be extended).
        :param persist_directory: Directory for ChromaDB to store embeddings.
        :param embedding_model_name: The model name for generating embeddings.
        :param chunk_size: Character size for each chunk of text.
        :param chunk_overlap: Overlap between consecutive chunks for better context.
        """
        self.client = client
        self.embedding_model = OpenAIEmbeddings(model=embedding_model_name)
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embedding_model
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def process_pdf(self, file_path: str):
        """
        Loads a PDF from the given path and splits it into smaller chunks (Documents).
        :param file_path: Path to the PDF file.
        :return: List of Documents.
        """
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        return self.splitter.split_documents(documents)

    def process_text(self, text: str):
        """
        Splits a raw text string into smaller chunked Documents.
        :param text: The text to split.
        :return: List of Documents.
        """
        return self.splitter.create_documents([text])

    def add_documents_to_vectorstore(self, docs):
        """
        Adds the given documents to the Chroma vector store, generating embeddings.
        :param docs: A list of Document objects (from LangChain).
        """
        self.vectorstore.add_documents(docs)

    def retrieve_relevant_chunks(self, query: str, top_k: int = 10):
        """
        Searches the vector store for the chunks most similar to the query.
        :param query: The text query.
        :param top_k: Number of top matching results to return.
        :return: List of chunk strings that are most relevant to the query.
        """
        try:
            results = self.vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            return [f"Error retrieving documents: {e}"]
