import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

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
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def get_vectorstore(self, collection_name: str):
        """Returns a Chroma vector store instance for a specific collection."""
        return Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def process_pdf(self, file_path: str):
        """Loads a PDF and splits it into chunks."""
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        print(f"Loaded {len(documents)} pages from {file_path}")
        for i, doc in enumerate(documents[:5]):
            print(f"Page {i+1} (first 500 chars): {doc.page_content[:500]}")
        split_docs = self.splitter.split_documents(documents)
        print(f"Split into {len(split_docs)} chunks")
        return split_docs

    def process_text(self, text: str):
        """Splits raw text into chunks."""
        return self.splitter.create_documents([text])

    def add_documents_to_vectorstore(self, docs, collection_name: str):
        """Adds documents to a specific collection in Chroma."""
        vectorstore = self.get_vectorstore(collection_name)
        print(f"Adding {len(docs)} documents to collection '{collection_name}'")
        for i, doc in enumerate(docs[:5]):
            print(f"Chunk {i+1}: {doc.page_content[:500]}")
        vectorstore.add_documents(docs)

    def retrieve_relevant_chunks(self, query: str, collection_name: str, top_k: int = 10):
        """Searches a specific collection for relevant chunks."""
        try:
            vectorstore = self.get_vectorstore(collection_name)
            results = vectorstore.similarity_search(query, k=top_k)
            return [doc.page_content for doc in results]
        except Exception as e:
            return [f"Error retrieving documents: {e}"]
