import getpass
import os

import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent import DOC_PATHS, DOCS_BASE


class RAG:
    def __init__(self):
        load_dotenv()
        self.embeddings = self.model()
        self.vector_store = InMemoryVectorStore(self.embeddings)

    @staticmethod
    def model() -> GoogleGenerativeAIEmbeddings:
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass(
                "Enter API key for Google Gemini: "
            )
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

    @staticmethod
    def load_langchain_docs(doc_paths: list[str] | None = None) -> list[Document]:
        paths = doc_paths or DOC_PATHS
        docs: list[Document] = []
        for path in paths:
            url = f"{DOCS_BASE}/{path}.md"
            try:
                response = requests.get(url, timeout=20)
                response.raise_for_status()
            except requests.RequestException:
                continue
            source = f"{DOCS_BASE}/{path}"
            docs.append(
                Document(page_content=response.text, metadata={"source": source})
            )
        return docs

    @staticmethod
    def split_docs(
        docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 0
    ) -> list[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(docs)

    def doc_process(self, doc_paths: list[str] | None = None) -> int:
        chunks = self.split_docs(self.load_langchain_docs(doc_paths))
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def search(self, query: str, k: int = 4) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    rag = RAG()
    print(f"Indexed {rag.doc_process()} chunks")
    for doc in rag.search("How do I create an agent?"):
        print(doc.metadata["source"], "->", doc.page_content[:100].replace("\n", " "))
