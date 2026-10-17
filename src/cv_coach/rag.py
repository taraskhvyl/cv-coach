"""CV loading, chunking and retrieval (LangChain splitter + in-memory vector store)."""
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_text(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    with open(path, encoding="utf-8") as f:
        return f.read()


def chunk(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    return RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap).split_text(text)


class Index:
    # ponytail: InMemoryVectorStore is fine for one CV; swap for Chroma/FAISS/pgvector for many docs
    def __init__(self, text: str, embeddings: Embeddings | None = None):
        self.store = InMemoryVectorStore(embeddings or OpenAIEmbeddings(model=config.EMBED_MODEL))
        self.store.add_texts(chunk(text))

    def search(self, query: str, k: int = 4) -> list[str]:
        return [d.page_content for d in self.store.similarity_search(query, k=k)]
