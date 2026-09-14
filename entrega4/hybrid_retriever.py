import os
from pathlib import Path
from typing import List, Dict
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever

from ingest import obtener_chunks, INDEX_NAME, NAMESPACE, EMBEDDING_MODEL

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


class RAGSystem:
    """Encapsula el EnsembleRetriever y expone un método simple para obtener el top-k."""

    def __init__(self, retriever, k: int = 5):
        self.retriever = retriever
        self.k = k

    def obtener_top_k(self, query: str) -> List[Dict]:
        docs = self.retriever.invoke(query)[: self.k]
        return [
            {
                "contenido": d.page_content,
                "fuente": d.metadata.get("source", "desconocida"),
                "categoria": d.metadata.get("categoria", "desconocida"),
            }
            for d in docs
        ]


def build_rag_system(k: int = 5) -> RAGSystem:
    chunks = obtener_chunks()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings,
        namespace=NAMESPACE,
    )

    retriever_vectorial = vectorstore.as_retriever(
        search_kwargs={"k": k, "namespace": NAMESPACE},
    )

    retriever_bm25 = BM25Retriever.from_documents(chunks)
    retriever_bm25.k = k

    retriever_hibrido = EnsembleRetriever(
        retrievers=[retriever_bm25, retriever_vectorial],
        weights=[0.5, 0.5],
    )

    return RAGSystem(retriever_hibrido, k=k)