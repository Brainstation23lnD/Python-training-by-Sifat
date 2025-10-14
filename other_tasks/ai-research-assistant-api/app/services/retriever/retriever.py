import numpy as np
from langchain_core.documents import Document as LCDocument
from langchain.schema import BaseRetriever
from pydantic import Field

from app.services import EmbeddingService
from app.services.vector_store import VectorStorage


class PostgresRetriever(BaseRetriever):
    vector_storage: VectorStorage = Field(...)
    embedding_service: EmbeddingService = Field(...)
    owner_id: int = Field(None)
    k: int = Field(default=5)
    metadata_filter: dict = Field(default_factory=dict)

    def filter_relevant_chunks(self, query_emb, chunks, threshold=0.7):
        relevant = []
        for chunk in chunks:
            if self.metadata_filter:
                match = True
                for key, value in self.metadata_filter.items():
                    if not chunk.meta_info or chunk.meta_info[key] != value:
                        match = False
                        break
                if not match:
                    continue
            chunk_emb = np.array(chunk.embedding)
            sim = np.dot(query_emb, chunk_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(chunk_emb))
            if sim >= threshold:
                relevant.append(chunk)
        return relevant

    def _get_relevant_documents(self, query: str, run_manager=None):
        query_emb = self.embedding_service.embed_texts([query])[0]
        chunks = self.vector_storage.search(query_emb, k=self.k, owner_id=self.owner_id)
        chunks = self.filter_relevant_chunks(query_emb, chunks, threshold=0.5)
        return [
            LCDocument(page_content=chunk.chunk_text, metadata={"doc_id": chunk.document_id, **(chunk.meta_info or {})})
            for chunk in chunks
        ]
