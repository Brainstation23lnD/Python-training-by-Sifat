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
        def is_metadata_match(chunk):
            if not self.metadata_filter:
                return True
            if not chunk.meta_info:
                return False
            return all(chunk.meta_info.get(k) == v for k, v in self.metadata_filter.items())

        def cosine_similarity(a, b):
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        relevant = []
        for chunk in chunks:
            if not is_metadata_match(chunk):
                continue
            if cosine_similarity(query_emb, chunk.embedding) >= threshold:
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
