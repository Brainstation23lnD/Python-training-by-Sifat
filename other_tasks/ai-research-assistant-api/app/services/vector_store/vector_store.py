from sqlalchemy import Select

from sqlalchemy.orm import Session

from app.db import DocumentChunk, Document


class VectorStorage:
    def __init__(self, db: Session):
        self.db = db

    def insert_chunk(self, **kwargs):
        chunk = DocumentChunk(
            document_id=kwargs.get('document_id'),
            chunk_text=kwargs.get('chunk_text'),
            embedding=kwargs.get('embedding'),
            start_pos=kwargs.get('start_pos'),
            end_pos=kwargs.get('end_pos'),
            tokens=kwargs.get('tokens'),
            meta_info=kwargs.get('metadata'),
        )
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        return chunk

    def search(self, query_embeddings, k=5, owner_id=None, metadata_filter=None):
        q = Select(DocumentChunk).order_by(DocumentChunk.embedding.l2_distance(query_embeddings)).limit(k)
        if owner_id:
            q = q.join(Document).filter(Document.owner_id == owner_id)
        if metadata_filter:
            for key, value in metadata_filter.items():
                q = q.filter(getattr(Document, key) == value)
        return self.db.scalars(q).all()
