from sqlalchemy.orm import Session
from app.db.models import Document
from app.services import EmbeddingService
from app.services.vector_store import VectorStorage
from app.utils import load_pdf, chunk_text




class IngestionService:
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStorage(db)
        self.embedding_service = EmbeddingService()

    def ingest_pdf(self, file_path: str, title: str, owner_id: int):
        doc = Document(title=title, owner_id=owner_id)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        # Load + chunk pdf
        text = load_pdf(file_path)
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            emb = self.embedding_service.embed_texts([chunk.page_content])[0]
            self.vector_store.insert_chunk(
                document_id=doc.id,
                chunk_text=chunk.page_content,
                embedding=emb,
                start_pos=0,
                end_pos=len(chunk.page_content),
                tokens=len(chunk.page_content.split()),
                metadata={"title": title},
            )
        return doc
