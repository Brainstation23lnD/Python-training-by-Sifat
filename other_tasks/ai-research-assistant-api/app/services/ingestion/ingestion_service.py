from sqlalchemy.orm import Session
from app.db.models import Document
from app.services import EmbeddingService
from app.services.vector_store import VectorStorage
from app.utils import load_pdf
import re


class IngestionService:
    def __init__(self, db: Session):
        self.db = db
        self.vector_store = VectorStorage(db)
        self.embedding_service = EmbeddingService()

    @staticmethod
    def split_sections(text: str):
        pattern = re.compile(r'(Abstract|Introduction|Methods|Materials|Results|Discussion|Conclusion|References)\s*',
                             re.IGNORECASE)
        splits = pattern.split(text)

        sections = {}
        if len(splits) <= 1:
            # If no headers found, put all text under 'FullText'
            sections["FullText"] = text
        else:
            for i in range(1, len(splits), 2):
                section_name = splits[i].strip()
                section_text = splits[i + 1].strip()
                sections[section_name] = section_text
        return sections

    @staticmethod
    def chunk_paragraphs(section_text: str):
        paragraphs = [p.strip() for p in section_text.split("\n\n") if len(p.strip()) > 0]
        return paragraphs

    def ingest_pdf(self, file_path: str, title: str, owner_id: int):
        doc = Document(title=title, owner_id=owner_id)
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)

        text = load_pdf(file_path)

        sections = self.split_sections(text)

        for section_name, section_text in sections.items():
            paragraphs = self.chunk_paragraphs(section_text)
            for idx, paragraph in enumerate(paragraphs):
                if len(paragraph) == 0:
                    continue
                emb = self.embedding_service.embed_texts([paragraph])[0]
                self.vector_store.insert_chunk(
                    document_id=doc.id,
                    chunk_text=paragraph,
                    embedding=emb,
                    start_pos=0,
                    end_pos=len(paragraph),
                    tokens=len(paragraph.split()),
                    metadata={
                        "title": title,
                        "section": section_name,
                        "paragraph_index": idx
                    },
                )
        return doc
