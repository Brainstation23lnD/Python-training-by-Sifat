from fastapi import APIRouter, Depends, UploadFile, File
from sqlalchemy.orm import Session
from app.api.v1.deps import get_current_user
from app.db.base import get_db
from app.services import IngestionService
from app.schemas import DocumentRead

document_router = APIRouter()

@document_router.post("/upload", response_model=DocumentRead)
def upload_pdf(file: UploadFile = File(...), title: str = "",
               # current_user=Depends(get_current_user),
               db: Session = Depends(get_db)):
    ingestion_service = IngestionService(db)
    file_path = f"/tmp/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(file.file.read())
    doc = ingestion_service.ingest_pdf(file_path, title or file.filename, 1)
    return doc
