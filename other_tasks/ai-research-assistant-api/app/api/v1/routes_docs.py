import shutil
import tempfile
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.orm import Session
from app.db.base import get_db
from app.services import IngestionService
from app.schemas import DocumentRead

document_router = APIRouter()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit


@document_router.post("/upload", response_model=DocumentRead)
def upload_pdf(file: UploadFile = File(...), title: str = "",
               db: Session = Depends(get_db)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF allowed.")

    # Validate file size
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)  # Reset pointer to beginning
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File too large. Max 10 MB allowed.")

    ingestion_service = IngestionService(db)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_file_path = tmp_file.name

    doc = ingestion_service.ingest_pdf(tmp_file_path, title or file.filename, 1)

    return doc
