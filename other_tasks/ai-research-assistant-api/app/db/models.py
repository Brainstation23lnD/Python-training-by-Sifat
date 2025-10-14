from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from .base import BaseModel


class User(BaseModel):
    __tablename__ = "users"
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)


class UserProfile(BaseModel):
    __tablename__ = "user_profiles"
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    display_name = Column(String)
    preferences = Column(JSONB, default={})


class Document(BaseModel):
    __tablename__ = "documents"
    title = Column(String, nullable=False)
    source_url = Column(String)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    meta_info = Column(JSONB, default={})


class DocumentChunk(BaseModel):
    __tablename__ = "document_chunks"
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    start_pos = Column(Integer)
    end_pos = Column(Integer)
    tokens = Column(Integer)
    embedding = Column(Vector(768))
    meta_info = Column(JSONB, default={})
