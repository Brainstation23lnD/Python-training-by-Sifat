from langchain_postgres import PGVector
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from app.config import settings


embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="document_chunks_collection",
    connection=settings.DATABASE_URL,
    use_jsonb=True,
)



