from langchain.chains import RetrievalQA
from app.config import settings
from app.services.llm.llama_service import LLaMAWrapper
from app.db import vector_store

def get_retrieval_qa() -> RetrievalQA:

    # Create retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # Initialize LLaMA LLM wrapper
    llm = LLaMAWrapper(endpoint=settings.LLM_SERVER_URL)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    return qa_chain
