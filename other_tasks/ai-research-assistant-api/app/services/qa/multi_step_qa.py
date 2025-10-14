from typing import List, TypedDict

from app.config import settings
from app.services import LLaMAWrapper, EmbeddingService
from app.services.retriever import PostgresRetriever
from app.services.vector_store import VectorStorage


class State(TypedDict):
    query: str
    docs: List[str]
    summary: str

def retrieve_documents(state: State, vector_storage, owner_id: int) -> State:
    retriever = PostgresRetriever(
        vector_storage=vector_storage,
        embedding_service=EmbeddingService(),
        owner_id=owner_id,
        k=5
    )
    docs = retriever.invoke(state["query"])
    state["docs"] = docs
    return state

def summarize_documents(state: State) -> State:
    llm = LLaMAWrapper(endpoint=settings.LLM_SERVER_URL)

    if state.get("docs"):
        context = "\n".join([d.page_content for d in state["docs"]])
        prompt = f"Summarize the following:\n{context}"
    else:
        prompt = f"Answer the following query:\n{state['query']}"
    summary = llm._call(prompt)
    state["summary"] = summary
    return state


def multiline_step_qa(query: str, db, owner_id: int) -> State:
    state: State = {"query": query, "docs": [], "summary": ""}
    vector_storage = VectorStorage(db)
    state = retrieve_documents(state, vector_storage, owner_id=owner_id)
    state = summarize_documents(state)
    return state
