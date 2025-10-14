from .routes_users import user_router
from .routes_chat import chat_router
from .routes_docs import document_router

__all__ = [
    "chat_router",
    "user_router",
    "document_router"
]