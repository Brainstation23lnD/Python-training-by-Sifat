from typing import List
from PyPDF2 import PdfReader
from langchain.docstore.document import Document
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter


def load_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text: str, chunk_size:int=1000, chunk_overlap:int=200) -> List[Document]:
    spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = spliter.create_documents([text])
    return docs
