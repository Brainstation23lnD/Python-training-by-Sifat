import ollama
from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions


# ---- Ollama wrapper ----
def query_ollama(prompt, model="llama2"):
    response = ollama.chat(model, messages=[{
        "role": "user",
        "content": prompt,
    }])
    return response["message"]["content"].strip()

# ---- Agents ----
class ResearchAgent:
    def __init__(self, _chroma_collection):
        self.collection = _chroma_collection

    def run(self, _query):
        try:
            results = self.collection.query(
                query_texts=[_query],
                n_results=2,
                include=['distance']
            )

            threshold = 0.3
            _docs = [doc for doc, dist in zip(results["documents"][0], results["distances"][0]) if dist < threshold]
            docs_text = " ".join(_docs)
            return f"Retrieved knowledge:\n{docs_text}" if docs_text else None
        except Exception as e:
            print(f"[ResearchAgent] Query failed: {e}")
            return None

class SummarizerAgent:
    @staticmethod
    def run(text):
        prompt = f"Summarize the following information clearly:\n\n{text}"
        return query_ollama(prompt)

class AnswerAgent:
    @staticmethod
    def run(summary):
        prompt = f"Answer the following information clearly:\n{summary}"
        return query_ollama(prompt)

# ---- Supervisor ----
class Supervisor:
    def __init__(self, _research, _summarizer, _answer):
        self.research = _research
        self.summarizer = _summarizer
        self.answer = _answer

    def run(self, user_query):
        print(f"[Supervisor] Received query: {user_query}")

        # Step 1: Research
        research_output = self.research.run(user_query)
        if not research_output:
            print("[Supervisor] No results in DB -- Asking Ollama directly.")
            research_output = query_ollama(f"Answer this: {user_query}")

        # Step 2: Summarize
        summary = self.summarizer.run(research_output)
        if not summary or len(summary) < 20:
            print("[Supervisor] Summarizer failed --> Retrying..")
            summary = self.summarizer.run(research_output)

        # Step 3: Final Answer
        final_answer = self.answer.run(summary)
        return final_answer

# ---- Main ----
if __name__ == "__main__":
    # Step 1: Setup ChromaDB
    chroma_client = Client(Settings(persist_directory="./vector_db"))
    embedding_func = embedding_functions.OllamaEmbeddingFunction(model_name="nomic-embed-text")
    chroma_collection = chroma_client.get_or_create_collection("docs", embedding_function=embedding_func)

    # Step 2: Add documents
    docs = [
        {"id": "1", "text": "Python is a programming language popular for AI and web development."},
        {"id": "2", "text": "RAG stands for Retrieval-Augmented Generation, combining search with LLMs."},
        {"id": "3", "text": "Ollama allows running LLMs locally like Llama3, Mistral, etc."}
    ]
    for d in docs:
        chroma_collection.add(ids=[d["id"]], documents=[d["text"]])

    # Step 3: Initialize agents and supervisor
    research = ResearchAgent(_chroma_collection=chroma_collection)
    summarizer = SummarizerAgent()
    answer = AnswerAgent()
    supervisor = Supervisor(research, summarizer, answer)

    # Step 4: Run pipeline
    # query = "Explain multi-agent orchestration in real-world companies."
    query = "What is Python?"
    output = supervisor.run(query)
    print("\n=== Final Answer ===\n", output)
