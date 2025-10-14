# 🧠 AI Research Assistant API

**An intelligent research assistant backend built with FastAPI, LangGraph, PostgresSQL (pgvector), and Ollama.**  
This system allows users to upload research papers, build personalized knowledge bases, and chat with an AI that remembers, retrieves, and reasons intelligently.

---

## 🚀 Overview

This project simulates how a **research-oriented agentic AI system** integrates structured memory, knowledge retrieval, and natural language reasoning.

It combines:
- 🧩 **FastAPI** for scalable API architecture  
- 🧠 **LangGraph** for agentic workflow orchestration  
- 🪣 **PostgresSQL + pgvector** for embedding-based retrieval  
- 🤖 **Ollama (Zephyr-7B)** for natural language reasoning  
- 📚 **LangChain integration** for document parsing, embeddings, and RAG (Retrieval-Augmented Generation)

---

## 🧱 System Architecture
        ┌──────────────────────────────────────────────────────┐
        │                     User Interface                   │
        └───────────────┬───────────────────────────────┬──────┘
                        │                               │
                        ▼                               ▼
     ┌────────────────────────────-┐    ┌─────────────────────────────┐
     │      Authentication API     │    │        Document API         │ 
     │ (User registration/login)   │    │  (Upload, Ingest, Embed)    │
     └──────────────┬──────────────┘    └──────────────┬─────────────-┘
                    │                                  │
                    ▼                                  ▼
     ┌───────────────────────────-─┐    ┌─────────────────────────────┐
     │   Vector Database (pgvector)│    │    LangGraph Agent Manager  │
     │ Stores user embeddings,     │    │ Controls retrieval + LLM    │
     │ paper vectors, and memory   │    │ orchestration (RAG flow)    │
     └──────────────┬──────────────┘    └-──────────────┬─────────────┘
                    │                                   │
                    ▼                                   ▼
     ┌────────────────────────────┐     ┌─────────────────────────────┐
     │  LangChain Retriever/Tools │     │   Ollama LLM / Zephyr Model │
     │ (Text parsing, search)     │     │ (Response generation)       │
     └────────────────────────────┘     └─────────────────────────────┘

---

## ✨ Core Features

| Feature                       | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| 🧑‍💻 **User Authentication** | JWT-based secure user registration and login                     |
| 📄 **PDF Ingestion**          | Upload and extract text content from research papers             |
| 🧬 **Vector Embeddings**      | Store document chunks in pgvector for fast semantic search       |
| 🔍 **RAG Pipeline**           | Retrieve relevant chunks before querying the LLM                 |
| 🧠 **Agentic Workflow**       | Built with LangGraph for multi-step reasoning and fallback logic |
| 💬 **Chat Interface API**     | Query knowledge base or directly the model                       |
| ☁️ **Containerized Infra**    | Docker + Kubernetes (Helm-ready) for scalable deployment         |

---

## 🧩 Project Structure
```ai-research-assistant-api/
├── app/
│ ├── api/v1/ # FastAPI route definitions
│ ├── chats/ # Chat models and services
│ ├── db/ # SQLAlchemy models, migrations, vector_db
│ ├── schemas/ # Pydantic schemas
│ ├── services/ # Core services: LLM, RAG, embeddings, retrieval
│ ├── utils/ # Helpers (PDF parsing, metrics)
│ └── main.py # FastAPI entry point
├── infra/ # K8s manifests and Helm templates
├── llm_model/ # Local GGUF models (Zephyr/Ollama)
├── scripts/ # Setup and ingestion scripts
├── alembic/ # Database migrations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```


---

## 🧠 How It Works

1. **User Uploads Papers**  
   → PDFs are parsed into text chunks using `utils/pdf.py`.

2. **Embeddings Generation**  
   → Text chunks are embedded via LangChain’s embedding model.

3. **Storage**  
   → Embeddings + metadata stored in PostgreSQL with `pgvector`.

4. **Query Phase**  
   → When a user asks a question:  
      - Retrieve top relevant chunks from pgvector.  
      - If no relevant context found → fallback to direct LLM reasoning.

5. **Response Generation**  
   → Context + query fed into Ollama LLM (Zephyr-7B) for coherent, research-grounded output.

---

## ⚙️ Technologies Used

| Layer             | Technology               |
|-------------------|--------------------------|
| **API Framework** | FastAPI                  |
| **Database**      | PostgresSQL + pgvector   |
| **Vector Search** | LangChain Retriever      |
| **LLM**           | Ollama (Zephyr-7B)       |
| **Orchestration** | LangGraph                |
| **Infra**         | Docker, Helm, Kubernetes |
| **Migrations**    | Alembic                  |
| **Auth**          | JWT, FastAPI middleware  |

---

## 🧩 Example Use Case

1. **Upload:**  
   Researcher uploads `company_12month_demo.pdf`.

2. **Ask Question:**  
   “What are the key metrics mentioned in the 12-month financial report?”

3. **Response Flow:**  
   → System retrieves relevant context chunks → LLM summarizes key insights with citations.

---

## 🧭 Future Scope

- 🌐 Add **multi-user workspace collaboration**  
- 🧩 Integrate **external paper sources (ArXiv, Semantic Scholar)**  
- 🧠 Add **LangGraph-based multi-agent reasoning** (e.g., Critic/Researcher/Planner agents)  
- 💾 Implement **Redis or Milvus** for distributed memory  
- 🛠 Add **monitoring dashboard** for query analytics and model metrics  
- ☁️ Deploy via **Kubernetes with autoscaling** (included in `infra/k8s`)

---

## 🧰 Setup Instructions

### Clone the Repo
```bash
git clone https://github.com/yourusername/ai-research-assistant-api.git
cd ai-research-assistant-api
```

### Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Database Migrations
```bash
alembic upgrade head
```

### Run application
```bash
uvicorn app.main:app --reload
```


