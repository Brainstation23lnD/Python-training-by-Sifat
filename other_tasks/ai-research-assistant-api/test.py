import os

# Define project structure
structure = {
    "app": {
        "main.py": "",
        "config.py": "",
        "api": {
            "v1": {
                "__init__.py": "",
                "routes_docs.py": "",
                "routes_chat.py": "",
                "routes_admin.py": "",
                "deps.py": ""
            }
        },
        "services": {
            "embedding_service.py": "",
            "vector_store.py": "",
            "retriever.py": "",
            "rag_service.py": "",
            "model_service.py": "",
            "ingestion.py": ""
        },
        "db": {
            "base.py": "",
            "models.py": "",
            "migrations": {}
        },
        "schemas": {},
        "workers": {},
        "utils": {
            "text.py": "",
            "pdf.py": "",
            "metrics.py": ""
        },
        "tests": {
            "unit": {},
            "integration": {}
        }
    },
    "docker": {
        "Dockerfile.api": "",
        "Dockerfile.ingest": ""
    },
    "infra": {
        "k8s": {
            "deployment-api.yaml": "",
            "deployment-llm.yaml": "",
            "hpa.yaml": ""
        },
        "helm": {}
    },
    "scripts": {
        "setup_pgvector.sh": "",
        "ingest_sample.sh": ""
    },
    "requirements.txt": "",
    "pyproject.toml": "",
    "README.md": ""
}

def create_structure(base_path, structure):
    for name, content in structure.items():
        path = os.path.join(base_path, name)
        if isinstance(content, dict):
            os.makedirs(path, exist_ok=True)
            create_structure(path, content)
        else:
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write(content)

# Run from root folder
create_structure(".", structure)
print("Project scaffold created (missing files only).")
