# config.py example

VECTOR_DB_DIR = "./vector_db"
RAG_CONFIG = {
    "collection_name": "textbook_knowledge_base",
    "chunk_size": 1000,         # fixed chunk size in characters
    "chunk_overlap": 200,       # overlap size in characters
    "top_k": 5
}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
