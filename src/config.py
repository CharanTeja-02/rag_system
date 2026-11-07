from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset paths
RAW_DATA_PATH = BASE_DIR / "data" / "raw"
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"

DOCUMENTS_PATH = PROCESSED_DATA_PATH / "documents.json"

# FAISS index path
INDEX_PATH = BASE_DIR / "indexes" / "faiss.index"

# Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Model configuration
EMBEDDING_DIMENSION = 384

# Dataset configuration
DATASET_NAME = "fancyzhx/ag_news"
DOCUMENT_LIMIT = 10000

# Retrieval configuration
TOP_K = 5

# Categories
LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


EVALUATION_QUERIES = [
    {
        "query": "Latest developments in international politics",
        "label": 0
    },
    {
        "query": "Football tournaments and players",
        "label": 1
    },
    {
        "query": "Stock market and company profits",
        "label": 2
    },
    {
        "query": "New developments in computer technology",
        "label": 3
    },
    {
        "query": "Scientific discoveries and research",
        "label": 3
    }
]