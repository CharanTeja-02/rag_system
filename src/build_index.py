import json

from src.config import (
    PROCESSED_DATA_PATH
)

from src.data_loader import download_dataset
from src.preprocessing import preprocess_documents
from src.embedding import EmbeddingModel
from src.vector_store import VectorStore


def build_index():

    # Download documents
    documents = download_dataset()

    # Preprocess documents
    documents = preprocess_documents(documents)

    # Save processed documents
    PROCESSED_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        PROCESSED_DATA_PATH / "documents.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(documents, file, ensure_ascii=False)

    # Generate embeddings
    model = EmbeddingModel()

    texts = [doc["text"] for doc in documents]

    embeddings = model.generate_embeddings(texts)

    # Build and save FAISS index
    vector_store = VectorStore()

    vector_store.build_index(embeddings)

    vector_store.save_index()

    print("\nIndexing completed successfully!")


if __name__ == "__main__":
    build_index()