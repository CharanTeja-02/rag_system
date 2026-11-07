import json

from src.config import (
    PROCESSED_DATA_PATH,
    TOP_K
)

from src.embedding import EmbeddingModel
from src.vector_store import VectorStore
from src.search import SemanticSearch


def main():

    # Load processed documents
    with open(
        PROCESSED_DATA_PATH / "documents.json",
        "r",
        encoding="utf-8"
    ) as file:

        documents = json.load(file)

    # Load embedding model
    embedding_model = EmbeddingModel()

    # Load existing FAISS index
    vector_store = VectorStore()

    vector_store.load_index()

    # Initialize search
    search_engine = SemanticSearch(
        embedding_model,
        vector_store,
        documents
    )

    print("\nSemantic Search Engine Ready!")

    while True:

        query = input("\nEnter query (or exit): ")

        if query.lower() == "exit":
            break

        results = search_engine.search(
            query,
            TOP_K
        )

        for rank, result in enumerate(results, start=1):

            print(f"\nRank: {rank}")
            print(f"Category: {result['category']}")
            print(f"Similarity: {result['score']:.4f}")
            print(f"Text: {result['text'][:300]}")


if __name__ == "__main__":
    main()