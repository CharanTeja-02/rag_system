import faiss
import numpy as np

from src.config import (
    EMBEDDING_DIMENSION,
    INDEX_PATH
)


class VectorStore:

    def __init__(self):

        self.index = None

    def build_index(self, embeddings):

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        self.index = faiss.IndexFlatIP(
            EMBEDDING_DIMENSION
        )

        self.index.add(embeddings)

        print(
            f"FAISS index created with "
            f"{self.index.ntotal} vectors"
        )

    def save_index(self):

        INDEX_PATH.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        print("FAISS index saved")

    def load_index(self):

        self.index = faiss.read_index(
            str(INDEX_PATH)
        )

        print("FAISS index loaded")

    def search(self, query_embedding, top_k=5):

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        return scores[0], indices[0]