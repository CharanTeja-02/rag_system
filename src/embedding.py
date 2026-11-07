from sentence_transformers import SentenceTransformer

from src.config import MODEL_NAME


class EmbeddingModel:

    def __init__(self):

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

    def generate_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        return embeddings

    def generate_query_embedding(self, query):

        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding