from src.config import LABEL_MAP


class SemanticSearch:

    def __init__(
        self,
        embedding_model,
        vector_store,
        documents
    ):

        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.documents = documents

    def search(self, query, top_k=5):

        # Step 1: Convert query into embedding
        query_embedding = (
            self.embedding_model.generate_query_embedding(
                query
            )
        )

        # Step 2: Search in FAISS
        scores, indices = self.vector_store.search(
            query_embedding,
            top_k
        )

        results = []

        # Step 3: Retrieve original documents
        for score, idx in zip(scores, indices):

            if idx == -1:
                continue

            document = self.documents[int(idx)]

            results.append({
                "id": document["id"],
                "text": document["text"],
                "category": LABEL_MAP[document["label"]],
                "score": float(score)
            })

        return results