from src.config import LABEL_MAP


def precision_at_k(results, relevant_label, k):

    top_results = results[:k]

    relevant_count = sum(
        1 for item in top_results
        if item["category"] == relevant_label
    )

    return relevant_count / k


def recall_at_k(
    results,
    relevant_label,
    total_relevant,
    k
):

    top_results = results[:k]

    relevant_count = sum(
        1 for item in top_results
        if item["category"] == relevant_label
    )

    if total_relevant == 0:
        return 0.0

    return relevant_count / total_relevant


def reciprocal_rank(results, relevant_label):

    for rank, item in enumerate(results, start=1):

        if item["category"] == relevant_label:

            return 1 / rank

    return 0.0


def evaluate_search(
    search_engine,
    documents,
    evaluation_queries,
    k=5
):

    precision_scores = []
    recall_scores = []
    reciprocal_ranks = []

    for query_data in evaluation_queries:

        query = query_data["query"]
        relevant_label = query_data["label"]

        results = search_engine.search(
            query,
            top_k=k
        )

        total_relevant = sum(
            1 for doc in documents
            if doc["label"] == relevant_label
        )

        precision = precision_at_k(
            results,
            LABEL_MAP[relevant_label],
            k
        )

        recall = recall_at_k(
            results,
            LABEL_MAP[relevant_label],
            total_relevant,
            k
        )

        rr = reciprocal_rank(
            results,
            LABEL_MAP[relevant_label]
        )

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_ranks.append(rr)

    return {
        "Precision@K": sum(precision_scores) / len(precision_scores),
        "Recall@K": sum(recall_scores) / len(recall_scores),
        "MRR": sum(reciprocal_ranks) / len(reciprocal_ranks)
    }