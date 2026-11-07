import json
from datasets import load_dataset

from src.config import (
    DATASET_NAME,
    DOCUMENT_LIMIT,
    RAW_DATA_PATH
)


def download_dataset():

    print("Downloading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=f"train[:{DOCUMENT_LIMIT}]"
    )

    RAW_DATA_PATH.mkdir(
        parents=True,
        exist_ok=True
    )

    documents = []

    for idx, item in enumerate(dataset):

        document = {
            "id": idx,
            "text": item["text"],
            "label": item["label"]
        }

        documents.append(document)

    output_path = RAW_DATA_PATH / "ag_news.json"

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4
        )

    print(f"Downloaded {len(documents)} documents")

    return documents