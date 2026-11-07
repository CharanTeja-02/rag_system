import re


def clean_text(text):

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


def preprocess_documents(documents):

    processed_documents = []

    for document in documents:

        cleaned_text = clean_text(
            document["text"]
        )

        if not cleaned_text:
            continue

        processed_documents.append({
            "id": document["id"],
            "text": cleaned_text,
            "label": document["label"]
        })

    return processed_documents