# Semantic Search Engine – Dense Retrieval

## Overview

A modular semantic search engine that retrieves relevant documents based on their semantic meaning rather than exact keyword matching.

Built using Sentence Transformers and FAISS.

## Features

- Text preprocessing and document management.
- Dense vector embeddings using all-MiniLM-L6-v2.
- 384-dimensional sentence embeddings.
- FAISS-based similarity search.
- Cosine similarity retrieval.
- Top-K document retrieval.
- Precision@K, Recall@K and MRR evaluation.
- Modular Python architecture.

## Tech Stack

- Python
- Sentence Transformers
- FAISS
- NumPy
- Hugging Face Datasets
- Scikit-learn

## Dataset

AG News dataset – 10,000 news documents across four categories.

## Project Structure

```text
semantic-search-engine/
├── data/
├── indexes/
├── src/
├── tests/
├── build_index.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd semantic-search-engine
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Build the vector index:

```bash
python build_index.py
```

Start semantic search:

```bash
python main.py
```

## Evaluation

The system evaluates retrieval performance using:

- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)

The initial benchmark uses document categories as relevance labels.

## Future Improvements

- Implement BM25 keyword retrieval.
- Compare dense retrieval against keyword retrieval.
- Experiment with different embedding models.
- Implement hybrid search.
- Add a FastAPI backend.
- Build a Streamlit user interface.
- Experiment with larger datasets and approximate FAISS indexes.

## License

For educational and learning purposes.