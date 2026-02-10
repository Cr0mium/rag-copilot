# RAG Python Helper (Retrieval-Only)

A lightweight CLI tool for **retrieval-augmented generation (RAG)** using FAISS and hybrid search.

Quickly retrieve top relevant answers from your knowledge base for Python errors, debugging questions, or general programming queries.

---

## Features

- **Hybrid Search**: Combines dense (embedding-based) and sparse (BM25) retrieval for high relevance.
- **Top-K Results**: Retrieve the most relevant answers with confidence scores.
- **Contextual Output**: Displays question, answer, source, and score clearly in the CLI.
- **Retrieval-Only Mode**: Focused on providing references and context before any generation.
- **Lightweight & Fast**: Uses FAISS vector store for quick semantic search.

---

## Installation

1. **Clone the repository**:

```bash
git clone <https://github.com/yourusername/rag-copilot.git>
cd rag-copilot
```

2. **Set up your Python environment** (recommended with Miniconda):

```python
conda create -n rag-copilot python=3.10
conda activate rag-copilot
```

3. **Install requirements**:

```python
pip install -r requirements.txt
```

4. **Ensure FAISS vectors are loaded**:

- Make sure your embeddings directory exists with precomputed vectors
- Example: embeddings/faiss.index

---

## **Usage**

Run the CLI tool:

```bash
python src/genereate_answers.py
```

**Commands:**

- Type your Python error or query (e.g., list index out of range).
- Press **Enter** to retrieve top relevant results.
- Type exit or quit to leave the CLI.

**Example Session:**

```bash
>> Enter error/query: list index out of range

=== Result ===
Query      : list index out of range
Confidence : high
Explanation: Retrieved top relevant results

Top Results:
------------------------------------------------------------
Source: stackoverflow
QID: 4788445 | AID: 4788460 | Score: 1.300
Answer: Title: list index out of range
...

Sources:
  QID: 4788445 | AID: 4788460 | Score: 1.300
  QID: 2918243 | AID: 2918291 | Score: 0.300
  QID: 6317287 | AID: 6317324 | Score: 0.208
```

---

## **Project Structure**

```bash
rag-copilot/
│
├─ src/
│  ├─ genereate_answers.py     # CLI entrypoint & retrieval logic
│  ├─ query_retrival.py        # Hybrid search functions
│  ├─ vector_persist.py        # FAISS vector store management
│  └─ embedding.py             # Embedding model wrapper
│
├─ embeddings/                 # FAISS and BM25 index files
│
├─ requirements.txt
└─ README.md
```

## **Configuration**

- **TOP_K**: Number of top results returned per query (3 by default).
- **MAX_CONTEXT_CHARS**: Maximum characters to include in context display (1500 by default).

You can modify these values directly in genereate_answers.py:

```python
TOP_K = 3
MAX_CONTEXT_CHARS = 1500
```

## **Notes**

- Designed for **Python error debugging**, but can work with any textual query if you have embeddings.
- Works **offline** once embeddings are generated.
- Supports **FAISS vector search** and optionally **BM25 tokenized search**.

---

## **Future Improvements**

- Add **summary of top results** for quicker scanning.
- Optionally integrate **LLM generation** for on-the-fly explanations.
- Add **file-based query batch processing**.

---

## **License**

MIT License — feel free to use and modify for personal or commercial projects.