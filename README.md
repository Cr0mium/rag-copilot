# 🧠 RAG System with Hybrid Retrieval, Reranking & Modular LLM Backends

A production-oriented Retrieval-Augmented Generation (RAG) system built with a strong focus on **retrieval quality, modularity, and deployability**.

---

## 🚀 Overview

This project implements an end-to-end RAG pipeline with:

* Hybrid retrieval (Dense + Sparse)
* Reranking using cross-encoders
* Retrieval evaluation (recall@k, mrr)
* Evaluation using RAGAS metrics
* Modular LLM backends (Hugging Face / Ollama)
* FastAPI inference service
* Dockerized deployment

---

## 🧩 System Architecture

```
User Query
   ↓
Hybrid Retrieval (FAISS + BM25)
   ↓
Reciprocal Rank Fusion (RRF)
   ↓
Cross-Encoder Reranker
   ↓
Top-K Context
   ↓
LLM (HF / Ollama)
   ↓
Final Answer
```

---

## ⚙️ Features

### 🔍 Hybrid Retrieval

* Dense retrieval using FAISS + embeddings
* Sparse retrieval using BM25
* Fusion via Reciprocal Rank Fusion (RRF)

### 🧠 Reranking

* Cross-encoder reranker improves relevance of retrieved chunks

### 📊 Evaluation

* RAGAS-based evaluation:

  * Context Precision
  * Faithfulness
  * Answer Correctness

### 🔌 Modular LLM Backend

Switch between:

* Hugging Face local models
* Hugging Face Inference API
* Ollama (optimized local inference)

### 🌐 FastAPI Service

* `/query` → inference endpoint
* `/build` → dataset/index build
* `/evaluate` → evaluation pipeline

### 🐳 Dockerized

* Fully containerized pipeline
* Environment-driven configuration
* Ready for cloud deployment

---

## 🛠️ Tech Stack

* Python
* FastAPI
* FAISS
* BM25
* Transformers (Hugging Face)
* RAGAS
* Docker

---

## 📦 Setup

### 1. Clone repo

```bash
git clone <your-repo>
cd <repo>
```

---

### 2. Environment variables

Create `.env` (for local):

```bash
HF_API_KEY=your_huggingface_token
LLM_BACKEND=hf_model  # or ollama / hf_api
LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2" #the llm name from HF
OLLAMA_MODEL="gemma3:1b" #if running local llm from ollama
```

---

### 3. Run locally

```bash
pip install -r requirements.txt
python api.py
```

---

## 🐳 Docker Usage

### Build image

```bash
docker build -t rag-system .
```

### Run container

```bash
docker run \
  -p 8000:8000 \
  -e HF_API_KEY=your_token \
  -e LLM_BACKEND=hf_model \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  rag-system
```

---

## 🧪 Example API Call

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is AutoModelForCausalLM?"}'
```

---

## 📊 Evaluation

Run evaluation pipeline via:

```bash
python main.py eval retrieval
python main.py eval ragas
```

Metrics:

* Faithfulness
* Answer Correctness
* Context Precision

---

## 🚀 Deployment Testing

Tested on GPU-backed cloud instances using Docker:

* Verified environment portability
* Measured inference latency improvements
* Validated GPU compatibility

---

## ⚡ Key Learnings

* Retrieval quality > LLM size in RAG systems
* Hybrid retrieval significantly improves recall
* Reranking is critical for precision
* Containerization introduces I/O and startup tradeoffs
* CPU inference is a major latency bottleneck

---

## 🔮 Future Improvements

* LangChain / LangGraph integration
* Streaming responses
* Caching layer for responses
* Async request handling
* Production-grade logging & monitoring

---

## 📌 Notes

* Hugging Face cache is mounted for faster startup
* Ollama supported for optimized CPU inference
* Designed for extensibility and experimentation

---

## 👨‍💻 Author

Built as part of a hands-on ML systems and RAG engineering journey.
