#ingestion

SOFT_LIMIT = 180
HARD_LIMIT = 220
OVERLAP = 40

EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
#-------------------------
# indexing

BATCH_SIZE = 64
EMBEDDING_DIM = 768
ARTIFACTS_DIR = "artifacts"
RAW_DIR = "data/raw"
MAX_DOCS = 1000
#-------------------------
#retrieval
QUERY_PREFIX = "query: "

# FAISS_DIR = "embeddings"
# BM25_DIR = "embeddings"

TOP_K_FAISS = 100
TOP_K_BM25_PER_SHARD = 100

RRF_K=60
RERANK_TOP_K = 60 # evaluate retrieval
RETURN_RERANKED_K = 3 #retrieval_query
FINAL_K= 50
RERANKER_MODEL="BAAI/bge-reranker-large"
#-------------------------
# evaluate
EVAL_QUESTIONS_PATH='data/evaluate/evaluation_dataset.json'
API_QUESTIONS_PATH='data/evaluate/eval_api.json'
RETRIEVAL_RESULTS_PATH='data/evaluate/'

#-------------------------
#generation
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# hf_api/hf_model/ollama
LLM_BACKEND= 'ollama' 
# config.py
from dotenv import load_dotenv
import os
load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
HF_API_MODEL= "google/gemma-2b-it"

LLM_MODEL="mistralai/Mistral-7B-Instruct-v0.2"


OLLAMA_MODEL="gemma3:1b"
OLLAMA_ADDRESS="http://127.0.0.1:11434"

#-------------------------
#rag

RAGAS_DATASET_PATH="data/evaluate/hybrid_ragas_dataset.json"