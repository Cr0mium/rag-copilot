#ingestion

SOFT_LIMIT = 180
HARD_LIMIT = 220
OVERLAP = 40

EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"

# indexing

BATCH_SIZE = 64
EMBEDDING_DIM = 768
EMBED_DIR = "embeddings"
RAW_DIR = "data/raw"
MAX_DOCS = 100

#retrieval
QUERY_PREFIX = "query: "

FAISS_DIR = "embeddings"
BM25_DIR = "embeddings"

TOP_K_FAISS = 100
TOP_K_BM25_PER_SHARD = 100

RRF_K=60
RERANK_TOP_K = 60 
RETURN_TOP_K = 5
FINAL_K= 140
RERANKER_MODEL="BAAI/bge-reranker-large"

# evaluate
EVAL_QUESTIONS_PATH='data/evaluate/mistral_gt.json'
API_QUESTIONS_PATH='data/evaluate/eval_api.json'
RETRIEVAL_RESULTS_PATH='data/evaluate/retrieval_results.json'

#generation
LLM_MODEL=""
AUTOTOKENIZER=""