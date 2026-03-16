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

# generate ans
QUERY_PREFIX = "query: "

FAISS_DIR = "embeddings"
BM25_DIR = "embeddings"

TOP_K_FAISS = 100
TOP_K_BM25_PER_SHARD = 100

RRF_K = 50

RERANKER_MODEL="BAAI/bge-reranker-large"

# evaluate
EVAL_QUESTIONS_PATH='data/evaluate/eval_questions.json'
API_QUESTIONS_PATH='data/evaluate/eval_api.json'