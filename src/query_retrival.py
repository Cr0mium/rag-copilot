import os
import gc
import pickle

from embedding import EmbeddingModel
from vector_persist import load_vector_store
from bm25_index import bm25_tokenize

MODEL_NAME = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

FAISS_DIR = "embeddings"
BM25_DIR = "embeddings"

TOP_K_FAISS = 50
TOP_K_BM25_PER_SHARD = 15

ALPHA = 0.6
BETA = 0.4
BOOST_BOTH = 1.3
SCORE_THRESHOLD = 0.15

SUMMARIZATION_MODEL = "facebook/bart-large-cnn"
MAX_CHARS = 1500  # Max combined characters for context
TOP_K_SUMMARIES = 5  # How many chunks to summarize

def load_bm25(path):
    with open(path, "rb") as f:
        return pickle.load(f)


embedder = EmbeddingModel(MODEL_NAME)

store = load_vector_store(FAISS_DIR)
print(f"[✓] FAISS loaded: {store.index.ntotal} vectors")

bm25_shard_paths = sorted(
    os.path.join(BM25_DIR, f)
    for f in os.listdir(BM25_DIR)
    if f.startswith("bm25_") and f.endswith(".pkl")
)


def hybrid_search(query: str):
    merged = {}


    q_emb = embedder.encode([QUERY_PREFIX + query])
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    faiss_results = store.search(q_emb, TOP_K_FAISS)

    for rank, r in enumerate(faiss_results):
        key = (r["source"], r["chunk_id"])
        merged[key] = {
            "text": r.get("text", ""),
            "source": r.get("source"),
            "question_id": r.get("question_id"),
            "answer_id": r.get("answer_id"),
            "tags": r.get("tags"),
            "faiss": 1 / (1 + rank),
            "bm25": 0.0,
        }

    q_tokens = bm25_tokenize(query)

    for shard_path in bm25_shard_paths:
        bm25 = load_bm25(shard_path)

        shard_results = bm25.search(
            q_tokens,
            top_k=TOP_K_BM25_PER_SHARD
        )

        for rank, r in enumerate(shard_results):
            key = (r["source"], r["chunk_id"])
            score = 1 / (1 + rank)

            if key in merged:
                merged[key]["bm25"] = max(merged[key]["bm25"], score)
            else:
                merged[key] = {
                    "text": r.get("text", ""),
                    "source": r.get("source"),
                    "question_id": r.get("question_id"),
                    "answer_id": r.get("answer_id"),
                    "tags": r.get("tags"),
                    "faiss": 0.0,
                    "bm25": score,
                }
        del bm25
        gc.collect()


    final = []
    for (source, chunk_id), v in merged.items():
        score = ALPHA * v["faiss"] + BETA * v["bm25"]

        if v["faiss"] > 0 and v["bm25"] > 0:
            score *= BOOST_BOTH

        if score >= SCORE_THRESHOLD:
            final.append({
                "source": source,
                "chunk_id": chunk_id,
                "score": score,
                "faiss": v["faiss"],
                "bm25": v["bm25"],
                "text": v["text"],
                "question_id": v.get("question_id"),
                "answer_id": v.get("answer_id"),
            })

    final.sort(key=lambda x: x["score"], reverse=True)
    return final

if __name__ =='__main__':

    queries=[
    "list index out of range",
    ]

    for q in queries:
        final=hybrid_search(q)
        for ans in final:
            print(ans)
