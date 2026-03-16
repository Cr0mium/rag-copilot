import json
import os
import pickle
from collections import defaultdict

from src.indexing.bm25_index import bm25_tokenize
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_persist import load_vector_store

from sentence_transformers import CrossEncoder

import src.config as config

# -------------------------
# Load embedder & stores
# -------------------------
print("[Loading embedder]")
embedder = EmbeddingModel(config.EMBEDDING_MODEL)

print("[Loading FAISS index]")
store = load_vector_store(config.FAISS_DIR)
print(f"[✓] FAISS loaded: {store.index.ntotal} vectors")

print("[Loading BM25 shards]")
bm25_shards = []
bm25_shard_paths = sorted(
    os.path.join(config.BM25_DIR, f)
    for f in os.listdir(config.BM25_DIR)
    if f.startswith("bm25") and f.endswith(".pkl")
)
for path in bm25_shard_paths:
    with open(path, "rb") as f:
        bm25_shards.append(pickle.load(f))
print(f"[✓] Loaded {len(bm25_shards)} BM25 shards")

# -------------------------
# RRF scoring
# -------------------------
def rrf_score(rank: int) -> float:
    return 1.0 / (config.RRF_K + rank)

# -------------------------
# Reranker
# -------------------------
class Reranker:
    def __init__(self):
        print("[Loading reranker]")
        self.model = CrossEncoder(config.RERANKER_MODEL)

    def rerank(self, query, results, top_k=5):
        pairs = []
        for r in results:
            passage = f"""
FILE: {r['filename']}
CHUNK: {r['chunk_id']}

{r['text']}
"""
            pairs.append([query, passage])
        scores = self.model.predict(pairs, batch_size=16)
        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)
        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

reranker = Reranker()

# -------------------------
# Retrieval functions
# -------------------------
def dense_search(query):
    q_emb = embedder.encode([config.QUERY_PREFIX + query])
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss_results = store.search(q_emb, config.TOP_K_FAISS)
    # Return list of docs in rank order
    return faiss_results

def sparse_search(query):
    q_tokens = bm25_tokenize(query)
    merged = {}
    for bm25 in bm25_shards:
        shard_results = bm25.search(q_tokens, top_k=config.TOP_K_BM25_PER_SHARD)
        for rank, r in enumerate(shard_results):
            key = (r["filepath"], r["chunk_id"])
            score = rrf_score(rank)
            if key in merged:
                merged[key]["score"] += score
            else:
                merged[key] = {
                    "filename": r.get("filename"),
                    "filepath": r.get("filepath"),
                    "chunk_id": r.get("chunk_id"),
                    "text": r.get("text"),
                    "score": score,
                }
    final = list(merged.values())
    final.sort(key=lambda x: x["score"], reverse=True)
    return final

def hybrid_search(query):
    # Combine dense + sparse
    dense_docs = dense_search(query)
    sparse_docs = sparse_search(query)
    # Merge by RRF
    merged = {}
    for rank, r in enumerate(dense_docs):
        key = (r["filepath"], r["chunk_id"])
        merged[key] = {
            "filename": r.get("filename"),
            "filepath": r.get("filepath"),
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text"),
            "score": rrf_score(rank),
        }
    for rank, r in enumerate(sparse_docs):
        key = (r["filepath"], r["chunk_id"])
        if key in merged:
            merged[key]["score"] += r["score"]
        else:
            merged[key] = r
    final = list(merged.values())
    final.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = final[:120]
    reranked = reranker.rerank(query, top_candidates, top_k=50)
    return reranked

# -------------------------
# Load evaluation questions
# -------------------------
with open(config.EVAL_QUESTIONS_PATH, "r") as f:
    eval_questions = json.load(f)

# -------------------------
# Run retrieval & save results
# -------------------------
results = {"dense": {}, "sparse": {}, "hybrid": {}}

for q in eval_questions:
    question = q["question"]
    print("Processing:", question)

    results["dense"][question] = dense_search(question)
    results["sparse"][question] = sparse_search(question)
    results["hybrid"][question] = hybrid_search(question)

# Save results
with open("evaluation/retrieval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("✅ Retrieval results saved to evaluation/retrieval_results.json")