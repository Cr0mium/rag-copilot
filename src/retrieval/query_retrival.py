import os
import pickle
from typing import Dict, List
import sys

import src.indexing.bm25_index as bm25_index

# compatibility shim for old pickle paths
sys.modules["bm25_index"] = bm25_index

from src.indexing.bm25_index import bm25_tokenize
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_persist import load_vector_store

from sentence_transformers import CrossEncoder

import src.config as config


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
    if f.startswith("bm25_") and f.endswith(".pkl")
)

for path in bm25_shard_paths:
    with open(path, "rb") as f:
        bm25_shards.append(pickle.load(f))

print(f"[✓] Loaded {len(bm25_shards)} BM25 shards")


def rrf_score(rank: int) -> float:
    return 1.0 / (config.RRF_K + rank)


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

        ranked = sorted(
            results,
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return ranked[:top_k]

reranker = Reranker()


def hybrid_search(query: str):

    merged = {}

    print("\n========================")
    print("QUERY:", query)
    print("========================")

    q_emb = embedder.encode([config.QUERY_PREFIX + query])

    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)

    # ---------------------
    # FAISS SEARCH
    # ---------------------

    print("\n--- FAISS RESULTS ---")

    faiss_results = store.search(q_emb, config.TOP_K_FAISS)

    for rank, r in enumerate(faiss_results):

        key = (r["filepath"], r["chunk_id"])

        merged[key] = {
            "filename": r.get("filename"),
            "filepath": r.get("filepath"),
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text"),
            "score": rrf_score(rank),
        }

        if rank < 5:
            print("\nRank:", rank + 1)
            print("FILE:", r["filename"])
            print("CHUNK:", r["chunk_id"])
            print(r["text"][:300])

    # ---------------------
    # BM25 SEARCH
    # ---------------------

    print("\n--- BM25 RESULTS ---")

    q_tokens = bm25_tokenize(query)

    for bm25 in bm25_shards:

        shard_results = bm25.search(q_tokens, top_k=config.TOP_K_BM25_PER_SHARD)

        for rank, r in enumerate(shard_results):

            key = (r["filepath"], r["chunk_id"])
            score = rrf_score(rank)

            if rank < 5:
                print("\nRank:", rank + 1)
                print("FILE:", r["filename"])
                print("CHUNK:", r["chunk_id"])
                print(r["text"][:300])

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

    # ---------------------
    # RRF MERGE
    # ---------------------

    final = list(merged.values())
    final.sort(key=lambda x: x["score"], reverse=True)

    # take top candidates for reranking
    top_candidates = final[:80]

    # ---------------------
    # RERANK
    # ---------------------

    reranked = reranker.rerank(query, top_candidates, top_k=5)

    return reranked


if __name__ == "__main__":

    queries = ["How to register a pipeline in transformers?"]

    for q in queries:

        final = hybrid_search(q)

        for r in final:

            print("\n=================================")
            print("FILE:", r["filename"])
            print("CHUNK:", r["chunk_id"])
            print("RERANK SCORE:", round(r["rerank_score"], 4))
            print(r["text"][:400])