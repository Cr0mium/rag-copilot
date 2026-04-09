import os
import pickle
from typing import List, Dict

from sentence_transformers import CrossEncoder

from src.indexing.bm25_index import bm25_tokenize
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_persist import load_vector_store
import src.config as config

class HybridRetriever:

    def __init__(self, config):

        self.config = config

        print("[Loading embedder]")
        self.embedder = EmbeddingModel(config.EMBEDDING_MODEL)

        print("[Loading FAISS index]")
        self.store = load_vector_store(config.ARTIFACTS_DIR)
        print(f"[✓] FAISS loaded: {self.store.index.ntotal} vectors")

        print("[Loading BM25 shards]")
        self.bm25_shards = self._load_bm25(config.ARTIFACTS_DIR)
        print(f"[✓] Loaded {len(self.bm25_shards)} BM25 shards")

        print("[Loading reranker]")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    # ----------------------------
    # Internal helpers
    # ----------------------------

    def _load_bm25(self, path):
        shards = []

        for f in sorted(os.listdir(path)):
            if f.startswith("bm25") and f.endswith(".pkl"):
                with open(os.path.join(path, f), "rb") as file:
                    shards.append(pickle.load(file))

        return shards

    def _rrf_score(self, rank: int) -> float:
        return 1.0 / (self.config.RRF_K + rank)

    def _rerank(self, query: str, results: List[Dict]):

        pairs = []

        for r in results:
            passage = f"""
FILE: {r['filename']}
CHUNK: {r['chunk_id']}

{r['text']}
"""
            pairs.append([query, passage])

        scores = self.reranker.predict(pairs, batch_size=16)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        return ranked[:self.config.RERANK_TOP_K]

    # ----------------------------
    # Main API
    # ----------------------------

    def search(self, query: str) -> List[Dict]:

        merged = {}

        # ---- Dense (FAISS)
        q_emb = self.embedder.encode([self.config.QUERY_PREFIX + query])

        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)

        faiss_results = self.store.search(q_emb, self.config.TOP_K_FAISS)

        for rank, r in enumerate(faiss_results):

            key = (r["filepath"], r["chunk_id"])

            merged[key] = {
                "filename": r.get("filename"),
                "filepath": r.get("filepath"),
                "chunk_id": r.get("chunk_id"),
                "text": r.get("text"),
                "score": self._rrf_score(rank),
            }

        # ---- Sparse (BM25)
        q_tokens = bm25_tokenize(query)

        for bm25 in self.bm25_shards:

            shard_results = bm25.search(
                q_tokens, top_k=self.config.TOP_K_BM25_PER_SHARD
            )

            for rank, r in enumerate(shard_results):

                key = (r["filepath"], r["chunk_id"])
                score = self._rrf_score(rank)

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

        # ---- Merge + sort
        final = list(merged.values())
        final.sort(key=lambda x: x["score"], reverse=True)

        # ---- Rerank
        top_candidates = final[:self.config.FINAL_K]
        reranked = self._rerank(query, top_candidates)

        return reranked[:self.config.RETURN_RERANKED_K]

if __name__ == "__main__":

    queries = [
        "How to register a pipeline in transformers?",
        "How do I add a custom pipeline to transformers?",
        "How can I extend transformers pipelines?"
               ]
    retriver=HybridRetriever(config=config)
    for q in queries:

        final = retriver.search(q)
        print("\n=================================")
        print(q)
        for r in final:

            print("\n=================================")
            print("FILE:", r["filename"])
            print("CHUNK:", r["chunk_id"])
            print("RERANK SCORE:", round(r["rerank_score"], 4))
            print(r["text"][:400])