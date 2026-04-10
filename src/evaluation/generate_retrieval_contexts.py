import json
import os
import pickle
from collections import defaultdict

from src.indexing.bm25_index import bm25_tokenize
from src.indexing.embedding import EmbeddingModel
from src.indexing.vector_persist import load_vector_store

from sentence_transformers import CrossEncoder

import src.config as config
RERANK_TOP_K = config.RERANK_TOP_K
FINAL_K = config.FINAL_K


class ContextDataset:
    def __init__(self):
        print("[Loading embedder]")
        self.embedder = EmbeddingModel(config.EMBEDDING_MODEL)

        print("[Loading FAISS index]")
        self.store = load_vector_store(config.ARTIFACTS_DIR)

        print("[Loading BM25 shards]")
        self.bm25_shards = []
        bm25_paths = sorted(
            os.path.join(config.ARTIFACTS_DIR, f)
            for f in os.listdir(config.ARTIFACTS_DIR)
            if f.startswith("bm25") and f.endswith(".pkl")
        )
        for path in bm25_paths:
            with open(path, "rb") as f:
                self.bm25_shards.append(pickle.load(f))

        print("[Loading reranker]")
        self.reranker = CrossEncoder(config.RERANKER_MODEL)

    # -------------------------
    # Helpers
    # -------------------------
    def rrf_score(self, rank: int):
        return 1.0 / (config.RRF_K + rank)

    # -------------------------
    # Retrieval methods
    # -------------------------
    def dense_search(self, query):
        q_emb = self.embedder.encode([config.QUERY_PREFIX + query])
        if q_emb.ndim == 1:
            q_emb = q_emb.reshape(1, -1)
        return self.store.search(q_emb, config.TOP_K_FAISS)

    def sparse_search(self, query):
        q_tokens = bm25_tokenize(query)
        merged = {}

        for bm25 in self.bm25_shards:
            shard_results = bm25.search(q_tokens, top_k=config.TOP_K_BM25_PER_SHARD)

            for rank, r in enumerate(shard_results):
                key = (r["filepath"], r["chunk_id"])
                score = self.rrf_score(rank)

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

    def hybrid_search(self, query):
        # Original method (kept for pipeline usage)
        dense_docs = self.dense_search(query)
        sparse_docs = self.sparse_search(query)
        return self.hybrid_search_from_results(query, dense_docs, sparse_docs)

    def hybrid_search_from_results(self, query, dense_docs, sparse_docs):
        # New method to avoid recomputation during evaluation
        merged = {}

        for rank, r in enumerate(dense_docs):
            key = (r["filepath"], r["chunk_id"])
            merged[key] = {
                "filename": r.get("filename"),
                "filepath": r.get("filepath"),
                "chunk_id": r.get("chunk_id"),
                "text": r.get("text"),
                "score": self.rrf_score(rank),
            }

        for r in sparse_docs:
            key = (r["filepath"], r["chunk_id"])
            if key in merged:
                merged[key]["score"] += r["score"]
            else:
                merged[key] = r

        final = list(merged.values())
        final.sort(key=lambda x: x["score"], reverse=True)

        top_candidates = final[:config.FINAL_K]

        return self.rerank(query, top_candidates)

    def rerank(self, query, results):
        pairs = []
        for r in results:
            passage = f"FILE: {r['filename']}\nCHUNK: {r['chunk_id']}\n{r['text']}"
            pairs.append([query, passage])

        scores = self.reranker.predict(pairs, batch_size=16)

        for r, s in zip(results, scores):
            r["rerank_score"] = float(s)

        ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:config.RERANK_TOP_K]

    # -------------------------
    # Evaluation
    # -------------------------
    def generate_retrieval_dataset(self):
        try:
            with open(config.EVAL_QUESTIONS_PATH, "r") as f:
                eval_questions = json.load(f)

            results = {"dense": {}, "sparse": {}, "hybrid": {}}

            for i, q in enumerate(eval_questions):
                question = q["question"]
                print(f"Processing: {i+1}/{len(eval_questions)}")

                # Compute once
                dense = self.dense_search(question)
                sparse = self.sparse_search(question)
                hybrid = self.hybrid_search_from_results(question, dense, sparse)

                results["dense"][question] = dense
                results["sparse"][question] = sparse
                results["hybrid"][question] = hybrid

            for key in results:
                path = os.path.join(
                    config.RETRIEVAL_RESULTS_PATH,
                    f"{key}_retrieval_contexts.json"
                )
                with open(path, "w") as f:
                    json.dump(results[key], f, indent=2)

            print("✅ Retrieval results saved")

        except Exception as e:
            print(f"[Evaluation error]: {e}")
            raise