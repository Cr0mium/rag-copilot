import faiss
import numpy as np
from typing import List, Dict

class VectorStore:
    def __init__(self, embedding_dim: int):
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata: List[Dict] = []

    def add(self, embeddings: np.ndarray, metadatas: List[Dict]):
        """
        embeddings: (batch_size* embedding_dim)
        metadatas: list of dicts, same length as embeddings
        """
        assert len(embeddings) == len(metadatas)

        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        query_embedding: (1* embedding_dim)
        """
        scores, indices = self.index.search(query_embedding, top_k)
            # scores  -> [[0.83, 0.79, 0.75]]
            # indices -> [[12, 5, 27]]

        results = []
        for score, idx in zip(scores[0], indices[0]):

            results.append({
                "score": float(score),
                **self.metadata[idx] #unpack dict
            })

        return results