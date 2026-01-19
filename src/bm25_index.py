from rank_bm25 import BM25Okapi
import re
import pickle

class BM25Index:
    def __init__(self, tokenized_corpus, metadatas):
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.metadatas = metadatas

    def search(self, query_tokens, top_k=5):
        scores = self.bm25.get_scores(query_tokens) #scores for the entire corpus
        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = []
        for idx, score in ranked:
            results.append({
                "score": float(score),
                **self.metadatas[idx]
            })
        return results
    

def bm25_tokenize(text: str):
    text = text.lower()
    tokens = re.findall(r"\b\w+\b", text)
    return tokens    