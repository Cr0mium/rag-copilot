import json
from src.retrieval.query_retrival import hybrid_search

TOP_K = 5

with open("data/evaluate/eval_questions.json") as f:
    dataset = json.load(f)

hits = 0
reciprocal_ranks = []

for item in dataset:
    if (item):
        question = item["question"]
        source = item["source"]

    results = hybrid_search(question)

    found = False

    for rank, r in enumerate(results[:TOP_K], start=1):

        if r["filename"] == source:
            hits += 1
            reciprocal_ranks.append(1 / rank)
            found = True
            break

    if not found:
        reciprocal_ranks.append(0)

recall = hits / len(dataset)
mrr = sum(reciprocal_ranks) / len(dataset)

print("Total queries:", len(dataset))
print("Recall@5:", round(recall, 3))
print("MRR@5:", round(mrr, 3))