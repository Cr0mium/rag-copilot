import json
import src.config as config

# -------------------------
# Paths
# -------------------------
EVAL_QUESTIONS_PATH = config.EVAL_QUESTIONS_PATH
RETRIEVAL_RESULTS_PATH = config.RETRIVAL_RESULTS_PATH

TOP_K = 5

# -------------------------
# Load data
# -------------------------
with open(EVAL_QUESTIONS_PATH) as f:
    dataset = json.load(f)

with open(RETRIEVAL_RESULTS_PATH) as f:
    retrieval_results = json.load(f)

# -------------------------
# Metrics
# -------------------------
def compute_metrics(results_dict, dataset, k=5):
    hits = 0
    reciprocal_ranks = []

    for item in dataset:
        question = item["question"]
        source = item.get("source")

        retrieved = results_dict.get(question, [])

        found = False

        for rank, r in enumerate(retrieved[:k], start=1):
            if r["filename"] == source:
                hits += 1
                reciprocal_ranks.append(1 / rank)
                found = True
                break

        if not found:
            # print("\nMISS:")
            # print("Q:", question)
            # print("GT:", source)
            # print("Top result:", retrieved[0]["filename"] if retrieved else "None")
            # print("Retirieved chunk:",retrieved[0]["text"] )
            reciprocal_ranks.append(0)

    recall = hits / len(dataset)
    mrr = sum(reciprocal_ranks) / len(dataset)

    return recall, mrr


# -------------------------
# Evaluate all retrievers
# -------------------------
for retriever_type in ["dense", "sparse", "hybrid"]:
    print(f"\n=== {retriever_type.upper()} ===")

    results_dict = retrieval_results[retriever_type]
    for k in [1, 3, 5, 10]:
        recall, mrr = compute_metrics(results_dict, dataset, k)
    
        print(f"Total queries: {len(dataset)}")
        print(f"Recall@{k}: {round(recall, 3)}")
        print(f"MRR: {round(mrr, 3)}")