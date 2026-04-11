import json
import os
import src.config as config

TOP_K_LIST = [1, 3, 5, 10]

# -------------------------
# Helpers
# -------------------------
def normalize(text):
    """Normalize strings for safer matching"""
    return text.strip().lower() if isinstance(text, str) else ""

def extract_filename(path_or_name):
    """Extract filename from path if needed"""
    return os.path.basename(path_or_name)

# -------------------------
# Load dataset
# -------------------------

# -------------------------
# Metrics
# -------------------------
def compute_metrics(results_dict, dataset, k=5):
    hits = 0
    reciprocal_ranks = []

    for item in dataset:
        question = item["question"]
        gt_filename = item["filename"]

        retrieved = results_dict.get(question, [])

        found = False

        for rank, r in enumerate(retrieved[:k], start=1):
            pred_filename = r.get("filename", "")

            if pred_filename == gt_filename:
                hits += 1
                reciprocal_ranks.append(1.0 / rank)
                found = True
                break

        if not found:
            reciprocal_ranks.append(0.0)

    recall = hits / len(dataset)
    mrr = sum(reciprocal_ranks) / len(dataset)

    return recall, mrr

# -------------------------
# Evaluate
# -------------------------
def evaluate():
    with open(config.EVAL_QUESTIONS_PATH) as f:
        dataset = json.load(f)

    # Normalize dataset once
    for item in dataset:
        item["question"] = normalize(item["question"])
        item["filename"] = normalize(extract_filename(item["filename"]))

    # -------------------------
    # Load retrieval files
    # -------------------------
    retrieval_files = {
        "dense": os.path.join(config.RETRIEVAL_RESULTS_PATH, "dense_retrieval_contexts.json"),
        "sparse": os.path.join(config.RETRIEVAL_RESULTS_PATH, "sparse_retrieval_contexts.json"),
        "hybrid": os.path.join(config.RETRIEVAL_RESULTS_PATH, "hybrid_retrieval_contexts.json"),
    }

    retrieval_results = {}
    for key, path in retrieval_files.items():
        with open(path) as f:
            data = json.load(f)

            # Normalize keys + filenames
            normalized_data = {}
            for q, results in data.items():
                nq = normalize(q)

                cleaned_results = []
                for r in results:
                    fname = normalize(extract_filename(r.get("filename", "")))

                    cleaned_results.append({
                        **r,
                        "filename": fname
                    })

                normalized_data[nq] = cleaned_results

            retrieval_results[key] = normalized_data

    for retriever_type in ["dense", "sparse", "hybrid"]:
        print(f"\n=== {retriever_type.upper()} ===")

        results_dict = retrieval_results[retriever_type]
        results={}
        for k in TOP_K_LIST:
            recall, mrr = compute_metrics(results_dict, dataset, k)
            results[f"Recall@{k}:"]=f"{recall:.3f} | MRR: {mrr:.3f}"
            # print(f"Recall@{k}: {recall:.3f} | MRR: {mrr:.3f}")
        print(results)
        print(f"Total queries: {len(dataset)}")
        return results