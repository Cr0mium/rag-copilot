import json
import os
import src.config as config
from src.generation.answer import AnswerGenerator
import re
TOP_K = 3

# -------------------------
# Helpers
# -------------------------
def normalize(text):
    
    text = text.lower()
    
    # remove section labels (huge win for your data)
    text = re.sub(r'section\s*:\s*', '', text)
    
    # remove punctuation everywhere (not just edges)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# -------------------------
# Load retrieval files
# -------------------------
retrieval_files = {
    # "dense": os.path.join(config.RETRIEVAL_RESULTS_PATH, "dense_retrieval_contexts.json"),
    # "sparse": os.path.join(config.RETRIEVAL_RESULTS_PATH, "sparse_retrieval_contexts.json"),
    "hybrid": os.path.join(config.RETRIEVAL_RESULTS_PATH, "hybrid_retrieval_contexts.json"),
}

retrieval_results = {}

for mode, path in retrieval_files.items():
    with open(path, "r") as f:
        data = json.load(f)

        # normalize keys
        retrieval_results[mode] = {
            normalize(q): v for q, v in data.items()
        }

# -------------------------
# Load eval dataset
# -------------------------
with open(config.EVAL_QUESTIONS_PATH, "r") as f:
    eval_questions = json.load(f)

# normalize questions
for q in eval_questions:
    q["question"] = normalize(q["question"])

# -------------------------
# Init generator
# -------------------------
generator = AnswerGenerator(config=config)

# -------------------------
# Prepare datasets
# -------------------------
datasets = {
    # "dense": [],
    # "sparse": [],
    "hybrid": []
}

for i, q in enumerate(eval_questions):
    # if i>3:
    #   break
    print("Processing:", i)

    question = q["question"]
    ground_truth = q["ground_truth"]

    for mode in datasets:

        retrieved = retrieval_results[mode].get(question, [])

        if not retrieved:
            continue  # skip missing safely

        seen = set()
        unique_contexts = []

        for c in retrieved:
            norm_text = normalize(c["text"])

            if norm_text not in seen:
                unique_contexts.append(c)
                seen.add(norm_text)

        contexts = [c["text"] for c in unique_contexts[:TOP_K]]

        answer = generator.generate(
            question=question,
            contexts=contexts
        )

        data = {
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": ground_truth
        }

        datasets[mode].append(data)

# -------------------------
# Save datasets
# -------------------------
for mode, data in datasets.items():
    output_path = f"{mode}_ragas_dataset.json"

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Saved: {output_path}")