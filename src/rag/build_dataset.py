import json
import src.config as config
from src.generation.answer import AnswerGenerator

EVAL_QUESTIONS_PATH = config.EVAL_QUESTIONS_PATH
RETRIEVAL_RESULTS_PATH = config.RETRIEVAL_RESULTS_PATH

with open(RETRIEVAL_RESULTS_PATH, 'r') as f:
    retrieval_results = json.load(f)

with open(EVAL_QUESTIONS_PATH, 'r') as f:
    eval_questions = json.load(f)

generator = AnswerGenerator(config=config)

# separate datasets per mode
datasets = {
    "dense": [],
    "sparse": [],
    "hybrid": []
}

for i, q in enumerate(eval_questions):
    if i > 10:
        break

    question = q['question']
    ground_truth = q['ground_truth']

    contexts = {
        "dense": retrieval_results['dense'][question][:5],
        "sparse": retrieval_results['sparse'][question][:5],
        "hybrid": retrieval_results['hybrid'][question][:5]
    }

    for mode in contexts:
        ctx = [c["text"] for c in contexts[mode]]

        ans = generator.generate(
            question=question,
            contexts=ctx
        )

        data = {
            "question": question,
            "contexts": ctx,
            "answer": ans,
            "ground_truth": ground_truth
        }

        datasets[mode].append(data)

# save separately
for mode in datasets:
    with open(f"{config.RAG_DATASET}_{mode}.json", "w") as f:
        json.dump(datasets[mode], f, indent=2)