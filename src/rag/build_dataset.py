import json
import os
import re
from typing import Dict, List

import src.config as config
from src.generation.answer import AnswerGenerator


class RAGASDatasetBuilder:
    def __init__(self, config, top_k: int = 3):
        self.config = config
        self.top_k = top_k
        self.generator = AnswerGenerator(config=config)

        self.retrieval_results = {}
        self.eval_questions = []

    # -------------------------
    # Helpers
    # -------------------------
    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower()
        text = re.sub(r'section\s*:\s*', '', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # -------------------------
    # Loaders
    # -------------------------
    def load_retrieval_results(self, retrieval_files: Dict[str, str]):
        for mode, path in retrieval_files.items():
            with open(path, "r") as f:
                data = json.load(f)

                self.retrieval_results[mode] = {
                    self.normalize(q): v for q, v in data.items()
                }

    def load_eval_questions(self):
        with open(self.config.EVAL_QUESTIONS_PATH, "r") as f:
            self.eval_questions = json.load(f)

        for q in self.eval_questions:
            q["question"] = self.normalize(q["question"])

    # -------------------------
    # Core logic
    # -------------------------
    def _deduplicate_contexts(self, retrieved: List[Dict]) -> List[str]:
        seen = set()
        unique_contexts = []

        for c in retrieved:
            norm_text = self.normalize(c["text"])

            if norm_text not in seen:
                unique_contexts.append(c)
                seen.add(norm_text)

        return unique_contexts[:self.top_k]

    def build(self, modes: List[str]) -> Dict[str, List[Dict]]:
        datasets = {mode: [] for mode in modes}

        for i, q in enumerate(self.eval_questions):
            print(f"Processing: {i}")

            question = q["question"]
            ground_truth = q["ground_truth"]

            for mode in modes:
                retrieved = self.retrieval_results.get(mode, {}).get(question, [])

                if not retrieved:
                    continue

                contexts = self._deduplicate_contexts(retrieved)

                answer = self.generator.generate(
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

        return datasets

    # -------------------------
    # Save
    # -------------------------
    def save(self, datasets: Dict[str, List[Dict]]):
        for mode, data in datasets.items():
            output_path = f"{mode}_ragas_dataset.json"

            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)

            print(f"[✓] Saved: {output_path}")