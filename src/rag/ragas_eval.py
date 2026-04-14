from dotenv import load_dotenv
load_dotenv()

import json
import pandas as pd
from datasets import Dataset

from openai import OpenAI

import src.config as config

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision
)

# ----------------------------
# OpenAI wrapper for RAGAS
# ----------------------------
class OpenAIWrapper:
    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def generate(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content


# ----------------------------
# Embeddings wrapper (OpenAI SDK)
# ----------------------------
class OpenAIEmbeddingsWrapper:
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_text(self, text: str):
        resp = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        return resp.data[0].embedding


# ----------------------------
# RAGAS runner
# ----------------------------
def run_ragas():
    try:
        print("[Loading RAGAS dataset]")
        with open(config.RAGAS_DATASET_PATH) as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)

        print("[Loading OpenAI LLM + embeddings]")
        llm = OpenAIWrapper(model="gpt-4o-mini")
        embeddings = OpenAIEmbeddingsWrapper()

        print("[Running RAGAS evaluation]")
        results = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_correctness,
                context_precision
            ],
            llm=llm,
            embeddings=embeddings
        )

        df = results.to_pandas()

        output_path = "./ragas_results.csv"
        df.to_csv(output_path, index=False)

        print(f"✅ RAGAS results saved to {output_path}")

        return {
            "faithfulness": df["faithfulness"].mean(),
            "answer_correctness": df["answer_correctness"].mean(),
            "context_precision": df["context_precision"].mean()
        }

    except Exception as e:
        print(f"[RAGAS evaluation error]: {e}")
        raise