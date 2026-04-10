from dotenv import load_dotenv
load_dotenv()

import json
from datasets import Dataset

import src.config as config

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings


def run_ragas():
    try:
        print("[Loading RAGAS dataset]")
        with open(config.RAGAS_DATASET_PATH) as f:
            data = json.load(f)

        dataset = Dataset.from_list(data)

        print("[Loading LLM + embeddings]")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        embeddings = OpenAIEmbeddings()

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
            'faithfulness':df['faithfulness'].mean(),
            'answer_correctness':df['answer_correctness'].mean(),
            'context_precision':df['context_precision'].mean()
        }

    except Exception as e:
        print(f"[RAGAS evaluation error]: {e}")
        raise