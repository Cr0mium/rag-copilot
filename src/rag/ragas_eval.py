from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
import src.config as config
import json

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

with open(config.RAGAS_DATASET_PATH) as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()

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
df.to_csv("./ragas_results.csv", index=False)