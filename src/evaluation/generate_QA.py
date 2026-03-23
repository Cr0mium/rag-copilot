import pickle
import json
from collections import defaultdict
from transformers import pipeline
import src.config as config
# Load metadata and evaluation questions
with open(config.EMBED_DIR + '/data.meta.pkl', "rb") as f:
    metadata = pickle.load(f)

docs = defaultdict(list)

# for i,data in enumerate(metadata):
#     print(data)
#     if i>5:
#         break

import gc
import torch

# Delete LLM and outputs


llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    max_new_tokens=120,
    do_sample=False
)

gc.collect()
torch.cuda.empty_cache()

import random
import json

output = []

PROMPT = """<s>[INST]
Answer the question using ONLY the information from the document.

Question:
{question}

Document:
{document}

Answer briefly and factually.
[/INST]"""

# Shuffle questions and pick 80
sample_questions = random.sample(questions, 80)

for q in sample_questions:
    question = q["question"]
    source = q["source"]

    # Join chunks for the current file only
    document = "\n".join(docs[source])
    document = document[:6000]  # truncate if needed

    prompt = PROMPT.format(
        question=question,
        document=document
    )

    response = llm(prompt)

    generated = response[0]["generated_text"]

    # Remove prompt text if present
    if generated.startswith(prompt):
        answer = generated[len(prompt):].strip()
    else:
        answer = generated.strip()

    output.append({
        "question": question,
        "source": source,
        "chunk_id":chunk_id
        "ground_truth": answer
    })

