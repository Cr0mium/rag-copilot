import pickle
import random
import json
from collections import defaultdict
from transformers import pipeline
import src.config as config
import gc
import torch

# ----------------------------
# Load metadata
# ----------------------------
with open(config.EMBED_DIR + '/data.meta.pkl', "rb") as f:
    metadata = pickle.load(f)

# ----------------------------
# Build chunk storage
# ----------------------------
chunks = []

for i, data in enumerate(metadata):
    chunks.append({
        "chunk_id": i,
        "text": data["text"],
        # "source": data.get("source", "unknown"),
        "filename":data["filename"]
    })

# ----------------------------
# Load LLM
# ----------------------------
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    device_map="auto",
    max_new_tokens=200,
    do_sample=False
)

gc.collect()
torch.cuda.empty_cache()

# ----------------------------
# Prompt (STRICT grounding)
# ----------------------------
PROMPT = """<s>[INST]
You are creating evaluation data for a QA system.

Given the context, generate:
1. A clear, specific question
2. A precise answer STRICTLY from the context

Rules:
- Do NOT use outside knowledge
- Do NOT hallucinate
- Answer must be directly supported by the text
- Keep answer concise (2-4 lines max)

Return format:
Question: <question>
Answer: <answer>

Context:
{context}
[/INST]"""

# ----------------------------
# Sample 150 random chunks
# ----------------------------
sampled_chunks = random.sample(chunks, 150)

output = []

# ----------------------------
# Generate Q&A
# ----------------------------
# ----------------------------
# Prepare prompts
# ----------------------------
prompts = []
valid_chunks = []

for chunk in sampled_chunks:
    context = chunk["text"]

    prompt = PROMPT.format(context=context)

    prompts.append(prompt)
    valid_chunks.append({
        "context": context,
        # "source": chunk["source"],
        "filename":chunk["filename"],
        "chunk_id": chunk["chunk_id"]
    })

# ----------------------------
# Batched generation
# ----------------------------
BATCH_SIZE = 8   # adjust (try 4–8 based on VRAM)

output = []

for i in range(0, len(prompts), BATCH_SIZE):
    print(f"Processing batch {i // BATCH_SIZE}")

    batch_prompts = prompts[i:i+BATCH_SIZE]
    batch_chunks = valid_chunks[i:i+BATCH_SIZE]

    responses = llm(batch_prompts)

    for j, res in enumerate(responses):
        prompt = batch_prompts[j]
        chunk_info = batch_chunks[j]

        generated = res["generated_text"]

        # Remove prompt text if present
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        # ----------------------------
        # Parse output
        # ----------------------------
        try:
            q_part = generated.split("Question:")[1].split("Answer:")[0].strip()
            a_part = generated.split("Answer:")[1].strip()
        except:
            continue

        output.append({
            "question": q_part,
            "ground_truth": a_part,
            "context": chunk_info["context"],
            # "source": chunk_info["source"],
            "filename":chunk_info["filename"],
            "chunk_id": chunk_info["chunk_id"]
        })
# ----------------------------
# Save
# ----------------------------
with open("evaluation_dataset.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Generated {len(output)} Q&A pairs")