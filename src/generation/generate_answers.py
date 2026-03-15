from typing import List

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from query_retrieval import hybrid_search
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

MODEL_NAME = "meta-llama/Llama-3-8B-Instruct"

TOP_K_CONTEXT = 5


print("[Loading LLaMA model]")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, device_map="auto", torch_dtype="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=pipe)


prompt_template = """
You are a helpful AI assistant answering questions using documentation.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


def build_context(chunks: List[dict]) -> str:

    context_parts = []

    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Source: {c['filename']} | Chunk {c['chunk_id']}]\n{c['text']}"
        )

    return "\n\n".join(context_parts)


def generate_answer(question: str):

    print(f"\n[Query] {question}")

    retrieved = hybrid_search(question)

    top_chunks = retrieved[:TOP_K_CONTEXT]

    context = build_context(top_chunks)

    final_prompt = prompt.format(context=context, question=question)

    response = llm(final_prompt)

    return response


if __name__ == "__main__":
    question = "How do I register a pipeline in transformers?"

    answer = generate_answer(question)

    print("\n===== ANSWER =====\n")
    print(answer)
