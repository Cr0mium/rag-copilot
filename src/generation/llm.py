# llm.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from generate_answers import generate_answer

MODEL_NAME = "google/flan-t5-large"
MAX_INPUT_CHARS = 1200
MAX_NEW_TOKENS = 200


def load_flan():
    print("Loading FLAN model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return tokenizer, model


def clean_context(context: str, max_chars: int = MAX_INPUT_CHARS) -> str:
    """
    Remove metadata-heavy lines and keep only meaningful text
    """
    useful_lines = []
    for line in context.splitlines():
        if line.startswith(("Source:", "QID:", "Title:", "-" * 10)):
            continue
        useful_lines.append(line)

    cleaned = "\n".join(useful_lines).strip()
    return cleaned[:max_chars]


def summarize_with_flan(query: str, context: str, tokenizer, model) -> str:
    cleaned_context = clean_context(context)

    prompt = f"""
Summarize the following Stack Overflow answers to clearly answer the question.

Question:
{query}

Answers:
{cleaned_context}

Provide a concise, practical explanation.
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False  # deterministic summaries
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ----------------- CLI loop -----------------
if __name__ == "__main__":
    tokenizer, model = load_flan()

    print("\nRAG Python Helper (retrieval + FLAN summarization)")
    print("Type 'exit' to quit\n")

    while True:
        query = input(">> Enter error/query: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break

        if not query:
            continue

        # Step 1: Retrieval (core system)
        result = generate_answer(query)

        if not result.get("context"):
            print("\nNo sufficient context retrieved.\n")
            continue

        # Step 2: Lightweight LLM summarization
        summary = summarize_with_flan(
            query=query,
            context=result["context"],
            tokenizer=tokenizer,
            model=model
        )

        print("\n=== Summary ===\n")
        print(summary)

        print("\nSources:")
        for src in result["sources"]:
            print(
                f"  QID: {src['question_id']} | "
                f"AID: {src['answer_id']} | "
                f"Score: {src['score']:.3f}"
            )

        print("\n" + "=" * 70 + "\n")