# generate_answers.py
import json
from typing import List, Dict
from query_retrival import hybrid_search

TOP_K = 3         # how many top results to use
MAX_CONTEXT_CHARS = 5000

def build_context(results: List[Dict], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Build a readable context string from retrieved results."""
    blocks = []
    for r in results:
        block_lines = [
            f"Source: {r.get('source', 'Unknown')}",
            f"QID: {r.get('question_id')} | AID: {r.get('answer_id')} | Score: {r.get('score'):.3f}"
        ]
        if 'title' in r and r['title']:
            block_lines.append(f"Title: {r['title']}")
        if 'question' in r and r['question']:
            block_lines.append(f"Question: {r['question']}")
        if 'text' in r and r['text']:
            block_lines.append(f"Answer: {r['text']}")
        blocks.append("\n".join(block_lines))

    context = "\n\n" + ("-"*60) + "\n\n".join(blocks)
    return context[:max_chars]

def generate_answer(query: str) -> Dict:
    """Retrieve top-k relevant chunks for the query."""
    retrieved = hybrid_search(query)

    if not retrieved:
        return {
            "query": query,
            "explanation": "Not enough information",
            "confidence": "low",
            "sources": [],
            "context": ""
        }

    top_results = retrieved[:TOP_K]
    context = build_context(top_results)

    return {
        "query": query,
        "explanation": "Retrieved top relevant results",
        "confidence": "high",
        "context": context,
        "sources": [
            {
                "question_id": r.get("question_id"),
                "answer_id": r.get("answer_id"),
                "score": r.get("score")
            }
            for r in top_results
        ]
    }

# ----------------- CLI loop -----------------
if __name__ == "__main__":
    print("RAG Python Helper (retrieval-only, type 'exit' to quit)\n")
    while True:
        query = input(">> Enter error/query: ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting...")
            break
        if not query:
            continue

        result = generate_answer(query)
        print("\n=== Result ===")
        print(f"Query      : {result['query']}")
        print(f"Confidence : {result['confidence']}")
        print(f"Explanation: {result['explanation']}\n")
        print("Top Results:\n")
        print(result['context'])
        print("\nSources:")
        for src in result['sources']:
            print(f"  QID: {src['question_id']} | AID: {src['answer_id']} | Score: {src['score']:.3f}")
        print("\n" + "="*70 + "\n")