# main.py

import argparse
from src.rag.pipeline import RAGPipeline

def run_index():
    pipe=RAGPipeline()
    pipe.index()

def run_query(question):
    pipe=RAGPipeline()
    return pipe.query(question)

def run_eval():
    pass

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")

    # create subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # INDEX command
    subparsers.add_parser("index", help="Build FAISS index")

    # QUERY command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="User query")

    # EVAL command
    subparsers.add_parser("eval", help="Run RAGAS evaluation")

    args = parser.parse_args()

    # -------------------
    # Command Routing
    # -------------------
    if args.command == "index":
        run_index()

    elif args.command == "query":
        answer = run_query(args.question)
        print("\nAnswer:\n", answer)

    elif args.command == "eval":
        run_eval()


if __name__ == "__main__":
    main()