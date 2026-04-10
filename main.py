# main.py

import argparse
from src.rag.pipeline import RAGPipeline

def run_index():
    pipe=RAGPipeline()
    pipe.index()

def run_query(question):
    pipe=RAGPipeline()
    return pipe.query(question)

def run_eval_retrieval():
    pipe=RAGPipeline()
    pipe.eval_retrieval()
    
def run_eval_ragas():
    pipe=RAGPipeline()
    pipe.eval_generation()

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")

    # create subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # INDEX command
    subparsers.add_parser("index", help="Build index")

    # QUERY command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="User query")

    # EVAL command
    eval_parser=subparsers.add_parser("eval", help="Run RAGAS evaluation")
    eval_parser.add_argument('type',choices=['retrieval','ragas'],help='Retrieval or RAGAS evalution')

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
        if args.type == "retrieval":
            run_eval_retrieval()
        elif args.type == "ragas":
            run_eval_ragas()


if __name__ == "__main__":
    main()