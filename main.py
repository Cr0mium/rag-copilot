# main.py

import argparse
from random import choices
from src.rag.pipeline import RAGPipeline

def run_index():
    pipe=RAGPipeline()
    pipe.index()

def run_query(question):
    pipe=RAGPipeline()
    return pipe.query(question)

def run_build_retrieval():
    pipe=RAGPipeline()
    pipe.build_retrieval()
    print("end of build_retrieval")
    
def run_build_ragas():
    pipe=RAGPipeline()
    pipe.build_ragas()
    print("end of build_ragas")

def run_eval_retrieval():
    pipe=RAGPipeline()
    pipe.eval_retrieval()
    
def run_eval_ragas():
    pipe=RAGPipeline()
    pipe.eval_ragas()

def main():
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")

    # subparser manager
    subparsers = parser.add_subparsers(dest="command", required=True)

    # INDEX command
    subparsers.add_parser("index", help="Build index")

    # QUERY command
    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", type=str, help="User query")
    
    #BUILD command
    
    build_parser= subparsers.add_parser("build",help="Build dataset for Ragas or Retreival")
    build_parser.add_argument('type',choices=['retrieval','ragas'])

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
        
    elif args.command=='build':
        if args.type=='retrieval':
            run_build_retrieval()
        elif args.type=='ragas':
            run_build_ragas()

    elif args.command == "eval":
        if args.type == "retrieval":
            print("only run after 'build retrieval' is completed. Continue?")
            res=input("[y/n]: ")
            if res.lower() !='y':
                return
            print(run_eval_retrieval())
        elif args.type == "ragas":
            print("only run after 'build ragas' is completed. Continue?")
            res=input("[y/n]: ")
            if res.lower() !='y':
                return
            print(run_eval_ragas())


if __name__ == "__main__":
    main()