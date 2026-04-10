from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline
from src.evaluation.evaluate_retrieval import evaluate
from src.rag.ragas_eval import run_ragas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize once (important)
pipeline = RAGPipeline()

# -------- Request Schemas --------
class QueryRequest(BaseModel):
    query: str

# -------- Routes --------
@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/query")
def query_rag(request: QueryRequest):
    answer = pipeline.query(request.query)
    return {"query": request.query, "answer": answer}

@app.post("/eval/retrieval")
def eval_retrieval():
    evaluate()
    return {"status": "retrieval evaluation completed"}

@app.post("/eval/ragas")
def eval_ragas():
    run_ragas()
    return {"status": "ragas evaluation completed"}