from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.pipeline import RAGPipeline
from src.evaluation.evaluate_retrieval import evaluate
from src.rag.ragas_eval import run_ragas

app = FastAPI(title="RAG API", version="1.0")

# -------------------------
# CORS
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Initialize pipeline ONCE
# -------------------------
pipeline = RAGPipeline()

# -------------------------
# Request Schemas
# -------------------------
class QueryRequest(BaseModel):
    query: str


# -------------------------
# Routes
# -------------------------
@app.get("/")
def root():
    return {"message": "RAG API is running"}


@app.post("/query")
def query_rag(request: QueryRequest):
    try:
        answer = pipeline.query(request.query)
        return {
            "query": request.query,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/build/retrieval")
def build_retrieval():
    try:
        pipeline.build_retrieval()
        return {'build':"success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/build/ragas")
def build_ragas():
    try:
        pipeline.build_ragas()
        return {'build':"success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
        
        

@app.post("/eval/retrieval")
def eval_retrieval():
    try:
        results = evaluate()

        if not results:
            raise HTTPException(status_code=500, detail="Evaluation failed")

        return {
            "status": "retrieval evaluation completed",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/eval/ragas")
def eval_ragas():
    try:
        results = run_ragas()  # assuming it returns metrics

        return {
            "status": "ragas evaluation completed",
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))