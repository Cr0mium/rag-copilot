class RAGPipeline:
    def __init__(self):
        import src.config as config
        self.retriever = None
        self.generator = None
        self.config=config

    def _load_query_components(self):
        
        from src.retrieval.query_retrieval import HybridRetriever
        self.retriever = HybridRetriever(config=self.config)

        from src.generation.answer import AnswerGenerator
        self.generator = AnswerGenerator()

    def query(self, question):
        if self.retriever is None:
            self._load_query_components()

        contexts = self.retriever.search(question)
        answer = self.generator.generate(question, contexts)

        return answer
    
    def index(self):
        from src.indexing.build_index import BuildIndex
        
        buildindex=BuildIndex()
        buildindex.indexing()
        
    def eval_retrieval(self):
        from src.evaluation.generate_retrieval_contexts import ContextDataset
        contextDataset=ContextDataset()
        contextDataset.generate_retrieval_dataset()
        
        from src.evaluation.evaluate_retrieval import evaluate
        
        evaluate()
        
    def eval_generation(self):
        from src.rag.ragas_eval import run_ragas
        run_ragas()
        


        