class RAGPipeline:
    def __init__(self):
        import src.config as config

        self.config = config

        # lazy-loaded components
        self.retriever = None
        self.generator = None
        self.buildindex = None
        self.contextDataset = None

    # -------------------------
    # Query path
    # -------------------------
    def _load_query_components(self):
        from src.retrieval.query_retrieval import HybridRetriever
        from src.generation.answer import AnswerGenerator

        self.retriever = HybridRetriever(config=self.config)
        self.generator = AnswerGenerator()

    def query(self, question):
        if self.retriever is None or self.generator is None:
            self._load_query_components()

        contexts = self.retriever.search(question)
        answer = self.generator.generate(question, contexts)

        return answer

    # def health_check(self):
    #     if self.generator==None:
    #         self.
    # -------------------------
    # Indexing
    # -------------------------
    def index(self):
        if self.buildindex is None:
            from src.indexing.build_index import BuildIndex
            self.buildindex = BuildIndex()

        self.buildindex.indexing()

    # -------------------------
    # Retrieval dataset build
    # -------------------------
    def build_retrieval(self):
        if self.contextDataset is None:
            from src.evaluation.generate_retrieval_contexts import ContextDataset
            self.contextDataset = ContextDataset()

        self.contextDataset.generate_retrieval_dataset()

    # -------------------------
    # Retrieval evaluation
    # -------------------------
    def eval_retrieval(self):
        from src.evaluation.evaluate_retrieval import evaluate
        evaluate()

    # -------------------------
    # RAGAS dataset build
    # -------------------------
    def build_ragas(self):
        import os
        from src.rag.build_dataset import RAGASDatasetBuilder

        retrieval_files = {
            "hybrid": os.path.join(
                self.config.RETRIEVAL_RESULTS_PATH,
                "hybrid_retrieval_contexts.json"
            )
        }

        builder = RAGASDatasetBuilder(config=self.config, top_k=3)

        builder.load_retrieval_results(retrieval_files)
        builder.load_eval_questions()

        datasets = builder.build(modes=["hybrid"])
        builder.save(datasets)

    # -------------------------
    # RAGAS evaluation
    # -------------------------
    def eval_ragas(self):
        from src.rag.ragas_eval import run_ragas
        return run_ragas()