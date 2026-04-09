import gc
import os
import pickle

from tqdm import tqdm

from src.indexing.bm25_index import BM25Index, bm25_tokenize
from src.ingestion.chunking import chunk_text
from src.indexing.embedding import EmbeddingModel
from src.ingestion.load_data import load_hf_docs
from src.indexing.vector_persist import save_vector_store
from src.indexing.vector_store import VectorStore



class BuildIndex():
    def __init__(self):
        import src.config as config
        self.config=config
    
    def indexing(self):
        try:
            os.makedirs(self.config.ARTIFACTS_DIR, exist_ok=True)

            doc_stream = load_hf_docs(self.config.RAW_DIR)
            store = VectorStore(self.config.EMBEDDING_DIM)
            embedder = EmbeddingModel(self.config.EMBEDDING_MODEL)

            chunk_id = 0


            tokenized_docs = []
            metadatas = []

            pbar = tqdm(enumerate(doc_stream), desc="Processing Documents")
            for i, doc in pbar:
                pbar.set_postfix({
                    "doc": i,
                    "chunks": chunk_id
                })
                # array[string]
                chunks = chunk_text(doc.page_content)  # n chunks
                for batch_start in range(0, len(chunks), self.config.BATCH_SIZE):

                    batch_chunks = chunks[batch_start:batch_start+self.config.BATCH_SIZE]
                    batch_metadata = []

                    for chunk in batch_chunks:
                        tokenized_docs.append(bm25_tokenize(chunk))

                        meta = doc.metadata.copy()
                        meta["chunk_id"] = chunk_id
                        meta["text"] = chunk

                        metadatas.append(meta)
                        batch_metadata.append(meta)

                        chunk_id += 1

                    embeddings = embedder.encode(batch_chunks, batch_size=len(batch_chunks))

                    store.add(embeddings, batch_metadata)    
                    
                    if i > 0 and i % 100000 == 0:
                        save_vector_store(store, self.config.ARTIFACTS_DIR)
                        bm25_index = BM25Index(tokenized_docs, metadatas)
                        with open(self.config.ARTIFACTS_DIR + f"/bm25_{i}.pkl", "wb") as f:
                            pickle.dump(bm25_index, f)

                        tokenized_docs.clear()
                        metadatas.clear()
                        del bm25_index
                        gc.collect()
                        
                if i >= self.config.MAX_DOCS:
                    break
            save_vector_store(store, self.config.ARTIFACTS_DIR)
            bm25_index = BM25Index(tokenized_docs, metadatas)
            with open(self.config.ARTIFACTS_DIR + "/bm25.pkl", "wb") as f:
                pickle.dump(bm25_index, f)

            tokenized_docs.clear()
            metadatas.clear()
            del bm25_index
            gc.collect()
            
        except Exception as e:
            print(f"[Indexing Error]: {str(e)}")
            raise