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
import src.config as config



os.makedirs(config.EMBED_DIR, exist_ok=True)

doc_stream = load_hf_docs(config.RAW_DIR)
store = VectorStore(config.EMBEDDING_DIM)
embedder = EmbeddingModel(config.EMBEDDING_MODEL)

chunk_id = 0


tokenized_docs = []
metadatas = []

top=[]
bottom=[]
pbar = tqdm(enumerate(doc_stream), desc="Processing Documents")
for i, doc in pbar:
    pbar.set_postfix(doc=i)
    # array[string]
    chunks = chunk_text(doc.page_content)  # n chunks
    for batch_start in range(0, len(chunks), config.BATCH_SIZE):

        batch_chunks = chunks[batch_start:batch_start+config.BATCH_SIZE]
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
            save_vector_store(store, config.EMBED_DIR)
            bm25_index = BM25Index(tokenized_docs, metadatas)
            with open(f"./embeddings/bm25_{i}.pkl", "wb") as f:
                pickle.dump(bm25_index, f)

            tokenized_docs.clear()
            metadatas.clear()
            del bm25_index
            gc.collect()
            
    if i > config.MAX_DOCS:
        break
save_vector_store(store, config.EMBED_DIR)
bm25_index = BM25Index(tokenized_docs, metadatas)
with open(config.EMBED_DIR + "/bm25.pkl", "wb") as f:
    pickle.dump(bm25_index, f)

tokenized_docs.clear()
metadatas.clear()
del bm25_index
gc.collect()
