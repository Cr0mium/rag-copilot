from load_data import stream_documents
from chunking import chunk_text
from vector_store import VectorStore
from embedding import EmbeddingModel
from vector_persist import save_vector_store
from tqdm import tqdm
from bm25_index import BM25Index, bm25_tokenize
import pickle
import gc

DATA_DIR='data/raw'
BATCH_SIZE=32
EMBEDDING_DIM=768
MODEL_NAME="BAAI/bge-base-en-v1.5"
EMBED_DIR='embeddings'

import os
os.makedirs(EMBED_DIR, exist_ok=True)

doc_stream = stream_documents()
store= VectorStore(EMBEDDING_DIM)
embedder= EmbeddingModel(MODEL_NAME)

chunk_id=0


tokenized_docs = []
metadatas = []




for i,doc in enumerate(tqdm(doc_stream, desc="Processing Documents")):
    # array[string]
    chunks=chunk_text(doc.page_content)#n chunks
    chunk_metas = []
    for chunk in chunks:
        tokenized_docs.append(bm25_tokenize(chunk))
        meta = doc.metadata.copy()
        meta['chunk_id'] = chunk_id
        metadatas.append(meta)
        chunk_metas.append(meta) #for batches
        chunk_id += 1
    # print(all_chunks)
    for j in range(0,len(chunks),BATCH_SIZE):
        batch_chunks= chunks[j:j+BATCH_SIZE]
        # print(f'chunk id: {i+j}',batch_chunks)
        embeddings= embedder.encode(batch_chunks,batch_size=BATCH_SIZE)

        batch_metadata = chunk_metas[j:j+BATCH_SIZE]
        # all_chunks.append(batch_chunks)

        # store.add- build index
        store.add(embeddings, batch_metadata) #batch=32
    if i>500000:
        break

print("Index size:", store.index.ntotal)

save_vector_store(store,EMBED_DIR)


bm25_index = BM25Index(tokenized_docs, metadatas)
print(f"BM25 Index size: {len(tokenized_docs)} chunks")
# Free memory
del tokenized_docs
gc.collect()


with open("./embeddings/bm25.pkl", "wb") as f:
    pickle.dump(bm25_index, f)



