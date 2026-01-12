from load_data import load_documents
from chunking import chunk_text
from vector_store import VectorStore
from embedding import EmbeddingModel

DATA_DIR='data/raw'
BATCH_SIZE=32
EMBEDDING_DIM=768
MODEL_NAME='all-mpnet-base-v2'

docs = load_documents(DATA_DIR)
store= VectorStore(EMBEDDING_DIM)
embedder= EmbeddingModel()


for i,doc in enumerate(docs):
    chunks=chunk_text(doc['text']) #array[string]
    # print(all_chunks)
    for j in range(0,len(chunks),BATCH_SIZE):
        batch_chunks= chunks[i:i+BATCH_SIZE]
        print(f'chunk id: {i+j}',batch_chunks)
        embeddings= embedder.encode(batch_chunks,batch_size=BATCH_SIZE)
        batch_metadata=[{
            'text':chunk,
            'source':doc['source'],
            'id': i+j,
        }for j,chunk in enumerate(batch_chunks)]
        # all_chunks.append(batch_chunks)

        # store.add- build index
        store.add(embeddings, batch_metadata) #batch=32

print("Index size:", store.index.ntotal)



