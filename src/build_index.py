from load_data import load_documents
from load_pdf import load_pdf_stream
from chunking import chunk_text
from vector_store import VectorStore
from embedding import EmbeddingModel
from vector_persist import load_vector_store,save_vector_store

DATA_DIR='data/raw'
BATCH_SIZE=32
EMBEDDING_DIM=768
MODEL_NAME="BAAI/bge-base-en-v1.5"

docs = load_documents(DATA_DIR)
pdf = load_pdf_stream(DATA_DIR+'postgresql.pdf')
store= VectorStore(EMBEDDING_DIM)
embedder= EmbeddingModel()

chunk_id=0
for i,doc in enumerate(docs):
    chunks=chunk_text(doc['text']) #array[string]
    # print(all_chunks)
    for j in range(0,len(chunks),BATCH_SIZE):
        batch_chunks= chunks[j:j+BATCH_SIZE]
        # print(f'chunk id: {i+j}',batch_chunks)
        embeddings= embedder.encode(batch_chunks,batch_size=BATCH_SIZE)
        batch_metadata=[]
        for chunk in batch_chunks:   
            batch_metadata.append({
                'text':chunk,
                'source':doc['source'],
                'chunk_id': chunk_id,
                'doc_id':i,
                'lang':'python'
            })
            chunk_id+=1
        # all_chunks.append(batch_chunks)

        # store.add- build index
        store.add(embeddings, batch_metadata) #batch=32


for i,page in pdf:
    # array[string]
    chunks=chunk_text(page['text'])
    # print(all_chunks)
    for j in range(0,len(chunks),BATCH_SIZE):
        batch_chunks= chunks[j:j+BATCH_SIZE]
        print(f'chunk id: {i+j}',batch_chunks)
        embeddings= embedder.encode(batch_chunks,batch_size=BATCH_SIZE)
        batch_metadata=[]
        for chunk in enumerate(batch_chunks):
            
            {
                'text':chunk,
                'source':page['source'],
                'chunk_id': chunk_id,
                'page':i,
                'lang':'sql'
            }
            chunk_id+=1
        # all_chunks.append(batch_chunks)

        # store.add- build index
        store.add(embeddings, batch_metadata) #batch=32
print("Index size:", store.index.ntotal)

save_vector_store(store,DATA_DIR)



