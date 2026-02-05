import faiss
import pickle
from vector_store import VectorStore

def save_vector_store(store: VectorStore, path: str):
    """
    Saves FAISS index + metadata
    """
    print(f'Saving index in {path}')
    faiss.write_index(store.index, f"{path}/data.index")
    with open(f"{path}/data.meta.pkl", "wb") as f:
        pickle.dump(store.metadata, f)


def load_vector_store(path: str) -> VectorStore:
    """
    Loads FAISS index + metadata
    """
    index = faiss.read_index(f"{path}/data.index")

    with open(f"{path}/data.meta.pkl", "rb") as f:
        metadata = pickle.load(f)

    store = VectorStore(index.d)
    store.index = index
    store.metadata = metadata
    return store