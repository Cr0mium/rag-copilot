import faiss
import pickle
from vector_store import VectorStore

def save_vector_store(store: VectorStore, path: str):
    """
    Saves FAISS index + metadata
    """
    faiss.write_index(store.index, f"{path}.index")
    with open(f"{path}.meta.pkl", "wb") as f:
        pickle.dump(store.metadata, f)


def load_vector_store(path: str) -> VectorStore:
    """
    Loads FAISS index + metadata
    """
    index = faiss.read_index(f"{path}.index")

    with open(f"{path}.meta.pkl", "rb") as f:
        metadata = pickle.load(f)

    store = VectorStore(index.d)
    store.index = index
    store.metadata = metadata
    return store