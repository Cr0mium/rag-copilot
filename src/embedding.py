from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

# embeddings = model.encode(
#         texts,                     
#         batch_size=32,             
#         convert_to_numpy=True,     # Return NumPy array
#         normalize_embeddings=True, # Normalize for cosine similarity
#         show_progress_bar=False    # Disable for production
#     )

class EmbeddingModel:
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32):
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings