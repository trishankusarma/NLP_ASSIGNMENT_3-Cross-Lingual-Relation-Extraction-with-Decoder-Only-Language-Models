import faiss
from sentence_transformers import SentenceTransformer

class FAISSRetriever:
    """
    Encodes all pool examples with a multilingual sentence encoder.
    At query time, finds top-k most similar examples using cosine similarity.
    """
    def __init__(self, pool, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):

        self.pool  = pool
        self.model = SentenceTransformer(model_name)
        self.faiss = faiss

        print(f"[FAISS] Encoding {len(pool)} pool examples...")
        # Query text = sentence + e1 + e2
        texts = [
            f"{ex['sentText']} {ex['em1Text']} {ex['em2Text']}"
            for ex in pool
        ]
        # Encode in batches
        embeddings = self.model.encode(
            texts, batch_size=256,
            show_progress_bar=True,
            normalize_embeddings=True,   # L2 normalize for cosine via inner product
            convert_to_numpy=True,
        )
        self.embeddings = embeddings.astype('float32')

        # Build FAISS flat index (exact search, inner product = cosine after normalization)
        dim          = self.embeddings.shape[1]
        self.index   = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"[FAISS] Index built: {self.index.ntotal} vectors, dim={dim}")

    def retrieve(self, query_text, k=5):
        """Return k most similar pool examples to query_text."""
        query_vec = self.model.encode(
            [query_text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype('float32')

        _, indices = self.index.search(query_vec, k)
        return [self.pool[i] for i in indices[0]]