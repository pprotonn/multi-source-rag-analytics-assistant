
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Generate text embeddings using a Hugging Face model."""

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def generate_embeddings(self, texts):
        """
        Convert multiple text documents into numerical vectors.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )

        return embeddings.tolist()

    def generate_query_embedding(self, query):
        """
        Convert a single user question into a numerical vector.
        """
        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding.tolist()
