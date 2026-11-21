from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def load_model() -> None:
    """Load the embedding model into RAM once at server startup."""
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")


def embed_text(text: str) -> list[float]:
    """Convert a single string into a 384-dimensional float vector."""
    if _model is None:
        raise RuntimeError("Embedding model is not loaded. Call load_model() first.")

    vector = _model.encode(text, normalize_embeddings=True)
    return vector.tolist()
