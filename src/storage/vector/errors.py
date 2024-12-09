"""Custom errors for the vector stores module."""


class VectorStoreError(Exception):
    """Base class for all vector store errors."""


class PipelineError(Exception):
    """Base class for all pipeline errors."""


class EmbeddingError(VectorStoreError):
    """Error for when a vector store operation fails to obtain embeddings."""
