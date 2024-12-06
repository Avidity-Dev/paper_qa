"""
Document repository querying functionality.
"""

from typing import Any

from paperqa.settings import Settings

from src.models import PQADocument
from src.vectorstores.vectordb import VectorStore


class Querier:
    """Interface for querying the document repository."""

    def __init__(self, vector_db: VectorStore, **kwargs):
        """Initialize the querier."""
        self._vector_db = vector_db

    def query(self, query: str, **kwargs) -> Any:
        """Query the document repository."""
        raise NotImplementedError

    def query_with_context(self, query: str, context: str, **kwargs) -> Any:
        """Query the document repository with a context."""
        raise NotImplementedError


class PQAQuerier(Querier):
    """
    Query class to interrogate a document repository using functionality from paper-qa.
    """

    def __init__(self, vector_db: VectorStore, pqa_settings: Settings):
        self._vector_db = vector_db
        self._pqa_settings = pqa_settings

        self._vector_db.connect()

    @property
    def llm(self):
        return self._pqa_settings.get_llm()

    @property
    def embedding_model(self):
        return self._pqa_settings.get_embedding_model()

    async def query(self, query: str, **kwargs) -> Any:
        results = await self._vector_db.max_marginal_relevance_search(
            query,
            k=kwargs.get("k", 10),
            fetch_k=kwargs.get("fetch_k", 100),
            embedding_model=self.embedding_model,
        )
        return results
