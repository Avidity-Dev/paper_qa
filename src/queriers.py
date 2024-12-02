"""
Document repository querying functionality.
"""

from typing import Any

from paperqa.settings import Settings

from src.models import Document
from src.adapters.vectordb import VectorStore


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
        super().__init__(vector_db)
        self._pqa_settings = pqa_settings

    def query(self, query: str, **kwargs) -> Any:
        pass
