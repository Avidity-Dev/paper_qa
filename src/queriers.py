"""
Document repository querying functionality.
"""

from typing import Any

from paperqa.docs import Docs as PQADocs
from paperqa.settings import Settings as PQASettings
from paperqa.settings import AnswerSettings as PQAAnswerSettings
from paperqa.types import PQASession

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


class PQAQuerier:
    """
    Query class to interrogate a document repository using functionality from paper-qa.
    """

    def __init__(
        self,
        vector_db: VectorStore,
        pqa_settings: PQASettings,
        pqa_answer_settings: PQAAnswerSettings,
    ):
        self._pqa_settings = pqa_settings
        self._pqa_answer_settings = pqa_answer_settings
        self._pqa_docs = PQADocs(
            texts_index=self._vector_db,
        )

    @property
    def llm(self):
        return self._pqa_settings.get_llm()

    @property
    def embedding_model(self):
        return self._pqa_settings.get_embedding_model()

    async def mmr(self, query: str, **kwargs) -> Any:
        return await self._pqa_docs.texts_index.max_marginal_relevance_search(
            query,
            k=kwargs.get("k", 10),
            fetch_k=kwargs.get("fetch_k", 100),
            embedding_model=self.embedding_model,
        )

    async def query(self, query: str, **kwargs) -> Any:
        # Get the results from the vector database, along with the embeddings
        relevant_docs = await self.mmr(query, **kwargs)

        # Get the LLM to answer the question
        return relevant_docs
