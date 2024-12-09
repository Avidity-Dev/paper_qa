"""
Document repository querying functionality.
"""

from typing import Any, Optional

from paperqa.docs import Docs as PQADocs
from paperqa.settings import Settings as PQASettings
from paperqa.settings import AnswerSettings as PQAAnswerSettings
from paperqa.settings import PromptSettings as PQAPromptSettings
from paperqa.settings import ParsingSettings as PQAParsingSettings
from paperqa.types import PQASession, Text as PQAText

from src.models import PQADocument
from src.vectorstores.stores import VectorStore


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
    ):
        self._vector_db = vector_db
        self._pqa_settings = pqa_settings

    @property
    def llm(self):
        return self._pqa_settings.get_llm()

    @property
    def summary_llm(self):
        return self._pqa_settings.get_summary_llm()

    @property
    def embedding_model(self):
        return self._pqa_settings.get_embedding_model()

    @property
    def settings(self):
        return self._pqa_settings

    async def mmr(self, query: str, **kwargs) -> Any:
        return await self._vector_db.max_marginal_relevance_search(
            query,
            k=kwargs.get("k", 10),
            fetch_k=kwargs.get("fetch_k", 100),
            embedding_model=self.embedding_model,
        )

    async def query(self, query: str, **kwargs) -> PQASession:
        """
        Query the document repository.

        Build a PQASession to inject context for query, instead of relying on the
        internal call stack using by paper-qa, since we are persisting our document
        metadata and embeddings externally.

        Parameters
        ----------
        query : str
            The query to ask the document repository.

        Returns
        -------
        PQASession
            _description_
        """
        session: PQASession
        relevant_text_objects: list[PQAText]

        # Get the relevant chunks and return text objects and similarity scores
        relevant_text_objects, scores = await self.mmr(query, **kwargs)

        # Setup the initial pqa objects to use for querying
        # Inject the text objects into the docs object
        docs = PQADocs()

        # Dummy doc for now
        # TODO: Add the actual doc metadata that's parsed during processing
        doc = PQADocument(dockey="", citation="", docname="")
        await docs.aadd_texts(relevant_text_objects, doc=doc)

        session = PQASession(question=query)

        # Generate the context objects and store in session
        session = await docs.aget_evidence(
            query=session,
            settings=self._pqa_settings,
            summary_llm_model=self.summary_llm,
        )

        # Get the LLM to answer the question
        session = await docs.aquery(
            query=session,
            settings=self.settings,
            llm_model=self.llm,
            summary_llm_model=self.summary_llm,
        )

        return session
