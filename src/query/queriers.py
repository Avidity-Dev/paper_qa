"""
Document repository querying functionality.
"""

from typing import Any, Optional, Union

from langchain_core.embeddings import Embeddings
from paperqa import EmbeddingModel
from paperqa.docs import Docs as PQADocs, Doc as PQADoc
from paperqa.settings import Settings as PQASettings
from paperqa.settings import AnswerSettings as PQAAnswerSettings
from paperqa.settings import PromptSettings as PQAPromptSettings
from paperqa.settings import ParsingSettings as PQAParsingSettings
from paperqa.types import PQASession, Text as PQAText

from src.storage.vector.stores import RedisVectorStore
from src.storage.vector.converters import LCVectorStorePipeline

# TODO: Potentially add a base class for the querier to allow for more granular
# functionality by subclasses


class PQAQuerier:
    """
    Query class to interrogate a document repository using functionality from paper-qa.

    """

    def __init__(
        self,
        vector_pipeline: LCVectorStorePipeline,
        pqa_settings: PQASettings,
    ):
        self._vector_pipeline = vector_pipeline
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

    @property
    def vector_store(self):
        return self._vector_pipeline.vector_store

    def mmr(
        self,
        query: Union[str, list[float]],
        k: int = 10,
        fetch_k: int = 15,
        embedding_model: Optional[EmbeddingModel] = None,
        **kwargs,
    ) -> Any:
        return self._vector_pipeline.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            embedding_model=self.embedding_model,
            **kwargs,
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
        relevant_docs: list[PQAText]

        # Get the relevant chunks and return text objects and similarity scores
        relevant_docs = await self.mmr(
            query, **kwargs, embedding_model=self.embedding_model
        )

        # Check to if embeddings are present and retrieve them if not
        embeddings = self.vector_store.get_embeddings(
            [doc.name for doc in relevant_docs]
        )
        unique_docs: dict[str, tuple[PQADoc, list[PQAText]]] = {}
        for doc, embedding in zip(relevant_docs, embeddings):
            doc.embedding = embedding
            if doc.doc.dockey not in unique_docs.keys():
                unique_docs[doc.doc.dockey] = (doc.doc, [])

            unique_docs[doc.doc.dockey][1].append(doc)

        # Setup the initial pqa objects to use for querying
        # Inject the text objects into the docs object
        docs = PQADocs()

        for doc, texts in unique_docs.values():
            await docs.aadd_texts(texts, doc=doc)

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
