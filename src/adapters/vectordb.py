from abc import ABC, abstractmethod
from collections.abc import (
    Sequence,
)
import json
import os
import time
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores.redis import Redis
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
import numpy as np
import pandas as pd
import paperqa as pqa
from paperqa.llms import VectorStore as PQAVectorStore
from paperqa.llms import EmbeddingModel as PQAEmbeddingModel

from pydantic import Field
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.models import Document

ListOfDict = List[Dict[str, Any]]


class VectorStore:
    """Base class for vector stores.

    Synchronus methods should be implemented in subclasses. No obligation to implement
    asynchronous methods.
    """

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of documents.

        Detects duplicates, and does not fail if a single document fails to embed.
        """
        raise NotImplementedError

    async def aembed_document(self, text: str) -> list[float]:
        """Generate embedding for a single document string."""
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Synchronous version of embed_documents."""
        pass

    @abstractmethod
    def embed_document(self, text: str) -> list[float]:
        """Synchronous version of embed_document."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Synchronous version of embed_query."""
        pass

    def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )


class LCRedisVectorStore(VectorStore):
    """
    A wrapper around the LangChain Community Redis Adapter.

    Always instantiated from an existing index, as our application will be using
    permanent external storage.
    """

    def __init__(
        self,
        index_name: str,
        index_schema: Union[Dict[str, ListOfDict], str, os.PathLike],
        embedding: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        vector_schema: Optional[Dict[str, Union[str, int]]] = None,
        **kwargs: Any,
    ):
        self._redis = Redis.from_existing_index(
            embedding,
            index_name,
            index_schema,
            vector_schema,
        )

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents asynchronously.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = await self.embeddings.aembed_documents(texts)
        return embeddings

    async def aembed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query string asynchronously.

        Args:
            query: Query text to embed

        Returns:
            Query embedding vector
        """
        embedding = await self.embeddings.aembed_query(query)
        return embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Synchronous version of embed_documents.

        Largely defers to the underlying Redis index.
        """
        return self.embeddings.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """Synchronous version of embed_query.

        Largely defers to the underlying Redis index.
        """
        return self.embeddings.embed_query(query)

    async def aadd_document(self, doc: Document):
        """Asynchronously add a document to the vector store.

        Expects metadata to be parsed and stored in the passed Document object.
        """
        raise NotImplementedError

    def add_document(self, doc: Document):
        """Synchronous version of add_document.

        Expects metadata to be parsed and stored in the passed Document object.
        """
        raise NotImplementedError

    async def aadd_documents(self, docs: List[Document]):
        """Asynchronously add a list of documents to the vector store.

        Expects metadata to be parsed and stored in the passed Document objects.
        """
        raise NotImplementedError

    def add_documents(self, docs: List[Document]):
        """Synchronous version of add_documents.

        Expects metadata to be parsed and stored in the passed Document objects.
        """
        raise NotImplementedError


class LCPineconeVectorStore(VectorStore):
    """A wrapper around the LangChain Community Pinecone Adapter."""

    def __init__(
        self,
        index_name: str,
        embedding: Embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    ):
        self._pinecone = Pinecone.from_existing_index(
            embedding,
            index_name,
        )
