"""Vector database adapter implementations.

This module provides adapters for various vector databases, implementing
the necessary interfaces for vector storage and retrieval operations.
"""

from abc import ABC, abstractmethod
from collections.abc import (
    Sequence,
)
import json
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union
import uuid

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
from src.vectorstores.keymanager import KeyManager, RedisKeyManager

ListOfDict = List[Dict[str, Any]]

# TODO: Implement session-based embedding logic to capture failed embedding attempts


class VectorStore:
    """Base class for vector stores.

    Synchronous methods should be implemented in subclasses. No obligation to implement
    asynchronous methods.
    """

    async def aembed_documents(
        self, input: Union[list[str], list[Document]]
    ) -> list[list[float]]:
        """Generate embeddings for a list of documents asynchronously.

        Parameters
        ----------
        input : Union[list[str], list[Document]]
            List of text strings or Document objects to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors for each document.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(
        self, input: Union[list[str], list[Document]]
    ) -> list[list[float]]:
        """Synchronously generate embeddings for a list of documents.

        Parameters
        ----------
        input : Union[list[str], list[Document]]
            List of text strings or Document objects to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors for each document.
        """
        pass

    @abstractmethod
    def embed_document(self, input: Union[str, Document]) -> list[float]:
        """Synchronously generate an embedding for a single document.

        Parameters
        ----------
        input : Union[str, Document]
            Text string or Document object to embed.

        Returns
        -------
        list[float]
            Embedding vector for the document.
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Synchronously generate an embedding for a query.

        Parameters
        ----------
        query : str
            Query text to embed.

        Returns
        -------
        list[float]
            Embedding vector for the query.
        """
        pass

    def cosine_similarity(
        self, embedding1: list[float], embedding2: list[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings.

        Parameters
        ----------
        embedding1 : list[float]
            First embedding vector.
        embedding2 : list[float]
            Second embedding vector.

        Returns
        -------
        float
            Cosine similarity between the two embeddings.
        """
        return np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )


class LCRedisVectorStore(VectorStore):
    """A wrapper around the LangChain Community Redis Adapter.

    Always instantiated from an existing index, as our application will be using
    permanent external storage.

    Attributes
    ----------
    _embeddings : Embeddings
        LangChain Embeddings object.

    Notes
    -----
    Does not directly handle duplicate detection but offers functionality to do so.
    """

    key_manager: RedisKeyManager

    def __init__(
        self,
        redis_url: str,
        index_name: str,
        embedding: Union[Embeddings, dict] = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
            chunk_size=1500,
        ),
        index_schema: Union[Dict[str, ListOfDict], str, os.PathLike] = None,
        key_prefix: Optional[str] = None,
        key_padding: int = 4,
        counter_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize the LCRedisVectorStore.

        Parameters
        ----------
        redis_url : str
            URL of the Redis server.
        index_name : str
            Name of the Redis index.
        embedding : Union[Embeddings, dict], optional
            Embedding model or configuration, by default OpenAIEmbeddings.
        index_schema : Union[Dict[str, ListOfDict], str, os.PathLike], optional
            Schema for the Redis index, by default None.
        key_prefix : Optional[str], optional
            Prefix for keys in Redis, by default None.
        **kwargs : Any
            Additional keyword arguments.
        """
        self._redis = Redis.from_existing_index(
            redis_url=redis_url,
            embedding=embedding,
            index_name=index_name,
            schema=index_schema,
            key_prefix=key_prefix,
        )
        self.key_manager = RedisKeyManager(
            self._redis.client, key_prefix, key_padding, counter_key
        )
        self._embeddings = embedding

    async def aembed_documents(
        self,
        input: Union[list[str], list[Document]],
    ) -> list[list[float]]:
        """Generate embeddings for a list of documents asynchronously.

        Parameters
        ----------
        input : Union[list[str], list[Document]]
            List of text strings or Document objects to embed.

        Returns
        -------
        list[list[float]]
            List of embedding vectors for each document.
        """
        embeddings = await self._embeddings.aembed_documents(input)
        return embeddings

    async def aembed_query(self, query: str) -> list[float]:
        """Generate embedding for a single query string asynchronously.

        Parameters
        ----------
        query : str
            Query text to embed.

        Returns
        -------
        list[float]
            Query embedding vector.
        """
        embedding = await self._embeddings.aembed_query(query)
        return embedding

    def embed_documents(
        self, input: Union[list[str], list[list[str]], Document, list[Document]]
    ) -> list[float] | list[list[float]]:
        """Synchronously generate embeddings for a single list of chunks or multiple lists
        of chunks (e.g. multiple documents).

        Defers embedding logic to the underlying Redis object.

        Parameters
        ----------
        input : Union[list[str], list[Document]]
            List of text strings or list of Document objects to embed.

        Returns
        -------
        list[float] | list[list[float]]
            List of embedding vectors for a single list of chunks or a multiple
            lists of chunks.
        """
        if isinstance(input, Document) or (
            isinstance(input, list) and isinstance(input[0], str)
        ):
            output = self.embed_document(input)
        elif isinstance(input[0], Document):
            output = [self.embed_document(doc) for doc in input]
        elif isinstance(input[0], list):
            output = []
            for i, lst in enumerate(input):
                doc_embeddings = []
                for chunk in lst:
                    doc_embeddings.append(self.embed_document(chunk))
                output.append(doc_embeddings)
        else:
            raise ValueError(
                "Invalid input type for document embedding. "
                "Expected Document, list of strings, or list of lists of strings."
            )
        return output

    def embed_document(self, input: Union[list[str], Document]) -> list[float]:
        """Synchronously generate an embedding for a single document.

        Parameters
        ----------
        input : Union[list[str], Document]
            Text string or Document object to embed.

        Returns
        -------
        list[float]
            Embedding vector for the document.
        """
        if isinstance(input, Document):
            input = input.text_chunks
        return self._embeddings.embed_documents(input)

    def embed_query(self, query: str) -> List[float]:
        """Synchronously generate an embedding for a query.

        Parameters
        ----------
        query : str
            Query text to embed.

        Returns
        -------
        list[float]
            Embedding vector for the query.
        """
        return self._embeddings.embed_query(query)

    async def aadd_documents(self, docs: List[Document]):
        """Asynchronously add a list of documents to the vector store.

        Expects metadata to be parsed and stored in the passed Document objects.

        Parameters
        ----------
        docs : List[Document]
            List of Document objects to add.
        """
        raise NotImplementedError

    def add_documents(self, docs: list[Document]):
        """Synchronous version of add_documents.

        Expects metadata to be parsed and stored in the passed Document objects.

        Parameters
        ----------
        docs : list[Document]
            List of Document objects to add.
        """
        for i, doc in enumerate(docs):
            text_len = len(doc.text_chunks)
            embeddings = self.embed_documents(doc.text_chunks)
            e_keys = self.key_manager.generate_batch_keys(text_len)
            doc_dict = doc.to_dict()
            if doc.id is None:
                doc_dict["id"] = str(uuid.uuid4())
            self._redis.add_texts(
                texts=doc.text_chunks,
                metadatas=[doc_dict] * text_len,
                embeddings=embeddings,
                ids=e_keys,
            )


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
        """Initialize the LCPineconeVectorStore.

        Parameters
        ----------
        index_name : str
            Name of the Pinecone index.
        embedding : Embeddings, optional
            Embedding model, by default OpenAIEmbeddings.
        """
        self._pinecone = Pinecone.from_existing_index(
            embedding,
            index_name,
        )
        pass
