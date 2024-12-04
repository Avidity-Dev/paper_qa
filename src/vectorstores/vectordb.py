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
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union
import uuid

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.redis.filters import RedisFilterExpression
from langchain_core.documents import Document as LCDocument
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


class VectorStore(Protocol):
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

    def embed_query(self, query: str) -> list[float]:
        """Synchronously generate an embedding for a query.

        Not called directly by retrieval methods since the underlying LangChain object
        has its own implementation.

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
    _redis : Redis
        LangChain Redis object.
    _embeddings : Embeddings
        LangChain Embeddings object.
    key_manager : RedisKeyManager
        Redis specific KeyManager object for generating and managing entry keys.

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
            redis_client=self._redis.client,
            key_prefix=key_prefix,
            key_padding=key_padding,
            counter_key=counter_key,
        )
        self._embeddings = embedding

    @property
    def index_name(self) -> str:
        return self._redis.index_name

    @property
    def key_prefix(self) -> str:
        return self._redis.key_prefix

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
        elif isinstance(input, list) and isinstance(input[0], Document):
            output = [self.embed_document(doc) for doc in input]
        elif isinstance(input, list) and isinstance(input[0], list):
            output = []
            for i, lst in enumerate(input):
                doc_embeddings = []
                for chunk in lst:
                    doc_embeddings.append(self.embed_document(chunk))
                output.append(doc_embeddings)
        else:
            raise ValueError(
                "Invalid input type for document embedding. "
                "Expected Document, list of strings, or list of lists of strings. "
                f"Got: {type(input), type(input[0])}"
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

        Expects metadata to be parsed, text to be chunked, and stored in the passed
        Document objects.

        Parameters
        ----------
        docs : list[Document]
            List of Document objects to add.
        """
        for i, doc in enumerate(docs):
            print(f"Adding document {i} of {len(docs)}")
            text_len = len(doc.text_chunks)
            texts = [chunk.text for chunk in doc.text_chunks]
            print(f"Embedding {text_len} chunks")
            embeddings = self.embed_documents(texts)
            print(f"Generating {text_len} keys")
            e_keys = self.key_manager.generate_batch_keys(text_len)
            doc_dict = doc.to_dict()
            if doc.id is None:
                doc_dict["id"] = str(uuid.uuid4())
            self.add_texts_json(
                texts=texts,
                keys=e_keys,
                metadatas=[doc_dict] * text_len,
                embeddings=embeddings,
            )

    # TODO: Make JSON document generation more dynamic
    def add_texts_json(
        self,
        texts: list[str],
        keys: list[str],
        embeddings: list[list[float]],
        metadatas: Optional[list[dict]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Add texts to Redis using JSON storage instead of Hash storage."""
        pipeline = self._redis.client.pipeline(transaction=False)

        for i, text in enumerate(texts):
            key = keys[i]
            if not key.startswith(self.key_prefix + ":"):
                key = self.key_prefix + ":" + key

            metadata = metadatas[i] if metadatas else {}

            # Create JSON document structure
            json_doc = {
                "id": key,
                "content_chunks": [
                    {
                        "text": text,
                        "embedding": embeddings[i],
                    }
                ],
                **metadata,
            }

            pipeline.json().set(key, "$", json_doc)

        # Execute batch if batch_size reached
        if i % batch_size == 0:
            pipeline.execute()

        # Execute final batch
        pipeline.execute()

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[RedisFilterExpression] = None,
        return_metadata: bool = True,
        distance_threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """
        Search for documents using max marginal relevance with a query. Utilizes the
        underlying Redis object's implementation for initial retrieval and converts
        the results to application-specific Document objects.

        Parameters
        ----------
        query : str
            Query text to search for.
        """
        lcdocs_results: list[LCDocument] = self._redis.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, filter, return_metadata, distance_threshold
        )

        return lcdocs_results

    def clear_index_records(
        self,
        batch_size: int = 1000,
        max_retries: int = 3,
        sleep_between_batches: float = 0.1,
    ) -> tuple[int, List[str]]:
        """
        Clear all records from a Redis index without dropping the index itself.

        Parameters
        ----------
        batch_size : int
            Number of records to delete in each batch.
        max_retries : int
            Maximum number of retry attempts for failed deletions.
        sleep_between_batches : float
            Sleep time between batches in seconds.

        Returns
        -------
        Tuple of (total_deleted, failed_deletions)
        """
        total_deleted = 0
        failed_deletions = []

        while True:
            # Search for a batch of records
            try:
                result = self._redis.client.execute_command(
                    "FT.SEARCH", self.index_name, "*", "LIMIT", 0, batch_size
                )
            except Exception as e:
                print(f"Error searching index: {e}")
                break

            # Check if we found any records
            count = result[0]
            if count == 0:
                break

            # Extract document IDs (every other element starting from index 1)
            doc_ids = result[1::2]

            # Delete each document
            for doc_id in doc_ids:
                retries = 0
                while retries < max_retries:
                    try:
                        self._redis.client.delete(doc_id)
                        total_deleted += 1
                        break
                    except Exception as e:
                        retries += 1
                        if retries == max_retries:
                            print(
                                f"Failed to delete {doc_id} after {max_retries} attempts: {e}"
                            )
                            failed_deletions.append(doc_id)
                        time.sleep(0.1)  # Short sleep between retries

            # Sleep between batches to reduce server load
            time.sleep(sleep_between_batches)

            return total_deleted, failed_deletions


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
