"""Vector database adapter implementations.

This module provides adapters for various vector databases, implementing
the necessary interfaces for vector storage and retrieval operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import Field, dataclass
from enum import StrEnum
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Tuple, Union
import uuid

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.redis.base import _prepare_metadata
from langchain_community.vectorstores.redis.filters import RedisFilterExpression
from langchain_community.vectorstores.redis.schema import RedisModel, read_schema
from langchain_core.documents import Document, Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.vectorstores import VectorStore as LCVectorStore
from langchain_openai import OpenAIEmbeddings
import numpy as np
import pandas as pd
import paperqa as pqa
from paperqa.llms import (
    EmbeddingModel as PQAEmbeddingModel,
    VectorStore as PQAVectorStore,
)
from paperqa.types import Embeddable as PQAEmbeddable, Text as PQAText
from pydantic import BaseModel
import redis
from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.models import PQADocument
from src.storage.vector.errors import EmbeddingError
from src.storage.vector.keymanagers import KeyManager, RedisKeyManager

ListOfDict = List[Dict[str, Any]]

# TODO: Implement session-based embedding logic to capture failed embedding attempts

logger = logging.getLogger(__name__)


@dataclass
class VectorStoreSearchResult:
    """Result of a search operation on a vector store."""

    text: str
    embedding: list[float]
    metadata: dict
    score: float


logger = logging.getLogger(__name__)


class IndexBuilder(ABC):
    """Interface for building a search index for a vector store."""

    @abstractmethod
    def build_index(self, **kwargs: Any) -> None:
        """Build the search index using a given schema and configuration."""


class RedisIndexBuilder(IndexBuilder):
    """Index builder for Redis-based vector stores that leverages the RedisModel from the langchain_community library."""

    def __init__(
        self,
        redis_client: redis.Redis,
        index_name: str,
        key_prefix: str,
        store_type: str,
        schema: Optional[Union[Dict[str, List[Any]], str, os.PathLike]],
    ):
        """
        Parameters
        ----------
        redis_client : redis.Redis
            Redis client connection.
        index_name : str
            Name of the Redis index.
        key_prefix : str
            Key prefix for documents in Redis.
        store_type : str
            "hash" or "json".
        schema : Optional[Union[Dict[str, List[Any]], str, os.PathLike]]
            The index schema, as a dictionary or a path to a YAML file defining fields.
            If None, a default minimal schema will be used.
        recreate_index : bool
            Whether to drop and recreate the index if it exists.
        """
        self.redis_client = redis_client
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.store_type = store_type.lower()
        self.schema = schema

    def build_index(
        self, recreate_index: bool = False, **kwargs: Any
    ) -> Optional[RedisModel]:
        """
        Build the Redis index using the RedisModel and schema logic provided by langchain_community.

        Steps:
        1. Parse the schema from a dict or YAML using `read_schema`.
        2. Initialize a RedisModel with the parsed schema.
        3. Add the vector field to the schema if it's not present.
        4. Drop the index if recreate_index=True.
        5. Create the index if it does not already exist.

        Returns
        -------
        RedisModel
            The RedisModel built from the schema.
        """

        if recreate_index:
            try:
                self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
                logger.info(f"Dropped existing index: {self.index_name}")
            except Exception:
                logger.info(f"Index {self.index_name} does not exist. Skipping drop.")
                return None

        # Need to extract the vector definition from the schema since the langchain
        # library expects the index definition and vector definitions to be passed
        # in separately.
        schema = read_schema(self.schema)
        vector_schema = schema.pop("vector")[0]
        content_key = schema.pop("content_key")
        content_vector_key = schema.pop("content_vector_key")

        model = RedisModel(**schema)
        model.add_vector_field(vector_schema)
        model.content_key = content_key
        model.content_vector_key = content_vector_key
        fields = model.get_fields()

        # Determine index type based on store_type
        index_type = IndexType.JSON if self.store_type == "json" else IndexType.HASH

        # Create the index definition
        definition = IndexDefinition(prefix=[self.key_prefix], index_type=index_type)

        if not self._index_exists():

            # Create the index
            self.redis_client.ft(self.index_name).create_index(
                fields=fields, definition=definition
            )
            logger.info(
                f"Created index: {self.index_name}. Store type: {self.store_type}"
            )
        else:
            logger.info(f"Index {self.index_name} already exists. Skipping creation.")

        return model

    def _index_exists(self) -> bool:
        """Check if a Redis index already exists."""
        try:
            self.redis_client.ft(self.index_name).info()
            return True
        except Exception:
            return False


class BaseVectorStore(Protocol):
    """Contract for a vector store in this application to allow for the implementations
    to be swapped out for different vector databases.
    """

    def add_texts(self, texts: List[str], **kwargs: Any) -> List[str]: ...

    def delete(
        self, ids: Optional[list[str]] = None, **kwargs: Any
    ) -> Optional[bool]: ...

    def search(
        self,
        query_embedding: Union[list[float], str],
        k: int,
        search_type: str,
        **kwargs: Any,
    ) -> List[Any]: ...

    def add_documents(self, documents: List[Any], **kwargs: Any) -> List[str]: ...


class RedisVectorStore(LCVectorStore, BaseVectorStore):
    """
    A Redis-based vector store inspired by the LangChain Community RedisVectorStore.

    This implementation uses the RedisModel and IndexBuilder classes to manage the Redis
    index, allowing for more flexibility in the types of indexes that can be created.

    Attributes
    ----------
    redis_url : str
        The URL of the Redis server.
    embedding : Embeddings
        The embedding model to use for encoding documents.
    index_name : str
        The name of the Redis index.
    """

    def __init__(
        self,
        redis_url: str,
        embedding: Optional[Embeddings] = None,
        index_name: str = "idx:docs",
        store_type: str = "json",
        key_prefix: str = "doc:",
        counter_keyname: str = "docs:counter",
        schema: Optional[Union[Dict[str, Any], str, os.PathLike]] = None,
        redis_username: Optional[str] = None,
        redis_password: Optional[str] = None,
        recreate_index: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize the RedisVectorStore.

        Parameters
        ----------
        redis_url : str
            The URL of the Redis server.
        embedding : Optional[Embeddings], optional
            The embedding model to use for encoding documents.
        index_name : _type_, optional
            The name of the Redis index.
        key_prefix : str, optional
            The prefix for the keys in the Redis index.
        counter_keyname : str, optional
            The name of the Redis key to store the counter.
        vector_dim : int, optional
            The dimensionality of the embedding vectors.
        distance_metric : str, optional
            The distance metric to use for vector search.
        store_type : str, optional
            _description_, by default "hash"
        algorithm : str, optional
            The vector indexing algorithm to use.
        algorithm_params : Optional[Dict[str, Any]], optional
            Additional parameters for the vector indexing algorithm.
        schema : Optional[Union[Dict[str, Any], str, os.PathLike]], optional
            _description_, by default None
        index_builder : Optional[IndexBuilder], optional
            The index builder to use for creating the Redis index.
        redis_username : Optional[str], optional
            The username for the Redis server.
        redis_password : Optional[str], optional
            The password for the Redis server.
        recreate_index : bool, optional
            Whether to recreate the Redis index if it already exists.

        Raises
        ------
        ValueError
            If the algorithm is not supported.
        ValueError
            If the number of ids does not match the number of texts.
        """
        self.redis_url = redis_url
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.store_type = store_type
        self.counter_keyname = counter_keyname
        self._embedding_model = embedding
        self._model: Optional[RedisModel] = None

        self.redis_client = redis.Redis.from_url(
            self.redis_url, username=redis_username, password=redis_password
        )

        self.index_builder = RedisIndexBuilder(
            redis_client=self.redis_client,
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            store_type=self.store_type,
            schema=schema,
        )

        self.key_manager = RedisKeyManager(
            redis_client=self.redis_client,
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            counter_key=self.counter_keyname,
        )

        # Build the index and get the RedisModel
        self._model = self.index_builder.build_index()
        if not isinstance(self.index_builder, RedisIndexBuilder):
            raise ValueError("IndexBuilder must be a RedisIndexBuilder or compatible.")

    @property
    def client(self) -> redis.Redis:
        return self.redis_client

    @property
    def embedding_model(self) -> Optional[Embeddings]:
        return self._embedding_model

    def delete(self, ids: Optional[list[str]] = None, **kwargs: Any) -> Optional[bool]:
        if ids is None:
            keys = self.redis_client.keys(f"{self.key_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            return True

        for doc_id in ids:
            self.redis_client.delete(doc_id)
        return True

    def drop_index(self, delete_documents: bool = True) -> None:
        self.redis_client.ft(self.index_name).dropindex(
            delete_documents=delete_documents
        )

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        *,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> List[str]:
        if metadatas and len(metadatas) != len(texts):
            raise ValueError("Number of metadatas must match number of texts.")

        if "embeddings" not in kwargs:
            kwargs["embeddings"] = await self.embedding_model.aembed_documents(texts)

        embeddings = kwargs["embeddings"]

        keys = self.key_manager.generate_batch_keys(len(texts))
        pipeline = self.redis_client.pipeline(transaction=False)

        for i, text in enumerate(texts):
            doc_id = keys[i]
            metadata = metadatas[i] if metadatas else {}
            # Clean and prepare metadata fields according to Redis indexing rules
            clean_meta = _prepare_metadata(metadata)

            # Prepare vector field
            embedding = np.array(embeddings[i], dtype=np.float32)
            if self.store_type == "json":
                # For JSON, we store everything as a JSON object
                doc_json = {
                    self._model.content_key: text,
                    self._model.content_vector_key: embedding.tolist(),
                }

                for k, v in clean_meta.items():
                    doc_json[k] = v
                logger.debug(f"Storing document: {doc_json}")
                pipeline.json().set(doc_id, "$", doc_json)
            else:
                # For hash, store each field as a separate key-value
                mapping = {
                    self._model.content_key: text,
                    self._model.content_vector_key: embedding.tobytes(),
                }
                # Add metadata fields
                for k, v in clean_meta.items():
                    mapping[k] = v
                pipeline.hset(doc_id, mapping=mapping)

            if i % batch_size == 0:
                pipeline.execute()

        pipeline.execute()
        return keys

    def get_embeddings(self, ids: list[str]) -> list[list[float]]:
        embeddings = []
        for id in ids:
            embedding = self.redis_client.json().get(id, self._model.content_vector_key)
            embeddings.append(embedding)
        return embeddings

    def _run_knn_query(self, embedding: list[float], k: int):
        if isinstance(embedding, list):
            # Redis expects a byte representation
            embedding = np.array(embedding, dtype=np.float32).tobytes()

        logger.debug(f"Running KNN query with {k} results...")
        vec_field = self._model.content_vector_key
        logger.debug(f"Retrieved vector field: {vec_field}")
        metadata_keys = [k for k in self._model.metadata_keys]
        split_keys = [k.split(".")[1] for k in metadata_keys]
        split_keys.remove(vec_field)

        logger.debug(f"Building query using metadata keys: {split_keys}")
        q = (
            Query(f"(*)=>[KNN {k} @{vec_field} $vec_param AS score]")
            .sort_by("score")
            .return_fields(*split_keys, "score")
            .dialect(2)
        )

        logger.debug(f"Query string: {q.query_string()}")
        params_dict = {"vec_param": embedding}
        logger.debug(f"Params: {params_dict}")
        logger.debug(f"Executing query on index: {self.index_name}...")
        return self.redis_client.ft(self.index_name).search(q, query_params=params_dict)

    async def asimilarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        return await run_in_executor(
            None, self.similarity_search_with_score, query, k, **kwargs
        )

    def similarity_search_with_score(
        self, query: Union[str, List[float]], k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        # If the query is a string, we need to embed it. Otherwise, we assume that the
        # query is already an embedding.
        query_embedding: List[float] = (
            self.embedding_model.embed_query(query) if isinstance(query, str) else query
        )
        res = self._run_knn_query(query_embedding, k)

        docs = []
        if self.store_type == "json":
            # For JSON, get full JSON object and extract fields
            for doc in res.docs:
                doc_json = self.redis_client.json().get(doc.id, "$")[0]
                text = doc_json.get(self._model.content_key, "")
                metadata = {}
                for mk in self._model.metadata_keys:
                    metadata[mk] = doc_json.get(mk, None)
                score = float(doc.score)
                d = Document(page_content=text, metadata=metadata, id=doc.id)
                docs.append((d, score))
        else:
            # For hash, fields are directly on doc object
            # doc.<field> is how redis-py search returns fields
            # We can use doc.get(...) to fetch fields
            for doc in res.docs:
                text = doc.get(self._model.content_key, "")
                metadata = {}
                for mk in self._model.metadata_keys:
                    val = doc.get(mk, None)
                    # Attempt to parse JSON if it's a string with json data
                    if val is not None and isinstance(val, str):
                        # If val looks like a JSON dump, try to parse
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, dict) or isinstance(parsed, list):
                                val = parsed
                        except:
                            # not JSON, keep val as is
                            pass
                    metadata[mk] = val
                score = float(doc.score)
                d = Document(page_content=text, metadata=metadata, id=doc.id)
                docs.append((d, score))

        return docs

    async def asimilarity_search(
        self, query: Union[str, List[float]], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_with_scores = await self.asimilarity_search_with_score(query, k, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search(
        self, query: Union[str, List[float]], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        docs_with_scores = self.similarity_search_with_score(query, k=k, **kwargs)
        return [doc for doc, _ in docs_with_scores]

    def similarity_search_by_vector(
        self, embedding: list[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        res = self._run_knn_query(embedding, k)
        docs = []
        if self.store_type == "json":
            for doc in res.docs:
                doc_json = self.redis_client.json().get(doc.id, "$")[0]
                text = doc_json.get(self._model.content_key, "")
                metadata = {}
                for mk in self._model.metadata_keys:
                    metadata[mk] = doc_json.get(mk, None)
                d = Document(page_content=text, metadata=metadata, id=doc.id)
                docs.append(d)
        else:
            for doc in res.docs:
                text = doc.get(self._model.content_key, "")
                metadata = {}
                for mk in self._model.metadata_keys:
                    val = doc.get(mk, None)
                    # Attempt JSON parse
                    if val is not None and isinstance(val, str):
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, (dict, list)):
                                val = parsed
                        except:
                            pass
                    metadata[mk] = val
                d = Document(page_content=text, metadata=metadata, id=doc.id)
                docs.append(d)
        return docs

    def max_marginal_relevance_search(
        self,
        query: Union[str, List[float]],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        embedding_model: Optional[Embeddings] = None,
        **kwargs: Any,
    ) -> List[Document]:

        # If the query is a string, we need to embed it. If the store instance has no
        # embedding model, then an embedding model must be passed in the kwargs.
        try:
            if isinstance(query, str):
                query_embedding = embedding_model.embed_query(query)
            else:
                query_embedding = query
        except:
            raise EmbeddingError(
                "Failed to embed query. If a query string is provided, ensure that "
                "the embedding model is correctly configured within the instance or "
                "provided in the kwargs."
            )

        docs_with_scores = self.similarity_search_with_score(
            query_embedding, k=fetch_k, **kwargs
        )
        if not docs_with_scores:
            return []

        docs = [d for d, _ in docs_with_scores]

        doc_texts = [d.page_content for d in docs]
        doc_embeddings = np.array(
            self._embeddings.embed_documents(doc_texts), dtype=np.float32
        )

        def normalize(v):
            norm = np.linalg.norm(v)
            return v / norm if norm != 0 else v

        query_norm = normalize(query_embedding)
        doc_norms = np.linalg.norm(doc_embeddings, axis=1)
        doc_normed = doc_embeddings / doc_norms[:, None]
        sim_to_query = doc_normed @ query_norm
        sim_matrix = doc_normed @ doc_normed.T

        selected_indices = [int(np.argmax(sim_to_query))]
        remaining_indices = set(range(len(docs))) - set(selected_indices)

        while len(selected_indices) < k and remaining_indices:
            best_score = -np.inf
            best_idx = None
            for idx in remaining_indices:
                max_sim_to_selected = np.max(sim_matrix[idx, selected_indices])
                mmr_score = (
                    lambda_mult * sim_to_query[idx]
                    - (1 - lambda_mult) * max_sim_to_selected
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        return [docs[i] for i in selected_indices]

    def _select_relevance_score_fn(self) -> Callable[[float], float]:
        return self._cosine_relevance_score_fn

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        redis_url: str = "redis://localhost:6379",
        index_name: str = "doc_index",
        key_prefix: str = "doc:",
        vector_dim: int = 1536,
        distance_metric: str = "COSINE",
        store_type: str = "hash",
        algorithm: str = "FLAT",
        algorithm_params: Optional[Dict[str, Any]] = None,
        schema: Optional[Dict[str, Any]] = None,
        recreate_index: bool = False,
        index_builder: Optional[IndexBuilder] = None,
        **kwargs: Any,
    ) -> "RedisVectorStore":
        store = cls(
            redis_url=redis_url,
            embedding=embedding,
            index_name=index_name,
            key_prefix=key_prefix,
            vector_dim=vector_dim,
            distance_metric=distance_metric,
            store_type=store_type,
            algorithm=algorithm,
            algorithm_params=algorithm_params,
            schema=schema,
            recreate_index=recreate_index,
            index_builder=index_builder,
            **kwargs,
        )
        store.add_texts(texts, metadatas=metadatas, ids=ids)
        return store


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product
