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
        vector_dim: int,
        distance_metric: str,
        algorithm: str,
        schema: Optional[Union[Dict[str, Any], str, os.PathLike, list[Field]]],
        algorithm_params: Optional[Dict[str, Any]] = None,
        recreate_index: bool = False,
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
        vector_dim : int
            Dimensionality of the embedding vectors.
        distance_metric : str
            Distance metric (e.g. "COSINE").
        algorithm : str
            Vector indexing algorithm ("FLAT" or "HNSW").
        algorithm_params : Dict[str, Any]
            Additional parameters for the vector indexing algorithm.
        schema : Optional[Union[Dict[str, Any], str, os.PathLike]]
            The index schema, as a dictionary or a path to a YAML file defining fields.
            If None, a default minimal schema will be used.
        recreate_index : bool
            Whether to drop and recreate the index if it exists.
        """
        self.redis_client = redis_client
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.store_type = store_type.lower()
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric.upper()
        self.algorithm = algorithm.upper()
        self.algorithm_params = algorithm_params
        self.schema = schema
        self.recreate_index = recreate_index

        if self.algorithm not in ["FLAT", "HNSW"]:
            raise ValueError(
                f"Unsupported algorithm '{self.algorithm}'. "
                "Redis supports 'FLAT' or 'HNSW' for vector search."
            )

        logger.info(f"Store type: {self.store_type}")

    def build_index(self, **kwargs: Any) -> RedisModel:
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

        try:
            self.redis_client.ft(self.index_name).dropindex(delete_documents=True)
            logger.info(f"Dropped existing index: {self.index_name}")
        except Exception:
            pass

        # Check if index already exists
        if not self._index_exists():
            # Determine index type based on store_type
            index_type = IndexType.JSON if self.store_type == "json" else IndexType.HASH

            # Create the index definition
            definition = IndexDefinition(
                prefix=[self.key_prefix], index_type=index_type
            )

            # Create the index
            self.redis_client.ft(self.index_name).create_index(
                fields=self.schema, definition=definition
            )
            logger.info(f"Created index: {self.index_name}")
        else:
            logger.info(f"Index {self.index_name} already exists. Skipping creation.")

        return self.schema

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
        self, query_embedding: list[float], k: int, search_type: str
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
        index_name: str = "idx:doc_index",
        key_prefix: str = "doc:",
        counter_keyname: str = "doc_counter",
        vector_dim: int = 1536,
        distance_metric: str = "COSINE",
        store_type: str = "JSON",
        algorithm: str = "FLAT",
        algorithm_params: Optional[Dict[str, Any]] = None,
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
        self.counter_keyname = counter_keyname
        self.vector_dim = vector_dim
        self.distance_metric = distance_metric
        self._embedding_model = embedding
        self.store_type = store_type.lower()
        self._model: Optional[RedisModel] = None

        algorithm = algorithm.upper()
        if algorithm not in ["FLAT", "HNSW"]:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                "Redis supports 'FLAT' or 'HNSW' for vector search."
            )
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params or {}

        self.redis_client = redis.Redis.from_url(
            self.redis_url, username=redis_username, password=redis_password
        )

        self.index_builder = RedisIndexBuilder(
            redis_client=self.redis_client,
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            vector_dim=self.vector_dim,
            distance_metric=self.distance_metric,
            store_type=self.store_type,
            algorithm=self.algorithm,
            algorithm_params=self.algorithm_params,
            schema=schema,
            recreate_index=recreate_index,
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
                    "text": text,
                    "embedding": embedding.tolist(),
                }

                for k, v in clean_meta.items():
                    doc_json[k] = v
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

    def _run_knn_query(self, embedding: Union[bytes, list[float]], k: int):
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32).tobytes()

        field_prefix = "$." if self.store_type == "json" else ""
        vec_field = f"{field_prefix}{self._model.content_vector_key}"
        q = (
            Query(f"*=>[KNN {k} {vec_field} $vec_param AS score]")
            .sort_by("score")
            # Return the content field and all metadata fields
            .return_fields(self._model.content_key, *self._model.metadata_keys, "score")
            .dialect(2)
        )
        params_dict = {"vec_param": embedding}
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

        docs_with_scores = self.similarity_search_with_score(query, k=fetch_k, **kwargs)
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


class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class PQAPineconeVectorStore(PQAVectorStore, BaseModel):
    """
    Implementation of the paper-qa VectorStore interface for use with an external
    Pinecone vector database.
    """

    model_config = {"arbitrary_types_allowed": True}

    pc: Pinecone = None
    index: any = None
    texts: list[PQAEmbeddable] = []  # Keep track of texts just like NumpyVectorStore
    mmr_lambda: float = 0.9  # Default value matching paperqa

    def __init__(self, index_name: str, api_key: str, environment: str):
        """Initialize PineconeVectorStore.

        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        super(BaseModel, self).__init__()
        super(PQAVectorStore, self).__init__()
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.texts = []
        logger.info(f"Initialized PineconeVectorStore with index: {index_name}")

    def __eq__(self, other) -> bool:
        """Implement equality comparison like NumpyVectorStore."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.texts == other.texts
            and self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
        )

    def add_texts_and_embeddings(self, texts: Iterable[PQAEmbeddable]) -> None:
        """Add texts and their embeddings to Pinecone."""
        super().add_texts_and_embeddings(texts)  # Update texts_hashes
        texts_list = list(texts)
        self.texts.extend(texts_list)  # Keep local record of texts

        logger.info(f"Adding {len(texts_list)} texts to Pinecone")

        # Prepare vectors for Pinecone
        upserts = []
        for text in texts_list:
            if text.embedding is None:
                logger.warning(f"Text {text.name} has no embedding!")
                continue

            # Convert embedding to list if it's numpy array
            embedding = (
                text.embedding.tolist()
                if hasattr(text.embedding, "tolist")
                else text.embedding
            )

            upserts.append(
                {
                    "id": text.name,
                    "values": embedding,
                    "metadata": {"text": text.text, "name": text.name},
                }
            )

        if upserts:
            try:
                self.index.upsert(vectors=upserts)
                logger.info(f"Successfully upserted {len(upserts)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Failed to upsert vectors: {str(e)}")
                raise

    async def similarity_search(
        self, query: str, k: int, embedding_model: PQAEmbeddingModel
    ) -> tuple[Sequence[PQAEmbeddable], list[float]]:
        logger.info(f"Starting similarity search for query: {query[:50]}...")

        k = min(k, len(self.texts))
        if k == 0:
            logger.info("No texts to search through")
            return [], []

        try:
            # Get query embedding
            embedding_model.set_mode(EmbeddingModes.QUERY)
            query_embedding = (await embedding_model.embed_documents([query]))[0]
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)
            logger.info("Generated query embedding")

            # Convert to list if it's numpy array
            if hasattr(query_embedding, "tolist"):
                query_embedding = query_embedding.tolist()

            # Use Pinecone's native query
            query_response = self.index.query(
                vector=query_embedding, top_k=k, include_metadata=True
            )

            matches = query_response.matches
            logger.info(f"Found {len(matches)} matches in Pinecone")

            # Convert Pinecone results to Embeddable objects
            results = []
            scores = []

            for match in matches:
                # Find the original document in self.texts
                original_doc = next(
                    t for t in self.texts if t.name == match.metadata["name"]
                )

                # Create new Text object (which extends Embeddable)
                text_obj = PQAText(
                    text=match.metadata["text"],
                    name=match.metadata["name"],
                    embedding=original_doc.embedding,
                    doc=original_doc.doc,  # This should now work since we're using Text instead of Embeddable
                )
                results.append(text_obj)
                scores.append(match.score)

            logger.info(f"Returning {len(results)} results")
            logger.info(f"Top similarity scores: {scores[:3] if scores else []}")

            return results, scores

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return [], []

    async def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, embedding_model: PQAEmbeddingModel
    ) -> tuple[Sequence[PQAEmbeddable], list[float]]:
        """Implement MMR search using parent VectorStore's implementation.

        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of results to fetch before MMR
            embedding_model: Model to use for embedding the query

        Returns:
            Tuple of (list of Embeddable results, list of similarity scores)
        """
        logger.info(f"Starting MMR search with k={k}, fetch_k={fetch_k}")

        if fetch_k < k:
            logger.warning(
                f"fetch_k ({fetch_k}) must be >= k ({k}), adjusting fetch_k to {k}"
            )
            fetch_k = k

        # Use parent implementation which relies on our similarity_search
        return await super().max_marginal_relevance_search(
            query, k, fetch_k, embedding_model
        )

    def clear(self) -> None:
        """Clear all data from the store."""
        try:
            self.index.delete(delete_all=True)
            self.texts.clear()
            super().clear()  # Clear the base class's texts_hashes set
            logger.info("Successfully cleared all texts and embeddings")
        except Exception as e:
            logger.error(f"Error clearing store: {e}")
            raise


class PQARedisVectorStore:
    """
    Provides functionality for storing and querying documents in a Redis vector
    database, using paper-qa objects for interaction.
    """

    redis_client: redis.Redis = None
    redis_url: str = None
    index_name: str = "idx:doc_chunks_v1"
    key_prefix: str = "doc_chunks:"
    counter_key: str = "doc_chunks_ctr"
    mmr_lambda: float = 0.9
    vector_dim: int = 1536
    distance_metric: str = "COSINE"
    index_schema: list[Field] = None
    index_type: IndexType = IndexType.JSON
    embedding_model: PQAEmbeddingModel = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    key_manager: KeyManager = None

    DEFAULT_INDEX_SCHEMA: list[Field] = [
        TextField("$.text", no_stem=True, as_name="text"),
        TextField("$.name", no_stem=True, as_name="name"),
        VectorField(
            "$.embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": 1536,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        ),
    ]

    def __init__(
        self,
        redis_url: str,
        key_prefix: str = "doc_chunks:",
        index_schema: list[Field] = DEFAULT_INDEX_SCHEMA,
    ):
        """Initialize RedisVectorStore.

        Parameters
        ----------
        redis_url : str
            Redis connection URL
        index_name : str
            Name of the Redis index
        vector_dim : int, optional
            Dimension of embedding vectors, by default 1536
        distance_metric : str, optional
            Distance metric for vector similarity, by default "COSINE"
        index_definition : Optional[IndexDefinition], optional
            Custom index definition, by default None
        """
        self.redis_url = redis_url
        self.index_schema = index_schema
        self._index_definition = IndexDefinition(
            prefix=[key_prefix], index_type=self.index_type
        )
        self.redis_client = redis.from_url(self.redis_url)

        self.key_manager = RedisKeyManager(
            redis_client=self.redis_client,
            index_name=self.index_name,
            key_prefix=self.key_prefix,
            counter_key=self.counter_key,
        )

    @property
    def index_definition(self) -> IndexDefinition:
        return self._index_definition

    def create_index(
        self,
        index_name: str,
        index_definition: IndexDefinition,
        index_schema: list[Field],
    ) -> None:
        """Create a Redis index with a dynamic schema if it doesn't exist.

        Parameters
        ----------
        index_name : str
            Name of the Redis index
        vector_dim : int
            Dimension of embedding vectors
        distance_metric : str, optional
            Distance metric for vector similarity, by default "COSINE"
        index_definition : Optional[IndexDefinition], optional
            Custom index definition, by default None

        Notes
        -----
        If no custom index definition is provided, a default schema including all
        fields from the Document model is used.
        """
        try:
            self.redis_client.ft(index_name).create_index(
                fields=index_schema,
                definition=index_definition,
            )
            logger.info(
                f"Successfully created index '{index_name}' with {index_schema} fields"
            )

        # Just inform the user if the index already exists, no need to raise an error
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                logger.warning(f"Index '{index_name}' already exists")

        except Exception as e:
            logger.error(f"Unexpected error creating index: {str(e)}")
            raise

    def add_texts_and_embeddings(self, docs: Iterable[PQADocument]) -> list[str]:
        """Add texts and their embeddings for each document in the iterable.

        Returns:
            List of keys for the added documents.
        """
        docs_list = list(docs)
        keys = []

        # logger.info(f"Adding {len(docs_list)} texts to Redis")

        pipeline = self.redis_client.pipeline(transaction=False)

        for doc in docs_list:
            for i, text in enumerate(doc.text_chunks):
                key = self.key_manager.get_next_key()
                keys.append(key)

                # Add the document ID and DocKey.
                # These are the same for now, but we may want to change this in the future.
                doc.id = key
                doc.dockey = key

                json_doc = {
                    "text": text,
                    "name": key,
                    "embedding": np.array(doc.embeddings[i], dtype=np.float32).tolist(),
                }

                pipeline.json().set(key, "$", json_doc)

        try:
            pipeline.execute()
            logger.info(f"Successfully added vectors to Redis")
            return keys
        except Exception as e:
            logger.error(f"Failed to add vectors: {str(e)}")
            raise

    async def similarity_search(
        self, query: str, k: int, embedding_model: PQAEmbeddingModel
    ) -> tuple[Sequence[PQAEmbeddable], list[float]]:
        """Perform similarity search using Redis."""
        logger.info(f"Starting similarity search for query: {query[:50]}...")

        try:
            # Get query embedding
            embedding_model.set_mode(EmbeddingModes.QUERY)
            query_embedding = (await embedding_model.embed_documents([query]))[0]
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)

            # Convert to numpy array and then to list
            query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

            # Construct Redis vector similarity query
            q = (
                Query(f"*=>[KNN {k} @embedding $query_vector AS score]")
                .sort_by("score")
                .return_fields("text", "name", "score")
                .dialect(2)
            )

            # Execute search
            results = self.redis_client.ft(self.index_name).search(
                q, query_params={"query_vector": query_vector}
            )

            # Process results
            matches = []
            scores = []

            for doc in results.docs:
                # Get the original embedding
                embedding = self.redis_client.json().get(doc.id, "$.embedding")[0]
                text_obj = PQAText(
                    text=doc.text,
                    name=doc.name,
                    doc=PQADocument(
                        id=doc.name,
                        dockey="",
                        citation="",
                    ),
                    embedding=embedding,
                )
                matches.append(text_obj)
                scores.append(float(doc.score))

            logger.info(f"Returning {len(matches)} results")
            logger.info(f"Top similarity scores: {scores[:3] if scores else []}")

            return matches, scores

        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return [], []

    async def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, embedding_model: PQAEmbeddingModel
    ) -> tuple[Sequence[PQAText], list[float]]:
        """Vectorized implementation of Maximal Marginal Relevance (MMR) search.

        Args:
            query: Query vector.
            k: Number of results to return.
            fetch_k: Number of results to fetch from the vector store.
            embedding_model: model used to embed the query

        Returns:
            List of tuples (doc, score) of length k.
        """
        if fetch_k < k:
            raise ValueError("fetch_k must be greater or equal to k")

        texts, scores = await self.similarity_search(query, fetch_k, embedding_model)
        if len(texts) <= k or self.mmr_lambda >= 1.0:
            return texts, scores

        embeddings = np.array([t.embedding for t in texts])
        np_scores = np.array(scores)
        similarity_matrix = cosine_similarity(embeddings, embeddings)

        selected_indices = [0]
        remaining_indices = list(range(1, len(texts)))

        while len(selected_indices) < k:
            selected_similarities = similarity_matrix[:, selected_indices]
            max_sim_to_selected = selected_similarities.max(axis=1)

            mmr_scores = (
                self.mmr_lambda * np_scores
                - (1 - self.mmr_lambda) * max_sim_to_selected
            )
            mmr_scores[selected_indices] = -np.inf  # Exclude already selected documents

            max_mmr_index = mmr_scores.argmax()
            selected_indices.append(max_mmr_index)
            remaining_indices.remove(max_mmr_index)

        return [texts[i] for i in selected_indices], [
            scores[i] for i in selected_indices
        ]

    def clear(self) -> None:
        """Clear all data from the store but keep the index."""
        try:
            # Delete all keys with the index prefix
            keys = self.redis_client.keys(f"{self.index_name}:*")
            if keys:
                self.redis_client.delete(*keys)

            self.texts.clear()
            super().clear()
            logger.info("Successfully cleared all texts and embeddings")
        except Exception as e:
            logger.error(f"Error clearing store: {e}")
            raise
