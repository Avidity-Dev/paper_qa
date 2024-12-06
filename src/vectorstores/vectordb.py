"""Vector database adapter implementations.

This module provides adapters for various vector databases, implementing
the necessary interfaces for vector storage and retrieval operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import Field
from enum import StrEnum
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Union
import uuid

from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores.redis import Redis
from langchain_community.vectorstores.redis.filters import RedisFilterExpression
from langchain_core.documents import Document as LCDocument
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore as LCVectorStore
from langchain_openai import OpenAIEmbeddings
import numpy as np
import pandas as pd
import paperqa as pqa
from paperqa.llms import VectorStore as PQAVectorStore
from paperqa.llms import EmbeddingModel as PQAEmbeddingModel
from paperqa.types import Embeddable as PQAEmbeddable
from paperqa.types import Text as PQAText
from pydantic import BaseModel
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.models import PQADocument
from src.vectorstores.keymanager import KeyManager, RedisKeyManager

ListOfDict = List[Dict[str, Any]]

# TODO: Implement session-based embedding logic to capture failed embedding attempts

logger = logging.getLogger(__name__)


def cosine_similarity(a, b):
    norm_product = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return a @ b.T / norm_product


class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"


class VectorStore(Protocol):
    """Base class for vector stores.

    Synchronous methods should be implemented in subclasses. No obligation to implement
    asynchronous methods.
    """

    async def aembed_documents(
        self, input: Union[list[str], list[PQADocument]]
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
        self, input: Union[list[str], list[PQADocument]]
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

    def embed_document(self, input: Union[str, PQADocument]) -> list[float]:
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


# TODO: Refactor this class to use more of the Core LangChain VectorStore interface instead
#       of the Community Redis Adapter.
class LCRedisVectorStore(LCVectorStore):
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
        input: Union[list[str], list[PQADocument]],
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
        self, input: Union[list[str], list[list[str]], PQADocument, list[PQADocument]]
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
        if isinstance(input, PQADocument) or (
            isinstance(input, list) and isinstance(input[0], str)
        ):
            output = self.embed_document(input)
        elif isinstance(input, list) and isinstance(input[0], PQADocument):
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

    def embed_document(self, input: Union[list[str], PQADocument]) -> list[float]:
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
        if isinstance(input, PQADocument):
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

    async def aadd_documents(self, docs: List[PQADocument]):
        """Asynchronously add a list of documents to the vector store.

        Expects metadata to be parsed and stored in the passed Document objects.

        Parameters
        ----------
        docs : List[PQADocument]
            List of PQADocument objects to add.
        """
        raise NotImplementedError

    def add_documents(self, docs: list[PQADocument]):
        """Synchronous version of add_documents.

        Expects metadata to be parsed, text to be chunked, and stored in the passed
        Document objects.

        Parameters
        ----------
        docs : list[PQADocument]
            List of PQADocument objects to add.
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
        metadatas: Optional[list[dict]] = None,
        embeddings: Optional[list[list[float]]] = None,
        batch_size: int = 1000,
        **kwargs: Any,
    ) -> None:
        """Add texts to Redis using JSON storage instead of Hash storage."""
        pipeline = self._redis.client.pipeline(transaction=False)

        for i, text in enumerate(texts):
            key = keys[i]
            if not key.startswith(self.key_prefix + ":"):
                key = self.key_prefix + ":" + key

            # check if metadata is provided and convert none values to strings
            metadata = metadatas[i] if metadatas else {}
            metadata = {k: "" if v is None else v for k, v in metadata.items()}

            # Create JSON document structure
            json_doc = {
                "id": str(key),
                "text": text,
                "embedding": embeddings[i],
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
    ) -> List[PQADocument]:
        """
        Search for documents using max marginal relevance with a query. Utilizes the
        underlying Redis object's implementation for initial retrieval and converts
        the results to application-specific PQADocument objects.

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
        super(VectorStore, self).__init__()
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
    """Implementation of the paper-qa VectorStore interface for Redis."""

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
        self.create_index(
            self.index_name,
            self.index_definition,
            self.index_schema,
        )
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
        index_definition: Optional[IndexDefinition] = None,
        index_schema: list[Field] = DEFAULT_INDEX_SCHEMA,
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
            # Create index with the provided or default schema
            if index_definition is None:
                index_definition = self.index_definition

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
    ) -> tuple[Sequence[PQAEmbeddable], list[float]]:
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
