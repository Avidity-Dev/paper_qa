from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterable,
    Sequence,
)
import json
import time

import numpy as np
import pandas as pd
from paperqa.llms import EmbeddingModel, EmbeddingModes, VectorStore
from paperqa.types import Embeddable, LLMResult
from paperqa.utils import is_coroutine_callable
from pydantic import Field
import requests
import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query


class RedisVectorStore(VectorStore):
    texts: list[Embeddable] = Field(default_factory=list)

    def __init__(
        self,
        host: str,
        port: int,
        password: str,
        index_name: str,
        decode_responses: bool = True,
    ):
        self.client = redis.Redis(
            host=host, port=port, password=password, decode_responses=decode_responses
        )

    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        super().add_texts_and_embeddings(texts)
        # TODO: Add to redis

    async def similarity_search(
        self, query: str, embedding_model: EmbeddingModel, k: int = 10
    ) -> tuple[Sequence[Embeddable], list[float]]:
        k = min(k, len(self.texts))
        if k == 0:
            return [], []

        query_embedding = await embedding_model.embed_query(query)
        # Convert the embedding to a comma-separated string
        # Redis expects vector data in this format
        vector_str = ",".join([str(x) for x in query_embedding])

        # Perform vector similarity search
        search_query = f"*=>[KNN {k} @embedding $vector AS similarity]"

        try:
            results = self.client.ft.search(
                self.index_name,
                search_query,
                {"vector": vector_str},
                params_dict={
                    "SORTBY": "similarity",
                    "DIALECT": 2,
                },
            )

            # Process results
            similar_docs = []
            for doc in results.docs:
                similar_docs.append(
                    {
                        "id": doc.id,
                        "similarity": float(doc.similarity),
                        "content": doc.content,  # Assuming you stored document content
                    }
                )

            return similar_docs

        except Exception as e:
            print(f"Search error: {e}")
            return []


class PineconeVectorStore(VectorStore):
    pass
