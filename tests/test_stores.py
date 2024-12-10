import os
from langchain_openai import OpenAIEmbeddings
import yaml
from paperqa.types import Text
import pytest

import dotenv
from dataclasses import dataclass, Field
from redis.commands.search.field import TextField, VectorField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition
from src.models import PQADocument
from src.storage.vector.keymanagers import RedisKeyManager
from src.storage.vector.stores import (
    PQARedisVectorStore,
    RedisVectorStore,
)
from src.config.config import INDEX_SCHEMA


@pytest.fixture
def test_chunks() -> list[str]:
    return ["test chunk 1", "test chunk 2", "test chunk 3"]


@pytest.fixture
def test_schema() -> dict:
    return {
        "text": [
            {
                "name": "$.text",
                "weight": 1.0,
                "no_stem": False,
                "sortable": False,
                "as_name": "text",
            }
        ],
        "vector": [
            {
                "name": "$.embedding",
                "algorithm": "FLAT",
                "datatype": "FLOAT32",
                "dims": 1536,
                "distance_metric": "COSINE",
                "as_name": "embedding",
            }
        ],
    }


def test_build_index():
    local_vector_db = RedisVectorStore(
        redis_url="redis://localhost:6379",
        index_name="idx:test",
        key_prefix="test:",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        schema=os.getenv("REDIS_VECTOR_CONFIG_PATH"),
    )

    assert local_vector_db is not None
    assert (
        local_vector_db.redis_client.execute_command(
            f"FT.INFO {local_vector_db.index_name}"
        )
        is not None
    )

    local_vector_db.drop_index()


@pytest.mark.asyncio
async def test_add_texts(test_chunks: list[str], test_schema: dict):
    local_vector_db = RedisVectorStore(
        redis_url="redis://localhost:6379",
        index_name="idx:test",
        key_prefix="test:",
        counter_key="test_ctr",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        schema=test_schema,
    )

    metadatas = [
        {"name": "test name 1"},
        {"name": "test name 2"},
        {"name": "test name 3"},
    ]

    ids = await local_vector_db.add_texts(test_chunks, metadatas)
    assert len(ids) == len(test_chunks)
    assert all(id.startswith("test:") for id in ids)
    local_vector_db.redis_client.execute_command("FLUSHALL")
