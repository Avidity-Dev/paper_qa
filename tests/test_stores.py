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


def test_build_index():
    schema = os.getenv("REDIS_VECTOR_CONFIG")
    local_vector_db = RedisVectorStore(
        redis_url="redis://localhost:6379",
        index_name="idx:docs",
        key_prefix="doc:",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        schema=INDEX_SCHEMA,
    )


@pytest.mark.asyncio
async def test_add_texts(test_chunks: list[str]):
    local_vector_db = RedisVectorStore(
        redis_url="redis://localhost:6379",
        index_name="idx:test",
        key_prefix="test:",
        counter_key="test_ctr",
        embedding=OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        schema=[
            TextField(name="$.text", no_stem=True, as_name="text"),
            VectorField(
                name="$.embedding",
                algorithm="FLAT",
                attributes={
                    "TYPE": "FLOAT32",
                    "DIM": 1536,
                    "DISTANCE_METRIC": "COSINE",
                },
                as_name="embedding",
            ),
        ],
    )

    metadatas = [
        {"name": "test name 1"},
        {"name": "test name 2"},
        {"name": "test name 3"},
    ]

    ids = await local_vector_db.add_texts(test_chunks, metadatas)
    assert len(ids) == len(test_chunks)
    assert all(id.startswith("test:") for id in ids)


@pytest.mark.integration
def test_add_documents(
    doc_objects: list[PQADocument], local_redis_vector_db: RedisVectorStore
):
    titles = ["TestTitle1", "TestTitle2", "TestTitle3"]
    # Add titles to docs
    for doc, title in zip(doc_objects, titles):
        doc.title = title

    # clear any preexisting test docs
    local_redis_vector_db.clear_index_records()

    # now add new docs
    local_redis_vector_db.add_documents(doc_objects)

    # Check that the docs were added
    result = local_redis_vector_db._redis.client.execute_command(
        "FT.SEARCH", local_redis_vector_db.index_name, "*"
    )

    doc_ids = result[1::2]
    # Check that the doc IDs match the expected titles and have embeddings
    assert len(doc_ids) == len(doc_objects)
    assert all(doc.id in doc_ids for doc in doc_objects)
    assert all(doc.embedding is not None for doc in doc_objects)

    # Clear the index
    local_redis_vector_db.drop_index()


def test_create_pqaredisvectorstore():
    pqa_vector_db = PQARedisVectorStore(
        redis_url="redis://localhost:6379",
    )
    assert pqa_vector_db is not None

    # flush the db
    pqa_vector_db.redis_client.flushall()
