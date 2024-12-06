import os
import yaml
from paperqa.types import Text
import pytest

import dotenv
from dataclasses import dataclass
from src.models import PQADocument
from src.vectorstores.keymanager import RedisKeyManager
from src.vectorstores.vectordb import LCRedisVectorStore, PQARedisVectorStore


def get_matching_doc_ids(
    redis: LCRedisVectorStore, index_name: str, query: str
) -> list[str]:
    """
    Extract document IDs from a RediSearch query result

    Args:
        redis_client: Redis client instance
        query: Search query string

    Returns:
        List of document IDs that matched the search
    """
    # Execute search
    result = redis._redis.client.execute_command("FT.SEARCH", index_name, query)

    # First element is count, remaining elements alternate between ID and data
    # We want every other element starting from index 1
    doc_ids = result[1::2]

    return doc_ids


@pytest.fixture
def local_redis_dict() -> dict:
    dotenv.load_dotenv()
    config_path = os.getenv("REDIS_VECTOR_CONFIG")
    with open(config_path, "r") as f:
        index_config = yaml.safe_load(f)

    # TODO: Make this more dynamic by importing from ConfigurationManager
    vector_config = {
        "index_schema": index_config,
        "index_name": "idx:docs_vss",
        "key_prefix": "docs",
        "counter_key": "docs_ctr",
        "redis_url": "redis://localhost:6379",
    }
    print(f"vector_config: {vector_config}")
    return vector_config


# Ensure that the index is created locally and redis is running before test
def test_vector_db_init(local_redis_dict: dict):
    vector_db = LCRedisVectorStore(**local_redis_dict)
    assert vector_db is not None


def test_vector_db_init_yaml():
    with open("src/config/vector.yaml", "r") as f:
        index_config = yaml.safe_load(f)

    vector_db = LCRedisVectorStore(
        index_schema=index_config,
        index_name="idx:docs_vss",
        key_prefix="docs",
        counter_key="docs_ctr",
        redis_url="redis://localhost:6379",
    )
    assert vector_db is not None
    assert vector_db.index_name == "idx:docs_vss"


@pytest.fixture
def local_lcredis_vector_db(local_redis_dict: dict) -> LCRedisVectorStore:
    return LCRedisVectorStore(**local_redis_dict)


def test_embed_chunks(
    chunked_text_strings: list[list[str]], local_redis_vector_db: LCRedisVectorStore
):
    # Only test single doc for now
    chunked_text_strings = chunked_text_strings[0]
    embeddings = local_redis_vector_db.embed_documents(chunked_text_strings)
    assert len(embeddings) == len(chunked_text_strings)


@pytest.mark.integration
def test_add_documents(
    doc_objects: list[PQADocument], local_redis_vector_db: LCRedisVectorStore
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
    local_redis_vector_db.clear_index_records()


def test_create_pqaredisvectorstore():
    pqa_vector_db = PQARedisVectorStore(
        redis_url="redis://localhost:6379",
    )
    assert pqa_vector_db is not None

    # flush the db
    pqa_vector_db.redis_client.flushall()
