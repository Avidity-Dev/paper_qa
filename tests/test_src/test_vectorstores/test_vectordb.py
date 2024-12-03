import pytest

from dataclasses import dataclass
from src.vectorstores.keymanager import RedisKeyManager
from src.vectorstores.vectordb import LCRedisVectorStore


@pytest.fixture
def vector_db():
    return LCRedisVectorStore(redis_url="redis://localhost:6379", key_prefix="doc")
