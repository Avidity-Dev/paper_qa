import os
import pytest
from paperqa.types import ParsedText, Text
from paperqa.settings import Settings as PQASettings
from unittest.mock import MagicMock, patch

import dotenv
import numpy as np

from src.config.config import ConfigurationManager, AppConfig
from src.process.processors import PQAProcessor
from src.storage.vector.stores import RedisVectorStore

dotenv.load_dotenv()

local_llm_config = dict(
    model_list=[
        dict(
            model_name="claude-3-5-sonnet-20240620",
            litellm_params=dict(
                model="claude-3-5-sonnet-20240620",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                api_base="https://api.anthropic.com/v1/messages",
            ),
        )
    ]
)


@pytest.fixture
def pqa_settings() -> PQASettings:
    return PQASettings(
        llm="claude-3-5-sonnet-20240620",
        llm_config=local_llm_config,
        summary_llm="claude-3-5-sonnet-20240620",
        summary_llm_config=local_llm_config,
    )


@pytest.fixture
def mock_vector_db() -> RedisVectorStore:
    """Create a mock LCRedisVectorStore that avoids Redis connection."""
    mock_redis = MagicMock()
    mock_embeddings = MagicMock()
    mock_key_manager = MagicMock()

    with patch(
        "langchain_community.vectorstores.Redis.from_existing_index"
    ) as mock_from_existing:
        mock_from_existing.return_value = mock_redis
        vector_store = RedisVectorStore(
            redis_url="mock://localhost:6379",
            index_name="test_index",
            embedding=mock_embeddings,
        )
        vector_store._redis = mock_redis
        vector_store.key_manager = mock_key_manager
        return vector_store


@pytest.mark.skip(reason="Needs to be updated")
def test_parse_pdf_bytes_to_pages(docs_list_bytes: list[bytes]):
    # parsed_text: ParsedText = PQADocumentProcessor.parse_pdf_bytes_to_pages(
    #     docs_list_bytes[0]
    # )
    # assert len(parsed_text.content) > 0
    pass


@pytest.mark.skip(reason="Needs to be updated")
def test_chunk_pdf(docs_list_bytes: list[bytes], pqa_settings: PQASettings):
    text_chunks: list[Text] = []

    # text_chunks = PQADocumentProcessor.chunk_pdf(docs_list_bytes[0])
    # assert len(text_chunks) > 0
    # assert isinstance(text_chunks, list) and isinstance(text_chunks[0], str)
    pass


@pytest.mark.integration_test
@pytest.mark.asyncio
async def test_process_documents(
    docs_list_bytes: list[bytes], local_redis_vector_db: RedisVectorStore
):
    pqa_settings = PQASettings(
        llm="claude-3-5-sonnet-20240620",
        llm_config=local_llm_config,
        summary_llm="claude-3-5-sonnet-20240620",
        summary_llm_config=local_llm_config,
    )
    processor = PQAProcessor(pqa_settings, local_redis_vector_db)
    client = local_redis_vector_db.client
    keys = await processor.process_documents(docs_list_bytes)
    # get the first key
    key = keys[0][1][0]

    # Check that the first key exists in the vector store index
    assert client.exists(key) == 1
