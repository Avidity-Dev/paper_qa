import os
from logging import getLogger
from typing import Optional

from langchain_openai import OpenAIEmbeddings
import pytest

from paperqa.settings import AnswerSettings, Settings
from pydantic import BaseModel


from manage import RedisManager
from src.models import DocumentMetadata
from src.query.queriers import PQAQuerier
from src.storage.vector.stores import RedisVectorStore
from src.storage.vector.converters import LCVectorStorePipeline, TextStorageType

# TODO: Turn this into a fixture and refactor all tests that use this.
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

logger = getLogger(__name__)


@pytest.mark.asyncio
@pytest.mark.integration_test
async def test_pqa_querier(
    local_redis_vector_db: RedisVectorStore,
):
    # Setup the vector store and clear any existing documents
    # TODO: Make the prefix dynamic

    redis_manager = RedisManager()
    logger.debug("Getting embedding service...")
    embedding_service = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_pipeline = LCVectorStorePipeline(
        vector_store=local_redis_vector_db,
        target_type=TextStorageType.PAPERQA,
    )

    answer_settings: AnswerSettings = AnswerSettings(
        evidence_retrieval=False,
        answer_max_sources=10,
        get_evidence_if_no_contexts=False,
    )

    pqa_settings: Settings = Settings(
        llm="claude-3-5-sonnet-20240620",
        llm_config=local_llm_config,
        summary_llm="claude-3-5-sonnet-20240620",
        summary_llm_config=local_llm_config,
        answer=answer_settings,
    )

    querier = PQAQuerier(vector_pipeline=vector_pipeline, pqa_settings=pqa_settings)
    queries = [
        "What is the Transformer architecture?",
        "How can RNA be delivered using antibodies?",
    ]
    for query in queries:
        embedded_query = embedding_service.embed_query(query)
        print(f"Embedded query: {embedded_query}, dims: {len(embedded_query)}")
        results = await querier.query(query)
        print(results.answer)
    assert results.answer is not None and isinstance(results.answer, str)
