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
    chunked_docs: list[list[str]],
):
    # Setup the vector store and clear any existing documents
    # TODO: Make the prefix dynamic

    redis_manager = RedisManager()
    logger.debug("Getting embedding service...")
    embedding_service = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    logger.debug("Generating embeddings...")
    doc_embeddings = []
    for paper in chunked_docs:
        logger.debug(f"Embedding document with shape: {len(paper)}")
        embeddings = embedding_service.embed_documents(paper)
        doc_embeddings.append(embeddings)

    logger.debug("Embeddings generated")
    logger.debug("Clearing vector store...")

    metadata = {
        "title": "test title 1",
        "authors": ["test authors 1", "test authors 2"],
        "doi": "test doi 1",
        "published_date": "2024-01-01",
        "created_at": "2024-01-01",
        "citation": "test citation 1",
        "journal": "test journal 1",
        "volume": "1",
        "issue": "1",
    }
    # Add the test documents to the vector store
    # We don't care about metadata so make dummy metadata
    logger.debug("Adding documents to vector store...")
    for doc, embeddings in zip(chunked_docs, doc_embeddings):
        metadata_list = [metadata] * len(doc)
        await local_redis_vector_db.add_texts(
            texts=doc,
            embeddings=embeddings,
            metadatas=metadata_list,
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
    query = "What is the Transformer architecture?"
    embedded_query = embedding_service.embed_query(query)
    logger.debug(f"Embedded query: {embedded_query}, dims: {len(embedded_query)}")
    results = await querier.query(embedded_query)
    print(results.answer)
    assert results.answer is not None and isinstance(results.answer, str)

    # Clean up the vector store
    redis_manager.clear_documents(prefix="docs:")
