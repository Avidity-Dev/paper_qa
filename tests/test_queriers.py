import os
import pytest

from paperqa.settings import Settings as PQASettings

from src.queriers import PQAQuerier
from src.vectorstores.vectordb import PQARedisVectorStore

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


@pytest.mark.asyncio
async def test_pqa_querier():
    pqa_vector_db = PQARedisVectorStore(redis_url="redis://localhost:6379")
    pqa_settings = PQASettings(
        llm="claude-3-5-sonnet-20240620",
        llm_config=local_llm_config,
        summary_llm="claude-3-5-sonnet-20240620",
        summary_llm_config=local_llm_config,
    )
    querier = PQAQuerier(pqa_vector_db, pqa_settings)
    query = "What is the Transformer architecture?"
    results = await querier.query(query)
    for text, score in zip(results[0], results[1]):
        print(text.text)
        print(score)
        print(text.embedding)
