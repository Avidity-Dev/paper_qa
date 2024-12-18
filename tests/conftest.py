import os

from langchain_openai import OpenAIEmbeddings
from paperqa.types import Text
from paperqa.settings import Settings as PQASettings
import pymupdf
import pytest

from src.config.config import INDEX_SCHEMA_PATH, ConfigurationManager, AppConfig
from src.models import PQADocument
from src.process.processors import PQAProcessor
from src.storage.vector.stores import RedisVectorStore

TEST_PDF_PATH = "tests/data/docs/"

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
def app_settings() -> AppConfig:
    config = ConfigurationManager()
    config.init_app_config(environment="local")
    return config.app_config


@pytest.fixture
def processor() -> PQAProcessor:
    return PQAProcessor(
        settings=PQASettings(
            llm="claude-3-5-sonnet-20240620",
            llm_config=local_llm_config,
            summary_llm="claude-3-5-sonnet-20240620",
            summary_llm_config=local_llm_config,
        )
    )


# Hard coding index schema based off the model
# Should be moved to a config file
@pytest.fixture
def index_schema_dict() -> dict:
    return {
        "text": [
            {"name": "id", "no_stem": True},
            {"name": "title", "weight": 1.0},
            {"name": "chunk_text", "weight": 1.0},
            {"name": "doi", "no_stem": True},
        ],
        "tag": [{"name": "authors", "separator": "|"}],
        "numeric": [
            {"name": "published_date", "sortable": True},
            {"name": "created_at", "sortable": True},
        ],
        "vector": [
            {
                "name": "vector",
                "dims": 1536,
                "algorithm": "FLAT",
                "datatype": "FLOAT32",
                "distance_metric": "COSINE",
                "initial_cap": 6,
            }
        ],
    }


@pytest.fixture
def local_redis_vector_db(app_settings: AppConfig) -> RedisVectorStore:
    return RedisVectorStore(
        redis_url=app_settings.vector_db_url,
        index_name=app_settings.index_name,
        key_prefix=app_settings.index_prefix,
        schema=INDEX_SCHEMA_PATH,
        counter_key=app_settings.counter_key,
    )


@pytest.fixture
def docs_list_bytes() -> list[bytes]:

    # Get all the pdfs in the test directory
    pdfs = [f for f in os.listdir(TEST_PDF_PATH) if f.endswith(".pdf")]
    pdf_bytes = []

    # Read each pdf into a byte stream
    for pdf in pdfs:
        with open(os.path.join(TEST_PDF_PATH, pdf), "rb") as f:
            pdf_bytes.append(f.read())

    return pdf_bytes


@pytest.fixture
def chunked_docs(docs_list_bytes: list[bytes]) -> list[list[str]]:
    chunked_docs = []
    for doc in docs_list_bytes:
        chunked_docs.append(PQAProcessor.chunk_pdf(doc))
    return chunked_docs


@pytest.fixture
def doc_objects(chunked_docs: list[list[str]]) -> list[PQADocument]:
    return [
        PQADocument(id=i, text_chunks=chunks) for i, chunks in enumerate(chunked_docs)
    ]


@pytest.fixture
def config() -> ConfigurationManager:
    return ConfigurationManager()
