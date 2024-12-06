import os

from paperqa.types import Text
import pymupdf
import pytest

from src.config.config import ConfigurationManager
from src.models import PQADocument
from src.processors import PQADocumentProcessor
from src.vectorstores.vectordb import LCRedisVectorStore

TEST_PDF_PATH = "tests/data/docs/"


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
def local_redis_vector_db(index_schema_dict: dict) -> LCRedisVectorStore:
    return LCRedisVectorStore(
        redis_url="redis://localhost:6379",
        index_name="idx:docs_vss",
        key_prefix="docs",
        index_schema=index_schema_dict,
        counter_key="docs_ctr",
        key_padding=4,
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
def chunked_text_objects(docs_list_bytes: list[bytes]) -> list[list[Text]]:
    chunked_docs = []
    for doc in docs_list_bytes:
        chunked_docs.append(PQADocumentProcessor.chunk_pdf(doc))
    return chunked_docs


@pytest.fixture
def chunked_text_strings(chunked_text_objects: list[list[Text]]) -> list[list[str]]:
    return [[text.text for text in chunk] for chunk in chunked_text_objects]


@pytest.fixture
def doc_embeddings(chunked_text_strings: list[list[str]]) -> list[list[float]]:
    return [
        local_redis_vector_db.embed_documents(chunk) for chunk in chunked_text_strings
    ]


@pytest.fixture
def doc_objects(chunked_text_objects: list[list[Text]]) -> list[PQADocument]:
    return [
        PQADocument(id=i, text_chunks=chunks)
        for i, chunks in enumerate(chunked_text_objects)
    ]


@pytest.fixture
def config() -> ConfigurationManager:
    return ConfigurationManager()
