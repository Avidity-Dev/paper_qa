import json
import pytest

from paperqa.settings import Settings as PQASettings

from src.process.metadata import pqa_build_mla, pqa_extract_publication_metadata
from src.models import DocumentChunk, DocumentMetadata
from src.process.processors import PQAProcessor


# TODO: Grab a known paper and confirm metadata is extracted correctly
@pytest.mark.asyncio
async def test_extract_publication_metadata(
    chunked_docs: list[list[str]], pqa_settings: PQASettings
):
    test_metadata_obj = DocumentMetadata()

    # just get first few chunks of each doc and process them
    metadata_keys = test_metadata_obj.keys
    llm = pqa_settings.get_llm()
    extracted_metadata = []

    for doc in chunked_docs:
        chunks = doc[0:2]
        metadata = await pqa_extract_publication_metadata(
            text=chunks, metadata_keys=metadata_keys, llm=llm
        )
        extracted_metadata.append(metadata)

    # pretty print the metadata
    print(json.dumps(extracted_metadata, indent=4))
    assert len(extracted_metadata) == len(chunked_docs)
    assert all(metadata is not None for metadata in extracted_metadata)


@pytest.mark.asyncio
async def test_build_mla(pqa_settings: PQASettings):
    llm = pqa_settings.get_llm()
    metadata = {
        "title": "Attention is All You Need",
        "author": "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
        "year": "2017",
        "journal": "Advances in Neural Information Processing Systems",
        "volume": "30",
    }
    mla = await pqa_build_mla(llm, **metadata)
    print(f"mla:\n{mla}")
