import json
import pytest
import os

from paperqa.settings import Settings as PQASettings

from src.process.metadata import pqa_build_mla, pqa_extract_publication_metadata, fetch_similar_papers
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


# TODO: Using a known paper, confirm that the MLA citation has expected fields
@pytest.mark.skip(reason="Not implemented.")
@pytest.mark.asyncio
async def test_build_mla(pqa_settings: PQASettings):
    llm = pqa_settings.get_llm()
    mla = await pqa_build_mla(llm)
    print(f"mla:\n{mla}")


@pytest.mark.asyncio
async def test_fetch_similar_papers():
    """Test the Crossref API paper fetching functionality"""
    # Test query
    test_query = "Deep learning in genomics and biomedicine"
    
    # Ensure CROSSREF_MAILTO is in .env
    if not os.getenv('CROSSREF_MAILTO'):
        pytest.skip("CROSSREF_MAILTO environment variable not set")
    
    # Test with small number of results for faster testing such as 3 for now
    results = await fetch_similar_papers(
        query=test_query,
        max_results=3
    )
    # Print results
    print("\nSimilar papers found:")
    for i, paper in enumerate(results, 1):
        print(f"\nPaper {i}:")
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Journal: {paper['journal']}")
        print(f"Year: {paper['published_year']}")
        print(f"DOI: {paper['doi']}")
        
    # Basic assertions to verify the structure and content
    assert isinstance(results, list), "Results should be a list"
    assert len(results) > 0, "Should return at least one result"
    
    # Check structure of first result
    first_paper = results[0]
    assert isinstance(first_paper, dict), "Each result should be a dictionary"
    
    # Verify required fields are present
    required_fields = {'title', 'authors', 'doi', 'published_year', 'journal'}
    assert all(field in first_paper for field in required_fields), \
        f"All required fields {required_fields} should be present"
    
    # Verify field types
    assert isinstance(first_paper['title'], str), "Title should be a string"
    assert isinstance(first_paper['authors'], list), "Authors should be a list"
    assert isinstance(first_paper['doi'], str), "DOI should be a string"
    assert isinstance(first_paper['journal'], str), "Journal should be a string"
    
    # Test error handling with invalid query
    with pytest.raises(ValueError):
        await fetch_similar_papers(
            query="",  # Empty query should raise error
            max_results=1
        )