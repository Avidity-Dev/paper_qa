import json
import pytest
import os
from typing import List
from paperqa.settings import Settings as PQASettings

from src.process.metadata import pqa_build_mla, pqa_extract_publication_metadata, enrich_metadata_list, find_similar_papers
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
@pytest.mark.integration_test
async def test_extract_and_enrich_metadata(
    chunked_docs: List[List[str]], 
    pqa_settings: PQASettings
):
    """
    Test both LLM metadata extraction and API enrichment using real documents
    and real API calls to Semantic Scholar and Crossref.
    """
    # Initialize test objects
    test_metadata_obj = DocumentMetadata()
    metadata_keys = test_metadata_obj.keys
    llm = pqa_settings.get_llm()
    
    try:
        # Directly call enrich_metadata_list which now handles extraction and enrichment
        enriched_metadata = await enrich_metadata_list(
            metadata_list=[{}] * len(chunked_docs),  # Start with empty metadata for each document
            llm=llm,
            text_chunks=[doc[0:2] for doc in chunked_docs],  # Use first 2 chunks of each document
            metadata_keys=metadata_keys,
            mailto="angel.murillo@aviditybio.com"
        )
        
        print("\nEnriched metadata using LLM and APIs:")
        print(json.dumps(enriched_metadata, indent=4))
        
        # Verify enrichment structure and content
        assert len(enriched_metadata) == len(chunked_docs), \
            "Should have one metadata entry per document"
            
        for entry in enriched_metadata:
            # Check required fields
            assert isinstance(entry, dict), "Each entry should be a dictionary"
            required_fields = {
                "title", "authors", "doi", "published_date", 
                "journal", "volume", "issue", "citation"
            }
            
            # Verify fields exist (can be None)
            assert all(field in entry for field in required_fields), \
                f"All required fields {required_fields} should be present"
            
            # Verify field types
            assert isinstance(entry.get("title"), (str, type(None))), "Title should be a string or None"
            assert isinstance(entry.get("authors"), (list, type(None))), "Authors should be a list or None"
            assert isinstance(entry.get("doi"), (str, type(None))), "DOI should be a string or None"
            assert isinstance(entry.get("published_date"), (str, type(None))), "Published date should be a string or None"
            assert isinstance(entry.get("journal"), (str, type(None))), "Journal should be a string or None"
            assert isinstance(entry.get("volume"), (str, type(None))), "Volume should be a string or None"
            assert isinstance(entry.get("issue"), (str, type(None))), "Issue should be a string or None"
            
            # At least some fields should be populated after enrichment
            populated_fields = sum(1 for v in entry.values() if v is not None)
            assert populated_fields > 0, \
                "At least some fields should be populated after LLM extraction and API enrichment"
            
    except Exception as e:
        print(f"\nError during metadata extraction and enrichment: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_similar_papers_for_extracted(
    chunked_docs: List[List[str]], 
    pqa_settings: PQASettings
):
    """
    Test finding similar papers using metadata from actual extracted documents,
    making real API calls to find similar papers.
    """
    # First get metadata from a real document
    test_metadata_obj = DocumentMetadata()
    metadata_keys = test_metadata_obj.keys
    llm = pqa_settings.get_llm()
    
    # Get metadata from first document
    first_doc_chunks = chunked_docs[0][0:2]
    extracted_metadata = await pqa_extract_publication_metadata(
        text=first_doc_chunks,
        metadata_keys=metadata_keys,
        llm=llm
    )
    
    # Debug print to check the structure of extracted_metadata
    print("\nExtracted Metadata:", extracted_metadata)

    # Use the extracted metadata directly
    test_metadata = extracted_metadata  # No need to index

    try:
        # Find similar papers using real API calls
        similar_papers = await find_similar_papers(
            test_metadata,
            max_results=5,
            mailto="angel.murillo@aviditybio.com"
        )
        
        print("\nSimilar papers found:")
        for i, paper in enumerate(similar_papers, 1):
            print(f"\nPaper {i}:")
            print(f"Title: {paper['title']}")
            print(f"Authors: {', '.join(paper['authors'])}")
            print(f"Journal: {paper['journal']}")
            print(f"Published Date: {paper['published_date']}")
            print(f"DOI: {paper['doi']}")
            if paper.get('citation_count'):
                print(f"Citations: {paper['citation_count']}")
        
        # Verify similar papers structure and content
        assert isinstance(similar_papers, list), "Results should be a list"
        assert len(similar_papers) > 0, "Should return at least one similar paper"
        
        # Check each similar paper
        for paper in similar_papers:
            assert isinstance(paper, dict), "Each paper should be a dictionary"
            
            # Verify required fields
            required_fields = {
                'title', 'authors', 'doi', 'published_date',
                'journal', 'volume', 'issue', 'citation'
            }
            assert all(field in paper for field in required_fields), \
                f"All required fields {required_fields} should be present"
            
            # Verify field types
            assert isinstance(paper['title'], str), "Title should be a string"
            assert isinstance(paper['authors'], list), "Authors should be a list"
            assert isinstance(paper['doi'], (str, type(None))), "DOI should be a string or None"
            assert isinstance(paper['published_date'], (str, type(None))), \
                "Published date should be a string or None"
            assert isinstance(paper['journal'], (str, type(None))), \
                "Journal should be a string or None"
            
            # Verify it's not the same as original paper
            assert paper['title'] != test_metadata['title'], \
                "Similar paper should be different from original"
            
    except ValueError as e:
        print(f"\nAPI Error: {str(e)}")
        raise

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling for edge cases"""
    # Test with empty metadata
    empty_metadata = {
        "title": None,
        "authors": [],
        "doi": None,
        "published_date": None,
        "citation": None,
        "journal": None,
        "volume": None,
        "issue": None
    }
    
    # Should handle empty metadata gracefully
    enriched = await enrich_metadata_list(
        [empty_metadata],
        mailto="your@email.com"  # Replace with your email
    )
    assert len(enriched) == 1, "Should return same number of entries"
    
    # Test with invalid DOI
    invalid_doi_metadata = {
        "title": "Test Paper",
        "authors": ["Test Author"],
        "doi": "invalid-doi",
        "published_date": None,
        "citation": None,
        "journal": None,
        "volume": None,
        "issue": None
    }
    
    # Should handle invalid DOI gracefully and try title-based search
    enriched = await enrich_metadata_list(
        [invalid_doi_metadata],
        mailto="your@email.com"  # Replace with your email
    )
    assert len(enriched) == 1, "Should return same number of entries"