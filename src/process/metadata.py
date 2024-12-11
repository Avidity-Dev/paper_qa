"""
Metadata processing for documents.
"""

import json
import re
import os
import aiohttp
from typing import Any

from paperqa.docs import DocMetadataClient
from paperqa.llms import LiteLLMModel
from src.process.rate_limiter import rate_limiter

from typing import List, Optional, Dict
from urllib.parse import quote

PUBLICATION_METADATA_EXTRACTION_PROMPT = (
    "Please attempt to extract the following metadata from the provided text input, "
    "which is a text chunk from a research paper or other publication, and return it as a "
    "JSON object:\n"
    "{metadata_keys}\n\n"
    "If any field can not be found, return it as null. Be sure to preserve the ordering"
    "of authors. Use the provided metadata keys as the keys in the JSON object. Do "
    "not provide an introduction or any other text within your response besides the "
    "JSON object.\n"
    "Text input:{text}"
)


CITATION_PROMPT = (
    "Please build a citation from the provided metadata.  The citation should be in "
    "MLA format. I've provided the MLA structure and an example citation for your "
    "reference. Do not use utilize any fields that are not provided within the "
    "metadata. Do not provide an introduction or any other text within your response "
    "besides the citation.\n"
    "Provided metadata:{metadata}\n\n"
    "MLA Reference structure:\n"
    "Article Author’s Last Name, First Name. “Title of Article.” Title of Journal, vol. "
    "number, issue no., date published, page range.\n\n"
    "Example citation:\n"
    "Brundan, Katy. “What We Can Learn From the Philologist in Fiction.” Criticism, "
    "vol. 61, no. 3, summer 2019, pp.285-310"
)

def _metadata_keys_to_json(metadata_keys: list[str]) -> str:
    """Converts a list of metadata keys to a JSON string with empty values"""
    return json.dumps({key: None for key in metadata_keys})

def unpack_metadata(obj: Any, metadata: dict) -> None:
    """Unpacks metadata into an object in place"""
    for key, value in metadata.items():
        setattr(obj, key, value)

async def pqa_extract_publication_metadata(
    text: str, metadata_keys: list[str], llm: LiteLLMModel
) -> dict:
    """Extracts metadata from a text input using a LiteLLMModel wrapper."""
    metadata_keys_json = _metadata_keys_to_json(metadata_keys)
    response = await llm.run_prompt(
        PUBLICATION_METADATA_EXTRACTION_PROMPT,
        data={"metadata_keys": metadata_keys_json, "text": text},
        system_prompt=None,
    )

    clean_text = response.text.split("{", 1)[-1].split("}", 1)[0]
    clean_text = "{" + clean_text + "}"

    try:
        return json.loads(clean_text)
    except json.JSONDecodeError:
        raise ValueError("Failed to parse metadata from response")

# TO DO: Add the MLA citation function from enriched metadata
async def pqa_build_mla(
    llm: LiteLLMModel,
    **kwargs,
) -> str:
    """Builds an MLA citation from the provided metadata."""
    metadata = {k: v for k, v in kwargs.items()}
    response = await llm.run_prompt(
        CITATION_PROMPT,
        data={"metadata": metadata},
        system_prompt=None,
    )
    return response

async def pqa_add_pages_mla(llm: LiteLLMModel, **kwargs) -> str:
    """Adds page numbers to an MLA citation."""
    pass  

# Added configuration for Semantic Scholar API with rate limiter
SEMANTIC_SCHOLAR_API_KEY = os.getenv('SEMANTIC_SCHOLAR_API_KEY')
if not SEMANTIC_SCHOLAR_API_KEY:
    raise ValueError("SEMANTIC_SCHOLAR_API_KEY environment variable is required")

async def enrich_metadata_with_semantic_scholar(metadata: Dict) -> Optional[Dict]:
    """
    Enriches a single metadata entry using rate-limited Semantic Scholar API.
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper"
    fields = "title,authors,year,venue,citationCount,publicationDate,externalIds"

    # Clean up DOI if present
    doi = metadata.get('doi')
    if doi and doi.startswith('https://doi.org/'):
        doi = doi.replace('https://doi.org/', '')

    # Try DOI first if available
    if doi:
        endpoint = f"{base_url}/DOI:{quote(doi)}"
        params = {'fields': fields}
    # Fall back to title search if no DOI
    elif metadata.get('title'):
        endpoint = f"{base_url}/search"
        params = {
            'query': metadata['title'],
            'fields': fields,
            'limit': 1
        }
    else:
        return None

    # Wait for rate limiter before making request
    await rate_limiter.acquire('semantic_scholar')

    headers = {
        'x-api-key': SEMANTIC_SCHOLAR_API_KEY
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint, params=params, headers=headers) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                paper_data = data['data'][0] if 'search' in endpoint else data
                
                if not paper_data:
                    return None

                enriched = metadata.copy()
                
                # Only update fields that were null in original metadata
                if not enriched.get('doi'):
                    enriched['doi'] = paper_data.get('externalIds', {}).get('DOI')
                if not enriched.get('published_date'):
                    enriched['published_date'] = paper_data.get('publicationDate')
                if not enriched.get('journal'):
                    enriched['journal'] = paper_data.get('venue')
                
                # Add some semantic scholar specific fields as extra info
                enriched['citation_count'] = paper_data.get('citationCount')
                enriched['year'] = paper_data.get('year')
                
                return enriched
                
        except Exception:
            return None


async def enrich_metadata_with_crossref(metadata: Dict, mailto: Optional[str] = None) -> Optional[Dict]:
    """
    Enriches a single metadata entry using rate-limited Crossref API.
    """
    mailto = mailto or os.getenv('CROSSREF_MAILTO')
    if not mailto:
        raise ValueError("CROSSREF_MAILTO environment variable or mailto parameter required")

    base_url = "https://api.crossref.org/works"

    # Clean up DOI if present
    doi = metadata.get('doi')
    if doi and doi.startswith('https://doi.org/'):
        doi = doi.replace('https://doi.org/', '')

    # Wait for rate limiter before making request
    await rate_limiter.acquire('crossref')

    # Try DOI first if available
    if doi:
        endpoint = f"{base_url}/{quote(doi)}"
        params = {'mailto': mailto}
    # Fall back to title search if no DOI
    elif metadata.get('title'):
        endpoint = base_url
        params = {
            'query': metadata['title'],
            'mailto': mailto,
            'rows': '1',
            'filter': 'type:journal-article'
        }
    else:
        return None

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(endpoint, params=params) as response:
                if response.status != 200:
                    return None
                
                data = await response.json()
                item = data['message'] if 'DOI' in endpoint else data['message']['items'][0]
                
                enriched = metadata.copy()
                
                # Only update fields that were null in original metadata
                if not enriched.get('doi'):
                    enriched['doi'] = item.get('DOI')
                if not enriched.get('journal'):
                    enriched['journal'] = item.get('container-title', [None])[0]
                if not enriched.get('volume'):
                    enriched['volume'] = item.get('volume')
                if not enriched.get('issue'):
                    enriched['issue'] = item.get('issue')
                if not enriched.get('published_date'):
                    date_parts = item.get('published-print', {}).get('date-parts', [[]])[0]
                    if len(date_parts) >= 3:
                        enriched['published_date'] = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                
                return enriched
                
        except Exception:
            return None

async def enrich_metadata_list(
    metadata_list: List[Dict],
    mailto: Optional[str] = None
) -> List[Dict]:
    """
    Enriches a list of metadata entries using both Semantic Scholar and Crossref.
    Maintains the original metadata structure while adding missing information.
    """
    enriched_list = []
    
    for metadata in metadata_list:
        # Try Semantic Scholar first
        enriched = await enrich_metadata_with_semantic_scholar(metadata)
        
        # If no results or missing key fields, try Crossref
        if not enriched or not all([
            enriched.get('doi'),
            enriched.get('published_date'),
            enriched.get('journal')
        ]):
            crossref_enriched = await enrich_metadata_with_crossref(
                enriched or metadata,
                mailto=mailto
            )
            if crossref_enriched:
                enriched = crossref_enriched
        
        # If still no enrichment, use original metadata
        enriched = enriched or metadata
        enriched_list.append(enriched)
    
    return enriched_list

async def find_similar_papers(
    metadata: Dict,
    max_results: int = 5,
    mailto: Optional[str] = None
) -> List[Dict]:
    """
    Finds similar papers using rate-limited API calls.
    """
    similar_papers = []
    
    if not metadata.get('title'):
        return similar_papers

    # Try Semantic Scholar first
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    fields = "title,authors,year,venue,citationCount,publicationDate,externalIds"
    
    # Wait for rate limiter before making request
    await rate_limiter.acquire('semantic_scholar')

    headers = {
        'x-api-key': SEMANTIC_SCHOLAR_API_KEY
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            params = {
                'query': metadata['title'],
                'fields': fields,
                'limit': max_results
            }
            
            async with session.get(base_url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    for item in data.get('data', []):
                        if item.get('title') != metadata.get('title'):
                            similar = {
                                'title': item.get('title'),
                                'authors': [author.get('name', '') for author in item.get('authors', [])],
                                'doi': item.get('externalIds', {}).get('DOI'),
                                'published_date': item.get('publicationDate'),
                                'citation': None,
                                'journal': item.get('venue'),
                                'volume': None,
                                'issue': None
                            }
                            similar_papers.append(similar)
        except Exception:
            pass
    
    # If we need more results, try Crossref
    if len(similar_papers) < max_results:
        try:
            # Wait for rate limiter before making request
            await rate_limiter.acquire('crossref')

            crossref_base = "https://api.crossref.org/works"
            remaining = max_results - len(similar_papers)
            
            params = {
                'query': metadata['title'],
                'mailto': mailto,
                'rows': str(remaining),
                'filter': 'type:journal-article'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(crossref_base, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for item in data['message']['items']:
                            if item.get('title', [None])[0] != metadata.get('title'):
                                date_parts = item.get('published-print', {}).get('date-parts', [[]])[0]
                                published_date = None
                                if len(date_parts) >= 3:
                                    published_date = f"{date_parts[0]}-{date_parts[1]:02d}-{date_parts[2]:02d}"
                                
                                similar = {
                                    'title': item.get('title', [None])[0],
                                    'authors': [
                                        f"{author.get('given', '')} {author.get('family', '')}".strip()
                                        for author in item.get('author', [])
                                    ],
                                    'doi': item.get('DOI'),
                                    'published_date': published_date,
                                    'citation': None,
                                    'journal': item.get('container-title', [None])[0],
                                    'volume': item.get('volume'),
                                    'issue': item.get('issue')
                                }
                                similar_papers.append(similar)
        except Exception:
            pass
    
    return similar_papers[:max_results]