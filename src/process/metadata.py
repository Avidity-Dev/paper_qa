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
    """Extracts metadata from a text input using a LiteLLMModel wrapper from the paperqa
    library.
    """
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


async def pqa_build_mla(
    llm: LiteLLMModel,
    **kwargs,
) -> str:
    """Builds an MLA citation from the provided metadata, using a LiteLLMModel wrapper
    from the paperqa library.

    Parameters:
    -----------
    **kwargs: dict
        Keyword arguments to pass to the citation builder.
    """
    # collect kwargs into a dict
    metadata = {k: v for k, v in kwargs.items()}

    # build the citation
    response = await llm.run_prompt(
        CITATION_PROMPT,
        data={"metadata": metadata},
        system_prompt=None,
    )
    return response


async def pqa_add_pages_mla(llm: LiteLLMModel, **kwargs) -> str:
    """Adds page numbers to an MLA citation."""
    pass

async def fetch_similar_papers(
    query: str,
    max_results: int = 5,
    mailto: Optional[str] = None
) -> List[Dict]:
    """
    Fetches similar papers using the Crossref API based on a query.
    
    Parameters:
    -----------
    query : str
        Search query to find similar papers
    max_results : int
        Maximum number of results to return
    mailto : str, optional
        Email for polite API usage. Defaults to CROSSREF_MAILTO env variable
        
    Returns:
    --------
    List[Dict]
        List of dictionaries containing paper information
    """
    # Validate query
    if not query:
        raise ValueError("Query cannot be empty")
    # Use environment variable if mailto not provided
    mailto = mailto or os.getenv('CROSSREF_MAILTO')
    if not mailto:
        raise ValueError("CROSSREF_MAILTO environment variable or mailto parameter required")

    # Construct the URL with parameters
    base_url = "https://api.crossref.org/works"
    params = {
        'query': query,
        'mailto': mailto,
        'rows': str(max_results),
        'filter': 'type:journal-article',
        'sort': 'relevance',
        'select': 'DOI,title,author,published-print,container-title'
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            # Build URL with parameters
            param_strings = [f"{k}={quote(str(v))}" for k, v in params.items()]
            url = f"{base_url}?{'&'.join(param_strings)}"
            
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"API request failed with status {response.status}")
                
                data = await response.json()
                
                similar_papers = []
                for item in data['message']['items']:
                    # Extract author information
                    authors = []
                    if 'author' in item:
                        for author in item['author']:
                            author_name = f"{author.get('given', '')} {author.get('family', '')}".strip()
                            if author_name:
                                authors.append(author_name)
                    
                    paper = {
                        'title': item.get('title', [None])[0],
                        'authors': authors,
                        'doi': item.get('DOI'),
                        'published_year': item.get('published-print', {}).get('date-parts', [[None]])[0][0],
                        'journal': item.get('container-title', [None])[0]
                    }
                    similar_papers.append(paper)
                
                return similar_papers
                
        except Exception as e:
            raise ValueError(f"Error fetching similar papers: {str(e)}")

async def fetch_semantic_scholar_papers(
    query: str,
    max_results: int = 5
) -> List[Dict]:
    """
    Fetches papers from Semantic Scholar API based on a query.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    
    # Modify the fields string to match Semantic Scholar's expected format
    fields = "title,authors,year,venue,citationCount,externalIds,publicationDate"
    
    params = {
        'query': query,
        'limit': max_results,
        'fields': fields
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(base_url, params=params) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"API request failed with status {response.status}. Error: {error_text}")
                
                data = await response.json()
                papers = []
                
                for item in data.get('data', []):
                    # Get DOI from externalIds if available
                    doi = item.get('externalIds', {}).get('DOI')
                    
                    paper = {
                        'title': item.get('title'),
                        'authors': [author.get('name', '') for author in item.get('authors', [])],
                        'doi': doi,
                        'published_year': item.get('year'),
                        'journal': item.get('venue'),
                        'citation_count': item.get('citationCount')
                    }
                    papers.append(paper)
                
                return papers
                
        except Exception as e:
            raise ValueError(f"Error fetching papers: {str(e)}")