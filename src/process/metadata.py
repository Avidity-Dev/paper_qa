"""
Metadata processing for documents.
"""

import json
import re
from typing import Any

from paperqa.docs import DocMetadataClient
from paperqa.llms import LiteLLMModel

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
