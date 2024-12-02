"""
Classes for ingesting documents into the application. These are the main points of
entry for data into the application.

Custom readers extend the functionality of paper-qa, which is currently limited
to ingesting files from the local filesystem.
"""

import hashlib
import warnings
import logging
from io import BytesIO
import sys
from typing import Any, List, Optional, Union
import os

from dotenv import load_dotenv
import paperqa as pqa
from paperqa.readers import chunk_pdf as pqa_chunk_pdf
from paperqa.readers import parse_pdf_to_pages as pqa_parse_pdf_to_pages
from paperqa.types import ParsedText, ParsedMetadata, ChunkMetadata, Text
import tiktoken

from src.adapters.vectordb import VectorStore

# Configure logging to suppress paper-qa's API-related messages
logging.getLogger("paper_qa").setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, stream=sys.stdout
)  # Changed from paperqa to paper_qa

load_dotenv()

print(os.getenv("ANTHROPIC_API_KEY"))

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


class ProcessingRoute:
    """Interface for processing routes strategies."""

    def process_document(self, input, **kwargs):
        """Process a document."""
        pass

    def process_documents(self, input, **kwargs):
        """Process a list of documents."""
        pass

    def chunk_pdf(self, text: Any, **kwargs) -> list[str]:
        """Chunk a PDF document into chunks of a given size."""
        pass

    def extract_metadata(self, text: Any, **kwargs) -> Any:
        """Extract metadata from text."""
        pass


class PQAProcessingRoute(ProcessingRoute):
    """Process documents using paper-qa."""

    def __init__(self, pqa_settings: Optional[pqa.Settings] = None):
        self._pqa_settings = pqa_settings

    def process_document( self, input: os.PathLike):
        doc =

    def process_documents(self, input: List[Union[os.PathLike, bytes, str, BytesIO]]):
        pass

    def chunk_pdf(
        self, input: Union[os.PathLike, bytes, str, BytesIO], chunk_chars: int, overlap: int
    ) -> list[Text]:

        if isinstance(input, os.PathLike):
            parsed_text: ParsedText = pqa_parse_pdf_to_pages(input)
        elif isinstance(input, (bytes, str, BytesIO)):

        return pqa_chunk_pdf(parsed_text, chunk_chars, overlap)


    def extract_metadata(self, text: ParsedText, **kwargs) -> ParsedMetadata:
        pass


class DocumentProcessor:
    """
    Base class for document processors.
    """

    def process_document(self, doc: Union[os.PathLike, bytes, str, BytesIO]):
        """
        Main processing method for document ingestion into the application.
        """
        pass


class PQADocumentProcessor(DocumentProcessor):
    """
    Handles document processing, utilizing core paper-qa functionality which itself
    is a wrapper around LiteLLM, among other libraries.).

    Attributes:
    -----------
    pqa_settings: pqa.Settings
        Paper-qa settings object used to configure processing behavior.
    vector_db: VectorStore
        Vector database to store document embeddings
    """

    def __init__(
        self,
        pqa_settings: pqa.Settings,
        vector_db: VectorStore,
    ):
        """
        Initialize the document processor.

        Parameters:
        -----------
        pqa_settings: pqa.Settings
            Paper-qa settings object used to configure processing behavior.
        vector_db: VectorStore
            VectorStore object to store document embeddings.
        """
        # Suppress warnings about missing APIs
        warnings.filterwarnings("ignore", message=".*API.*")
        warnings.filterwarnings("ignore", message=".*Provider.*")

        self.settings = pqa.Settings(
            llm="claude-3-5-sonnet-20240620",
            summary_llm="claude-3-5-sonnet-20240620",
            llm_config=local_llm_config,
            summary_llm_config=local_llm_config,
        )
        # TODO: Check vector store connection

    def process_document(self, doc: Union[os.PathLike, bytes, str, BytesIO]):
        """
        Process a document and store basic metadata.

        Parameters:
        -----------
        doc: Union[os.PathLike, bytes, str, BytesIO]
            Document to process. Uses paper-qa's read_doc function if a file path is
            provided, otherwise custom processing is done.
        """

       pass
