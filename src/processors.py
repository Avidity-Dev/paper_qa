"""
Classes for ingesting documents into the application. These are the main points of
entry for data into the application.

Custom readers extend the functionality of paper-qa, which is currently limited
to ingesting files from the local filesystem.
"""

import hashlib
import uuid
import warnings
import logging
from io import BytesIO
import sys
from typing import Any, List, Optional, Union
import os

from dotenv import load_dotenv
import paperqa as pqa
from paperqa.utils import ImpossibleParsingError
from paperqa.readers import chunk_pdf as pqa_chunk_pdf
from paperqa.readers import parse_pdf_to_pages as pqa_parse_pdf_to_pages
from paperqa.types import ParsedText, ParsedMetadata, ChunkMetadata, Text
import pymupdf
import tiktoken

from src.vectorstores.vectordb import VectorStore
from src.models import Document

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
    Handles document processing utilizing paperqa functionality along with custom
    logic.

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

        self._settings = pqa_settings
        self._vector_db = vector_db

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


    def process_documents(self, input: List[Union[os.PathLike, bytes, str, BytesIO]]):
        pass

    @staticmethod
    def chunk_pdf(
        input: Union[os.PathLike, bytes, str, BytesIO], chunk_chars: int, overlap: int
    ) -> list[Text]:
        """
        Chunk a PDF document into chunks of a given size. Uses paper-qa's chunk_pdf
        function for the actual chunking, but pagination is handled by either paperqa or
        custom logic depending on the input type.
        """

        if isinstance(input, (os.PathLike, str)):
            parsed_text = pqa_parse_pdf_to_pages(input)
        elif isinstance(input, (bytes, BytesIO)):
            parsed_text = PQADocumentProcessor.parse_pdf_bytes_to_pages(input)
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

        return pqa_chunk_pdf(parsed_text, chunk_chars, overlap)

    @staticmethod
    def parse_pdf_bytes_to_pages(
        input: Union[bytes, BytesIO], page_size_limit: int | None = None
    ) -> ParsedText:
        """
        Parse a PDF within a byte stream the `ParsedText` paperqa object.

        Nearly identical to paper-qa's parse_pdf_to_pages function, but handles bytes
        instead of a file path.

        Parameters
        ----------
        input : Union[bytes, BytesIO]
            Byte stream of a PDF document.

        Returns
        -------
        ParsedText
            Paper-qa object containing the parsed PDF document.
        """

        with pymupdf.open(stream=input, filetype="pdf") as pdf:
            pages: dict[int, str] = {}
            total_length = 0

            for i in range(pdf.page_count):
                try:
                    page = pdf.load_page(i)
                except pymupdf.mupdf.FzErrorFormat as exc:
                    raise ImpossibleParsingError(
                    f"Page loading via {pymupdf.__name__} failed on page {i} of"
                        f" {pdf.page_count} for the PDF bytes stream."
                    ) from exc
                text = page.get_text("text", sort=True)
                if page_size_limit and len(text) > page_size_limit:
                    raise ImpossibleParsingError(
                        f"The text in page {i} of {pdf.page_count} was {len(text)} chars"
                        f" long, which exceeds the {page_size_limit} char limit for the PDF"
                        f" bytes stream."
                    )
                pages[str(i + 1)] = text
                total_length += len(text)

        metadata = ParsedMetadata(
            parsing_libraries=[f"pymupdf {pymupdf.__version__}"],
            paperqa_version=pqa.__version__,
            total_parsed_text_length=total_length,
            parse_type="pdf_bytes",
        )

        return ParsedText(content=pages, metadata=metadata)


    def extract_document_metadata(
        self, text: ParsedText,
        doc: Optional[Document] = None,
        from_pages: Union[int, str, list[Union[int, str]]] = 1
    ) -> Document:

        if doc is None:
            doc = Document(
                id=str(uuid.uuid4()),
            )

        # TODO: Extract metadata from the document using LLM prompts first, as paperqa
        # does, then try additional methods for missing metadata.

        return










