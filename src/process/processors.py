"""
Classes for ingesting documents into the application. These are the main points of
entry for data into the application.

Custom readers extend the functionality of paper-qa, which is currently limited
to ingesting files from the local filesystem.
"""

from datetime import datetime
import hashlib
import uuid
import warnings
import logging
from io import BytesIO
import sys
from typing import Any, Dict, List, Optional, Union
import os

from dotenv import load_dotenv
import numpy as np
import paperqa as pqa
from paperqa.docs import Doc as PQADoc
from paperqa.docs import Docs as PQADocs
from paperqa.llms import LiteLLMModel, EmbeddingModel
from paperqa.utils import ImpossibleParsingError
from paperqa.readers import chunk_pdf as pqa_chunk_pdf
from paperqa.readers import parse_pdf_to_pages as pqa_parse_pdf_to_pages
from paperqa.types import Text as PQAText
from paperqa.types import ParsedText, ParsedMetadata, ChunkMetadata
import pymupdf
import tiktoken

from src.storage.vector.stores import RedisVectorStore
from src.models import DocumentMetadata, PQADocument
from src.process.metadata import (
    pqa_build_mla,
    extract_and_enrich_metadata,
)
from src.storage.vector.converters import (
    LCVectorStorePipeline,
    BaseVectorStorePipeline,
    TextStorageType,
    LCVectorStore,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

load_dotenv()

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


# TODO: Add in logic for storing documents in cloud object storage
# TODO: Add in logic for parsing metadata from documents
class PQADocumentProcessor:
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
        vector_db: RedisVectorStore,
        metadata_keys: list[str] = DocumentMetadata.keys(),
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
        self._vector_db: RedisVectorStore = vector_db
        self._metadata_keys = metadata_keys

    @property
    def vector_db(self) -> RedisVectorStore:
        """Get the underlying vector store for operations that do not require
        document type conversion.
        """
        return self._vector_db

    async def chunk_and_embed(
        self, input: Union[os.PathLike, bytes, str, BytesIO]
    ) -> tuple[list[str], list[float]]:
        """Process a single document and return a tuple of chunks and embeddings."""
        try:
            chunks = self.chunk_pdf(input)
            # Utilize paper-qa to retrieve general embeddings
            embedding_model = self._settings.get_embedding_model()
            embeddings = await embedding_model.embed_documents(chunks)

        except Exception as e:
            logger.error(f"Error building document: {str(e)}", exc_info=True)
            raise

        return chunks, embeddings

    async def process_documents(
        self, input: List[Union[os.PathLike, bytes, str, BytesIO]]
    ) -> list[str]:
        """
        Driver function for processing a list of documents and storing basic metadata.

        Parameters:
        -----------
        input: List[Union[os.PathLike, bytes, str, BytesIO]]
            List of documents to process.
        """
        processed_docs: list[tuple[dict, list[str]]] = []

        for _doc in input:
            try:
                chunks, embeddings = await self.chunk_and_embed(_doc)

                # Take the first two chunks to extract metadata
                chunk = " ".join(chunks[0:2])
                metadata = await extract_and_enrich_metadata(
                    llm=self.llm,
                    text_chunks=[chunk],
                    metadata_keys=self._metadata_keys,
                )

                # Add MLA citation to metadata if not already present
                if not metadata[0].get("citation"):
                    metadata[0]["citation"] = await pqa_build_mla(
                        llm=self.llm, **metadata[0]
                    )
                metadata = [metadata[0]] * len(chunks)

                keys = await self.vector_db.add_texts(
                    texts=chunks, metadatas=metadata, embeddings=embeddings
                )
                processed_docs.append((metadata, keys))
                logger.info(f"Processed document:\n{_doc}\nwith {len(keys)} chunks.")
            except Exception as e:
                logger.error(f"Error processing document {_doc}: {str(e)}\nSkipping...")
                raise

        return keys

    @staticmethod
    def chunk_pdf(
        input: Union[os.PathLike, bytes, str, BytesIO],
        chunk_chars: int = 1500,
        overlap: int = 100,
    ) -> list[str]:
        """
        Chunk a PDF document into chunks of a given size. Uses paper-qa's chunk_pdf
        function for the actual chunking, but pagination is handled by either paperqa or
        custom logic depending on the input type.
        """
        # dummy doc object
        doc = PQADoc(docname="", citation="", dockey="")
        try:
            if isinstance(input, (os.PathLike, str)):
                parsed_text = pqa_parse_pdf_to_pages(input)
            elif isinstance(input, (bytes, BytesIO)):
                parsed_text = PQADocumentProcessor.parse_pdf_bytes_to_pages(input)
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}", exc_info=True)
            raise

        # We won't use the doc argument, but it's required by paper-qa's chunk_pdf
        chunks: list[PQAText] = pqa_chunk_pdf(
            parsed_text=parsed_text, doc=doc, chunk_chars=chunk_chars, overlap=overlap
        )

        return [chunk.text for chunk in chunks]

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


class PQAPineconeDocs(PQADocs):
    """Extension of paperqa's Docs class that uses Pinecone for vector storage."""

    def __init__(self, index_name: str, api_key: str, environment: str, **kwargs):
        logger.info("Initializing PineconeDocs...")
        try:
            # Initialize with a custom texts_index
            pinecone_store = PQAPineconeVectorStore(
                index_name=index_name, api_key=api_key, environment=environment
            )
            logger.info("Successfully created PineconeVectorStore")

            # Set the texts_index before calling super().__init__
            kwargs["texts_index"] = pinecone_store

            # Initialize the rest of the Docs class
            super().__init__(**kwargs)
            logger.info("Successfully initialized base Docs class")

        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    async def aadd_texts(self, texts, doc, settings=None, embedding_model=None):
        """Override aadd_texts to ensure proper indexing"""
        logger.info(f"Adding {len(texts)} texts for document {doc.docname}")
        try:
            # Call parent implementation
            result = await super().aadd_texts(texts, doc, settings, embedding_model)

            if result:
                logger.info(f"Successfully added document {doc.docname}")
                logger.info(f"Current number of texts: {len(self.texts)}")
                logger.info(f"Current size of texts_index: {len(self.texts_index)}")

                # Ensure all texts are properly indexed
                if len(self.texts) != len(self.texts_index):
                    logger.info("Synchronizing texts index...")
                    unindexed_texts = [
                        t for t in self.texts if t not in self.texts_index
                    ]
                    if unindexed_texts:
                        self.texts_index.add_texts_and_embeddings(unindexed_texts)
                        logger.info(f"Added {len(unindexed_texts)} texts to index")

            return result

        except Exception as e:
            logger.error(f"Error in aadd_texts: {str(e)}", exc_info=True)
            raise

    async def aquery(self, query_text: str, **kwargs):
        """Override aquery to ensure texts are indexed before querying"""
        logger.info(f"Processing query: {query_text}")
        try:
            # Ensure all texts are indexed before querying
            if len(self.texts) != len(self.texts_index):
                logger.info("Synchronizing texts index before query...")
                unindexed_texts = [t for t in self.texts if t not in self.texts_index]
                if unindexed_texts:
                    self.texts_index.add_texts_and_embeddings(unindexed_texts)
                    logger.info(f"Added {len(unindexed_texts)} texts to index")

            result = await super().aquery(query_text, **kwargs)

            logger.info(f"Query completed with {len(result.contexts)} contexts")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise


class PQAProcessor:
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
        vector_db: RedisVectorStore,
        metadata_keys: list[str] = DocumentMetadata.keys(),
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

        self._settings: pqa.Settings = pqa_settings
        self._llm: LiteLLMModel = self.llm
        self._metadata_keys: list[str] = metadata_keys
        self._vector_db: RedisVectorStore = vector_db

    @property
    def llm(self) -> LiteLLMModel:
        """Get the LLM model."""
        return self._settings.get_llm()

    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model."""
        return self._settings.get_embedding_model()

    @property
    def vector_store(self) -> RedisVectorStore:
        """Get the underlying vector store for operations that do not require
        document type conversion.
        """
        return self._vector_db

    async def chunk_and_embed(
        self, input: Union[os.PathLike, bytes, str, BytesIO]
    ) -> tuple[list[str], list[float]]:
        """Process a single document and return a tuple of chunks and embeddings."""
        try:
            chunks = self.chunk_pdf(input)
            # Utilize paper-qa to retrieve general embeddings
            embedding_model = self._settings.get_embedding_model()
            embeddings = await embedding_model.embed_documents(chunks)

        except Exception as e:
            logger.error(f"Error building document: {str(e)}", exc_info=True)
            raise

        return chunks, embeddings

    async def process_documents(
        self, input: List[Union[os.PathLike, bytes, str, BytesIO]]
    ) -> list[tuple[dict, list[str]]]:
        """
        Driver function for processing a list of documents and storing basic metadata.

        Parameters:
        -----------
        input: List[Union[os.PathLike, bytes, str, BytesIO]]
            List of documents to process.
        """
        processed_docs: list[tuple[dict, list[str]]] = []

        for _doc in input:
            try:
                chunks, embeddings = await self.chunk_and_embed(_doc)

                # Take the first two chunks to extract metadata
                chunk = " ".join(chunks[0:2])
                metadata = await extract_and_enrich_metadata(
                    llm=self.llm,
                    text_chunks=[chunk],
                    metadata_keys=self._metadata_keys,
                )

                # Add MLA citation to metadata if not already present
                if not metadata[0].get("citation"):
                    metadata[0]["citation"] = await pqa_build_mla(
                        llm=self.llm, **metadata[0]
                    )

                # Annotate creation of document
                metadata[0]["created_at"] = str(datetime.now().isoformat())
                metadatas = [metadata[0]] * len(chunks)

                keys = await self.vector_store.add_texts(
                    texts=chunks, metadatas=metadatas, embeddings=embeddings
                )
                processed_docs.append((metadata, keys))
                logger.info(
                    f"Processed document:\n{metadata}\nwith {len(keys)} chunks."
                )
            except Exception as e:
                logger.error(
                    f"Error processing document {metadata}: {str(e)}\nSkipping..."
                )
                raise

        return processed_docs

    @staticmethod
    def chunk_pdf(
        input: Union[os.PathLike, bytes, str, BytesIO],
        chunk_chars: int = 1500,
        overlap: int = 100,
    ) -> list[str]:
        """
        Chunk a PDF document into chunks of a given size. Uses paper-qa's chunk_pdf
        function for the actual chunking, but pagination is handled by either paperqa or
        custom logic depending on the input type.
        """
        # dummy doc object
        doc = PQADoc(docname="", citation="", dockey="")
        try:
            if isinstance(input, (os.PathLike, str)):
                parsed_text = pqa_parse_pdf_to_pages(input)
            elif isinstance(input, (bytes, BytesIO)):
                parsed_text = PQAProcessor.parse_pdf_bytes_to_pages(input)
            else:
                raise ValueError(f"Unsupported input type: {type(input)}")
        except Exception as e:
            logger.error(f"Error chunking document: {str(e)}", exc_info=True)
            raise

        # We won't use the doc argument, but it's required by paper-qa's chunk_pdf
        chunks: list[PQAText] = pqa_chunk_pdf(
            parsed_text=parsed_text, doc=doc, chunk_chars=chunk_chars, overlap=overlap
        )

        return [chunk.text for chunk in chunks]

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
