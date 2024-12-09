"""
Core data structures.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from pydantic import BaseModel, ConfigDict

from paperqa.types import Text
from paperqa.types import Doc as PQADoc


class PQADocument(PQADoc, BaseModel):
    """
    Data model for a research paper.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    doi: str | None = None
    published_date: datetime | None = None
    created_at: datetime | None = None
    text_chunks: list[str] | None = None
    embeddings: list[list[float]] | None = None
    dockey: str | None = None
    docname: str | None = None
    citation: str | None = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def to_dict(self) -> dict:
        """Convert the Document object to a dictionary, extracting the text from the
        text_chunks.
        """
        out = {k: v for k, v in self.__dict__.items() if k != "text_chunks"}
        return out


class DocumentMetadata(BaseModel):
    """
    Data model for metadata extracted from a research paper.
    """

    title: str | None = None
    authors: list[str] | None = None
    doi: str | None = None
    published_date: datetime | None = None
    citation: str | None = None
    journal: str | None = None
    volume: str | None = None
    issue: str | None = None


class DocumentChunk(BaseModel):
    """
    Data model for a chunk of a research paper.
    """

    id: str | None = None
    key: str | None = None
    text: str | None = None
    pages: list[int] | None = None
    embedding: list[float] | None = None
    metadata: DocumentMetadata | None = None

    def to_dict(self, embedding: bool = False) -> dict:
        """Return a dictionary representation of the DocumentChunk object.

        Parameters:
        -----------
        embedding: bool, optional
            Whether to include the embedding in the dictionary, by default False
        """
        out = {**self.model_dump(), **self.metadata.model_dump()}
        if embedding:
            return out
        else:
            del out["embedding"]
            return out
