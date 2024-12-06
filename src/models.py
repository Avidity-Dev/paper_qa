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
