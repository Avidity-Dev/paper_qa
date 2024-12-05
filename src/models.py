"""
Core data structures.
"""

from dataclasses import dataclass
from datetime import datetime

from paperqa.types import Text


@dataclass
class Document:
    """
    Universal data model to represent research papers within the application.
    """

    id: str | None = None
    title: str | None = None
    authors: list[str] | None = None
    doi: str | None = None
    published_date: datetime | None = None
    created_at: datetime | None = None
    text_chunks: list[Text] | None = None

    def to_dict(self) -> dict:
        """Convert the Document object to a dictionary, extracting the text from the
        text_chunks.
        """
        out = {k: v for k, v in self.__dict__.items() if k != "text_chunks"}
        return out
