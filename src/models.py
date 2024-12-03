"""
Core data structures.
"""

from dataclasses import dataclass
from datetime import datetime


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
    text_chunks: list | None = None

    def to_dict(self) -> dict:
        """Convert the Document object to a dictionary, removing the text_chunks."""
        return {k: v for k, v in self.__dict__.items() if k != "text_chunks"}
