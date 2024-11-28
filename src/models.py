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

    id: str
    doi: str
    title: str
    authors: list[str]
    published_date: datetime
    created_at: datetime
    vector_key: str
    relational_key: str
