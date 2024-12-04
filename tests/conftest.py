import os

import pymupdf
import pytest

from src.config.config import ConfigurationManager

TEST_PDF_PATH = "tests/data/docs/"


@pytest.fixture
def docs_list_bytes() -> list[bytes]:

    # Get all the pdfs in the test directory
    pdfs = [f for f in os.listdir(TEST_PDF_PATH) if f.endswith(".pdf")]
    pdf_bytes = []

    # Read each pdf into a byte stream
    for pdf in pdfs:
        with open(os.path.join(TEST_PDF_PATH, pdf), "rb") as f:
            pdf_bytes.append(f.read())

    return pdf_bytes


@pytest.fixture
def config() -> ConfigurationManager:
    return ConfigurationManager()
