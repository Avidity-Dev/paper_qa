import sys
from typing import List, Optional
import os

from dotenv import load_dotenv
from paperqa import Docs, Settings, LLMModel
from pydantic import BaseModel
from datetime import datetime
import warnings
import logging

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


class DocumentMetadata(BaseModel):
    """Data model for document metadata."""

    filepath: str
    filename: str
    processed_date: datetime = datetime.now()


class DocumentProcessor:
    """Handles document processing and querying."""

    def __init__(self):
        """Initialize the document processor."""
        # Suppress warnings about missing APIs
        warnings.filterwarnings("ignore", message=".*API.*")
        warnings.filterwarnings("ignore", message=".*Provider.*")
        self.settings = Settings(
            llm="claude-3-5-sonnet-20240620",
            summary_llm="claude-3-5-sonnet-20240620",
            llm_config=local_llm_config,
            summary_llm_config=local_llm_config,
        )
        # Initialize Docs
        self.docs = Docs()
        self.document_metadata: List[DocumentMetadata] = []

    def process_document(self, filepath: str) -> Optional[DocumentMetadata]:
        """
        Process a document and store basic metadata.

        Args:
            filepath (str): Path to the PDF document

        Returns:
            Optional[DocumentMetadata]: Document metadata if successful
        """
        try:
            # Add document to paper-qa Docs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                print(f"Adding document {filepath}")
                logging.info(f"Adding document {filepath}")
                self.docs.add(filepath, settings=self.settings)
            print(f"Document added: {filepath}")
            logging.info(f"Document added: {filepath}")
            # Create basic metadata
            metadata = DocumentMetadata(
                filepath=filepath, filename=os.path.basename(filepath)
            )

            self.document_metadata.append(metadata)
            return metadata

        except Exception as e:
            print(f"Error processing document {filepath}: {e}")
            return None

    def query_documents(self, question: str) -> str:
        """
        Query the documents with a specific question.

        Args:
            question (str): The question to ask about the documents

        Returns:
            str: The formatted answer from the query
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                answer = self.docs.query(question, settings=self.settings)
            return answer.formatted_answer
        except Exception as e:
            print(f"Error querying documents: {e}")
            return "Sorry, I encountered an error while processing your query."

    def get_processed_files(self) -> List[str]:
        """
        Get list of processed document filenames.

        Returns:
            List[str]: List of processed filenames
        """
        return [metadata.filename for metadata in self.document_metadata]
