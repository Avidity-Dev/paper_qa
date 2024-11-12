import pytest
from datetime import datetime
import os
from src.document_processor import DocumentProcessor, DocumentMetadata
from unittest.mock import Mock, patch
import paperqa

class TestDocumentProcessor:
    """Test suite for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Fixture to create a fresh DocumentProcessor instance for each test."""
        return DocumentProcessor()

    def test_initialization(self, processor):
        """Test that DocumentProcessor initializes with correct default state."""
        assert hasattr(processor, 'docs')
        assert isinstance(processor.document_metadata, list)
        assert len(processor.document_metadata) == 0

    @patch('paperqa.Docs')
    def test_process_document_valid(self, mock_docs_class):
        """Test processing a valid document."""
        # Create processor with mocked Docs
        processor = DocumentProcessor()
        test_file = "test_document.pdf"
        
        # Replace processor.docs with a new mock
        processor.docs = Mock()
        
        # Process the document
        metadata = processor.process_document(test_file)
        processor.docs.add.assert_called_once_with(test_file)
        
        # Check metadata was created correctly
        assert metadata is not None
        assert metadata.filepath == test_file
        assert metadata.filename == "test_document.pdf"
        assert isinstance(metadata.processed_date, datetime)
        
        # Check metadata was stored
        assert len(processor.document_metadata) == 1
        assert processor.document_metadata[0] == metadata

    def test_process_document_invalid(self, processor):
        """Test processing an invalid document."""
        metadata = processor.process_document("nonexistent.pdf")
        assert metadata is None
        assert len(processor.document_metadata) == 0

    def test_get_processed_files_empty(self, processor):
        """Test getting processed files when none have been processed."""
        files = processor.get_processed_files()
        assert isinstance(files, list)
        assert len(files) == 0

    @patch('paperqa.Docs')
    def test_get_processed_files_with_documents(self, mock_docs_class):
        """Test getting processed files after processing documents."""
        processor = DocumentProcessor()
        
        # Replace processor.docs with a new mock
        processor.docs = Mock()
        
        # Process test documents
        processor.process_document("test1.pdf")
        processor.process_document("test2.pdf")
        
        files = processor.get_processed_files()
        assert len(files) == 2
        assert "test1.pdf" in files
        assert "test2.pdf" in files

    def test_query_documents(self, processor):
        """Test querying documents."""
        question = "test question"
        result = processor.query_documents(question)
        assert isinstance(result, str)
        assert len(result) > 0

    @patch('paperqa.Docs')
    def test_query_documents_error(self, mock_docs_class):
        """Test querying documents when an error occurs."""
        processor = DocumentProcessor()
        
        # Replace processor.docs with a new mock
        processor.docs = Mock()
        processor.docs.query.side_effect = Exception("Query error")
        
        result = processor.query_documents("test question")
        assert "Sorry, I encountered an error while processing your query" in result