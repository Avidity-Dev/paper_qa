from typing import Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain.docstore.document import Document as LangchainDocument
from paperqa import Doc
from paperqa.readers import read_doc
from dotenv import load_dotenv
import hashlib
import os
from datetime import datetime

class DocumentProcessor:
    """Simple document processor that stores embeddings in Pinecone."""
    
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        """Initialize with Pinecone credentials."""
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize Pinecone index
        self.index = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings
        )

    def process_document(self, filepath: str) -> Optional[Dict]:
        """Process a document and store its embeddings in Pinecone."""
        try:
            filename = os.path.basename(filepath)
            print(f"\nProcessing document: {filename}")
            
            # Generate document ID
            doc_id = hashlib.md5(filepath.encode()).hexdigest()
            
            # Create Doc object with required fields
            doc = Doc(
                docname=filename,
                citation=f"File: {filename}",
                dockey=doc_id
            )
            
            # Use paper-qa's read_doc function directly
            print("Loading document...")
            texts = read_doc(
                filepath,
                doc,
                chunk_chars=1500,
                overlap=150
            )
            
            if not texts:
                print("No text chunks were generated")
                return None
            
            print(f"Generated {len(texts)} text chunks")
            
            # Create Langchain documents
            langchain_docs = []
            for i, text in enumerate(texts):
                chunk_id = hashlib.md5(f"{doc_id}_{i}".encode()).hexdigest()
                langchain_docs.append(
                    LangchainDocument(
                        page_content=text.text,
                        metadata={
                            "text": text.text,
                            "doc_name": text.name,
                            "doc_id": doc_id,
                            "citation": doc.citation,
                            "chunk_index": i,
                            "chunk_id": chunk_id
                        }
                    )
                )
            
            # Store text chunks with metadata
            print("Storing chunks in Pinecone...")
            self.index.add_documents(langchain_docs)
            
            # Store document metadata
            metadata = {
                "filepath": filepath,
                "filename": filename,
                "doc_id": doc_id,
                "chunk_count": len(texts),
                "processed_date": datetime.now().isoformat(),
                "type": "document_metadata"
            }
            
            # Store document metadata as a document
            metadata_doc = LangchainDocument(
                page_content=f"Document metadata for {filename}",
                metadata=metadata
            )
            self.index.add_documents([metadata_doc])
            
            print(f"Successfully processed document with {len(texts)} chunks")
            return metadata
            
        except Exception as e:
            print(f"Error processing document {filepath}: {str(e)}")
            return None

def load_environment_variables():
    """Load environment variables from .env file."""
    load_dotenv()
    
    required_vars = {
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
        'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return required_vars

if __name__ == "__main__":
    # Load environment variables
    env_vars = load_environment_variables()
    
    # Initialize processor with environment variables
    processor = DocumentProcessor(
        pinecone_api_key=env_vars['PINECONE_API_KEY'],
        pinecone_environment=env_vars['PINECONE_ENVIRONMENT'],
        index_name=env_vars['PINECONE_INDEX_NAME']
    )
    
    # Process your document
    filepath = "insert_path_here.pdf"
    metadata = processor.process_document(filepath)
    if metadata:
        print(f"Successfully processed document: {metadata['filename']}")
        processor.verify_document(metadata['doc_id'])
        