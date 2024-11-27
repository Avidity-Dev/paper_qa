from typing import List, Optional, Dict
import os
import requests
from paperqa import Docs, Doc, Text
from pydantic import BaseModel, Field
from datetime import datetime
import warnings
import logging
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
import hashlib
import json
import time

# Configure logging
logging.getLogger('paper_qa').setLevel(logging.ERROR)

class CrossrefMetadata(BaseModel):
    """Data model for Crossref metadata."""
    doi: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)
    authors: Optional[List[str]] = Field(default_factory=list)
    published_date: Optional[str] = Field(default=None)
    related_works: Optional[List[Dict]] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

class DocumentMetadata(BaseModel):
    """Data model for document metadata."""
    filepath: str
    filename: str
    doc_id: str
    processed_date: datetime = Field(default_factory=datetime.now)
    crossref_data: Optional[CrossrefMetadata] = Field(default=None)
    vector_ids: List[str] = Field(default_factory=list)
    chunk_count: int = Field(default=0)

    class Config:
        arbitrary_types_allowed = True
class CrossrefAPI:
    """Handles interactions with Crossref API."""
    def __init__(self):
        self.base_url = "https://api.crossref.org"
        self.headers = {
            "User-Agent": f"Research_Department_RAGBot/1.0 (mailto:{os.getenv('CROSSREF_MAILTO', '')})"
        }

    def search_by_title(self, title: str) -> Optional[Dict]:
        """Search for a paper by title."""
        try:
            params = {
                "query.title": title,
                "rows": 1,
                "select": "DOI,title,author,published,reference"
            }
            response = requests.get(
                f"{self.base_url}/works",
                params=params,
                headers=self.headers
            )
            results = response.json().get("message", {}).get("items", [])
            return results[0] if results else None
        except Exception as e:
            logging.warning(f"Crossref API search error: {e}")
            return None

    def get_related_works(self, doi: str) -> List[Dict]:
        """Get related works for a given DOI."""
        try:
            response = requests.get(
                f"{self.base_url}/works/{doi}",
                headers=self.headers
            )
            return response.json().get("message", {}).get("reference", [])
        except Exception as e:
            logging.warning(f"Crossref API related works error: {e}")
            return []

class DocumentProcessor:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        try:
            warnings.filterwarnings('ignore', message='.*API.*')
            warnings.filterwarnings('ignore', message='.*Provider.*')
            
            self.document_metadata: Dict[str, DocumentMetadata] = {}
            
            # Sanitize index name
            sanitized_index_name = index_name.lower().replace('_', '-')
            sanitized_index_name = ''.join(c for c in sanitized_index_name if c.isalnum() or c == '-')
            
            # Initialize Pinecone
            self.pc = Pinecone(api_key=pinecone_api_key)
            
            # Create index if it doesn't exist
            if sanitized_index_name not in self.pc.list_indexes().names():
                print(f"Creating new Pinecone index: {sanitized_index_name}")
                try:
                    self.pc.create_index(
                        name=sanitized_index_name,
                        dimension=1536,
                        metric='cosine',
                        spec=ServerlessSpec(
                            cloud='aws',
                            region=pinecone_environment
                        )
                    )
                    # Wait for index to be ready
                    while sanitized_index_name not in self.pc.list_indexes().names():
                        print("Waiting for index to be created...")
                        time.sleep(1)
                except Exception as e:
                    print(f"Error creating index: {str(e)}")
                    raise

            # Initialize embeddings and vector store
            self.embeddings = OpenAIEmbeddings()
            self.index = self.pc.Index(sanitized_index_name)
            self.vectorstore = LangchainPinecone(
                self.index, 
                self.embeddings.embed_query, 
                "text"
            )
            print(f"Successfully initialized Pinecone with index: {sanitized_index_name}")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise

    def _process_text_chunks(self, filepath: str, doc: Doc, doc_id: str) -> List[str]:
        """Process text chunks and store them in Pinecone."""
        vector_ids = []
        try:
            print(f"\nProcessing text chunks for {os.path.basename(filepath)}...")
            
            # Create texts using paper-qa's Text class
            texts = []
            for i, chunk in enumerate(doc.text_chunks):
                chunk_id = hashlib.md5(f"{doc_id}_{i}_{chunk}".encode()).hexdigest()
                text_obj = Text(
                    text=chunk,
                    name=f"{os.path.basename(filepath)}_{chunk_id}",
                    doc=doc
                )
                texts.append((chunk_id, text_obj))
                vector_ids.append(chunk_id)

            print(f"Generated {len(texts)} text chunks")

            # Generate embeddings
            print("Generating embeddings...")
            embeddings = self.embeddings.embed_documents([t[1].text for t in texts])
            print(f"Generated {len(embeddings)} embeddings")

            # Prepare vectors with metadata
            vectors = [
                (texts[i][0], embedding, {
                    "text": texts[i][1].text,
                    "doc_name": texts[i][1].name,
                    "doc_id": doc_id,
                    "citation": doc.citation,
                    "chunk_index": i
                })
                for i, embedding in enumerate(embeddings)
            ]
            
            # Upsert to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                print(f"Upserting batch {i//batch_size + 1} of {(len(vectors)-1)//batch_size + 1}")
                self.index.upsert(vectors=batch)
            
            print("Successfully stored all vectors in Pinecone")
            return vector_ids
            
        except Exception as e:
            print(f"Error processing text chunks: {str(e)}")
            logging.error(f"Error processing text chunks: {e}")
            return []

    def process_document(self, filepath: str) -> Optional[DocumentMetadata]:
        """Process a document and store in Pinecone."""
        try:
            print(f"\nProcessing document: {os.path.basename(filepath)}")
            
            # Generate document ID
            doc_id = hashlib.md5(filepath.encode()).hexdigest()
            
            # Create Doc object
            doc = Doc(filepath)
            print("Successfully created Doc object")
            
            # Process text chunks and store in Pinecone
            vector_ids = self._process_text_chunks(filepath, doc, doc_id)
            if not vector_ids:
                print("No vectors generated for document")
                return None
                
            # Create metadata
            metadata = DocumentMetadata(
                filepath=filepath,
                filename=os.path.basename(filepath),
                doc_id=doc_id,
                vector_ids=vector_ids,
                chunk_count=len(vector_ids)
            )
            
            # Store metadata
            metadata_text = f"Document metadata for {os.path.basename(filepath)}"
            metadata_embedding = self.embeddings.embed_query(metadata_text)
            
            self.index.upsert(vectors=[(
                doc_id,
                metadata_embedding,
                {
                    "document_metadata": metadata.json(),
                    "type": "document_metadata"
                }
            )])
            
            # Store locally
            self.document_metadata[doc_id] = metadata
            print(f"Successfully processed document with {len(vector_ids)} chunks")
            return metadata
            
        except Exception as e:
            print(f"Error processing document {filepath}: {str(e)}")
            logging.error(f"Error processing document {filepath}: {e}")
            return None