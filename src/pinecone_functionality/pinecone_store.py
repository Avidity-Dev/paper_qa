from pinecone import Pinecone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from paperqa import Docs, Doc, Text
from paperqa.llms import LiteLLMEmbeddingModel

@dataclass
class Document:
    """Simple document class to hold text and metadata."""
    page_content: str
    metadata: dict

class PineconeStore:
    def __init__(self, api_key: str, environment: str, index_name: str, embedding_model_name: str = "text-embedding-3-small"):
        """Initialize Pinecone client and index."""
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.embedding_model = LiteLLMEmbeddingModel(name=embedding_model_name)
    
    async def store_documents(self, documents: List[Document]):
        """
        Store documents in Pinecone.
        
        Args:
            documents: List of Document objects containing text and metadata
        """
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding_model.embed_documents(texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            vector_id = f"vec_{i}"
            metadata = {
                "text": doc.page_content,
                "docname": doc.metadata.get("docname", ""),
                "citation": doc.metadata.get("citation", ""),
                "dockey": doc.metadata.get("dockey", ""),
                "name": doc.metadata.get("name", "")
            }
            vectors.append((vector_id, embedding, metadata))
        
        # Upsert to Pinecone in batches
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
    
    async def retrieve_docs(self, query: str, k: int = 4) -> Docs:
        """
        Retrieve documents from Pinecone and build paper-qa Docs object.
        
        Args:
            query: Query text
            k: Number of documents to retrieve
            
        Returns:
            paper-qa Docs object containing retrieved documents
        """
        # Get query embedding
        query_embedding = (await self.embedding_model.embed_documents([query]))[0]
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True
        )
        
        # Build paper-qa Docs object
        docs = Docs()
        
        for match in results.matches:
            # Create Doc object
            doc = Doc(
                docname=match.metadata['docname'],
                citation=match.metadata['citation'],
                dockey=match.metadata.get('dockey', '')
            )
            
            # Create Text object
            text = Text(
                text=match.metadata['text'],
                name=match.metadata.get('name', ''),
                doc=doc
            )
            
            # Add to Docs collection
            docs.add_texts([text], doc)
        
        return docs

    def delete_all(self):
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)