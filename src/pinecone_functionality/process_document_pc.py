from typing import List, Tuple, Set
import pinecone
from paperqa.types import Embeddable
from paperqa.llms import VectorStore
import numpy as np

class PineconeVectorStore(VectorStore):
    """Custom Pinecone vector store implementation for paper-qa."""
    
    def __init__(self, index_name: str, api_key: str, environment: str):
        """Initialize Pinecone vector store."""
        super().__init__()
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        self.index = pinecone.Index(index_name)
        self.texts_hashes: Set[int] = set()  # Track stored text hashes
        
    def add_texts_and_embeddings(self, texts: List[Embeddable]) -> None:
        """Add texts and their embeddings to Pinecone."""
        vectors_to_upsert = []
        
        for text in texts:
            vector_id = str(hash(text))  # Convert hash to string for Pinecone
            # Include necessary metadata
            metadata = {
                "text": text.text,  # Original text needed for retrieval
                "name": getattr(text, 'name', ''),  # Document name if available
                "citation": getattr(text, 'citation', '')  # Citation if available
            }
            # Convert embedding to list for Pinecone
            embedding = text.embedding.tolist() if isinstance(text.embedding, np.ndarray) else text.embedding
            
            vectors_to_upsert.append((vector_id, embedding, metadata))
        
        # Batch upsert to Pinecone
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            # Update tracked hashes
            self.texts_hashes.update(hash(t) for t in texts)
    
    async def similarity_search(
        self, 
        query: str, 
        k: int, 
        embedding_model
    ) -> Tuple[List[Embeddable], List[float]]:
        """Perform similarity search in Pinecone."""
        # Get query embedding
        query_embedding = (await embedding_model.embed_documents([query]))[0]
        
        # Convert numpy array to list if necessary
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            include_values=True  # Need embeddings for Embeddable objects
        )
        
        # Create Embeddable objects from results
        embeddables = []
        scores = []
        
        for match in results.matches:
            embeddable = Embeddable(
                text=match.metadata["text"],
                embedding=np.array(match.values),  # Convert back to numpy array
                name=match.metadata.get("name", ""),
                citation=match.metadata.get("citation", "")
            )
            embeddables.append(embeddable)
            scores.append(match.score)
        
        return embeddables, scores
    
    def clear(self) -> None:
        """Clear all vectors from the index."""
        try:
            # Delete all vectors
            self.index.delete(delete_all=True)
            # Clear tracked hashes
            self.texts_hashes.clear()
        except Exception as e:
            raise Exception(f"Failed to clear Pinecone index: {str(e)}")
    
    def __len__(self) -> int:
        """Return number of vectors in store."""
        try:
            stats = self.index.describe_index_stats()
            return stats.total_vector_count
        except Exception as e:
            raise Exception(f"Failed to get index stats: {str(e)}")
        