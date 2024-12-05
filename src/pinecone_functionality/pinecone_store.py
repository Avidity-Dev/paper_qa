import numpy as np
from enum import StrEnum
from paperqa.types import Text
from paperqa.llms import VectorStore
from paperqa.llms import EmbeddingModel
from paperqa.types import Embeddable
from typing import Iterable, Sequence
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class EmbeddingModes(StrEnum):
    DOCUMENT = "document"
    QUERY = "query"

class PineconeVectorStore(VectorStore, BaseModel):
    """Pinecone implementation of VectorStore interface that maintains compatibility with paperqa."""
    
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    pc: Pinecone = None
    index: any = None
    texts: list[Embeddable] = []  # Keep track of texts just like NumpyVectorStore
    mmr_lambda: float = 0.9  # Default value matching paperqa
    
    def __init__(self, index_name: str, api_key: str, environment: str):
        """Initialize PineconeVectorStore.
        
        Args:
            index_name: Name of the Pinecone index
            api_key: Pinecone API key
            environment: Pinecone environment
        """
        super(BaseModel, self).__init__()
        super(VectorStore, self).__init__()
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(index_name)
        self.texts = []
        logger.info(f"Initialized PineconeVectorStore with index: {index_name}")
    
    def __eq__(self, other) -> bool:
        """Implement equality comparison like NumpyVectorStore."""
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.texts == other.texts
            and self.texts_hashes == other.texts_hashes
            and self.mmr_lambda == other.mmr_lambda
        )

    def add_texts_and_embeddings(self, texts: Iterable[Embeddable]) -> None:
        """Add texts and their embeddings to Pinecone."""
        super().add_texts_and_embeddings(texts)  # Update texts_hashes
        texts_list = list(texts)
        self.texts.extend(texts_list)  # Keep local record of texts
        
        logger.info(f"Adding {len(texts_list)} texts to Pinecone")
        
        # Prepare vectors for Pinecone
        upserts = []
        for text in texts_list:
            if text.embedding is None:
                logger.warning(f"Text {text.name} has no embedding!")
                continue
            
            # Convert embedding to list if it's numpy array
            embedding = text.embedding.tolist() if hasattr(text.embedding, 'tolist') else text.embedding
            
            upserts.append({
                'id': text.name,
                'values': embedding,
                'metadata': {
                    'text': text.text,
                    'name': text.name
                }
            })
        
        if upserts:
            try:
                self.index.upsert(vectors=upserts)
                logger.info(f"Successfully upserted {len(upserts)} vectors to Pinecone")
            except Exception as e:
                logger.error(f"Failed to upsert vectors: {str(e)}")
                raise

    async def similarity_search(
        self, query: str, k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        logger.info(f"Starting similarity search for query: {query[:50]}...")
        
        k = min(k, len(self.texts))
        if k == 0:
            logger.info("No texts to search through")
            return [], []

        try:
            # Get query embedding
            embedding_model.set_mode(EmbeddingModes.QUERY)
            query_embedding = (await embedding_model.embed_documents([query]))[0]
            embedding_model.set_mode(EmbeddingModes.DOCUMENT)
            logger.info("Generated query embedding")

            # Convert to list if it's numpy array
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

            # Use Pinecone's native query
            query_response = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            matches = query_response.matches
            logger.info(f"Found {len(matches)} matches in Pinecone")

            # Convert Pinecone results to Embeddable objects
            results = []
            scores = []
            
            for match in matches:
                # Find the original document in self.texts
                original_doc = next(
                    t for t in self.texts 
                    if t.name == match.metadata["name"]
                )
                
                # Create new Text object (which extends Embeddable)
                text_obj = Text(
                    text=match.metadata["text"],
                    name=match.metadata["name"],
                    embedding=original_doc.embedding,
                    doc=original_doc.doc  # This should now work since we're using Text instead of Embeddable
                )
                results.append(text_obj)
                scores.append(match.score)
                
            logger.info(f"Returning {len(results)} results")
            logger.info(f"Top similarity scores: {scores[:3] if scores else []}")
            
            return results, scores
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return [], []

    async def max_marginal_relevance_search(
        self, query: str, k: int, fetch_k: int, embedding_model: EmbeddingModel
    ) -> tuple[Sequence[Embeddable], list[float]]:
        """Implement MMR search using parent VectorStore's implementation.
        
        Args:
            query: Query string
            k: Number of results to return
            fetch_k: Number of results to fetch before MMR
            embedding_model: Model to use for embedding the query
            
        Returns:
            Tuple of (list of Embeddable results, list of similarity scores)
        """
        logger.info(f"Starting MMR search with k={k}, fetch_k={fetch_k}")
        
        if fetch_k < k:
            logger.warning(f"fetch_k ({fetch_k}) must be >= k ({k}), adjusting fetch_k to {k}")
            fetch_k = k
            
        # Use parent implementation which relies on our similarity_search
        return await super().max_marginal_relevance_search(
            query, k, fetch_k, embedding_model
        )

    def clear(self) -> None:
        """Clear all data from the store."""
        try:
            self.index.delete(delete_all=True)
            self.texts.clear()
            super().clear()  # Clear the base class's texts_hashes set
            logger.info("Successfully cleared all texts and embeddings")
        except Exception as e:
            logger.error(f"Error clearing store: {e}")
            raise