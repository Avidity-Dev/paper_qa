from paperqa import Docs
from paperqa.llms import VectorStore
from pydantic import Field
from functools import partial
from dotenv import load_dotenv
import os
import logging
import asyncio  # Added for async operation handling

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
from pinecone_store import PineconeVectorStore

class PineconeDocs(Docs):
    """Extension of paperqa's Docs class that uses Pinecone for vector storage."""
    
    def __init__(self, index_name: str, api_key: str, environment: str, **kwargs):
        logger.info("Initializing PineconeDocs...")
        try:
            # Initialize with a custom texts_index
            pinecone_store = PineconeVectorStore(
                index_name=index_name,
                api_key=api_key,
                environment=environment
            )
            logger.info("Successfully created PineconeVectorStore")
            
            # Set the texts_index before calling super().__init__
            kwargs['texts_index'] = pinecone_store
            
            # Initialize the rest of the Docs class
            super().__init__(**kwargs)
            logger.info("Successfully initialized base Docs class")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}", exc_info=True)
            raise

    async def aadd_texts(self, texts, doc, settings=None, embedding_model=None):
        """Override aadd_texts to ensure proper indexing"""
        logger.info(f"Adding {len(texts)} texts for document {doc.docname}")
        try:
            # Call parent implementation
            result = await super().aadd_texts(texts, doc, settings, embedding_model)
            
            if result:
                logger.info(f"Successfully added document {doc.docname}")
                logger.info(f"Current number of texts: {len(self.texts)}")
                logger.info(f"Current size of texts_index: {len(self.texts_index)}")
                
                # Ensure all texts are properly indexed
                if len(self.texts) != len(self.texts_index):
                    logger.info("Synchronizing texts index...")
                    unindexed_texts = [t for t in self.texts if t not in self.texts_index]
                    if unindexed_texts:
                        self.texts_index.add_texts_and_embeddings(unindexed_texts)
                        logger.info(f"Added {len(unindexed_texts)} texts to index")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in aadd_texts: {str(e)}", exc_info=True)
            raise

    async def aquery(self, query_text: str, **kwargs):
        """Override aquery to ensure texts are indexed before querying"""
        logger.info(f"Processing query: {query_text}")
        try:
            # Ensure all texts are indexed before querying
            if len(self.texts) != len(self.texts_index):
                logger.info("Synchronizing texts index before query...")
                unindexed_texts = [t for t in self.texts if t not in self.texts_index]
                if unindexed_texts:
                    self.texts_index.add_texts_and_embeddings(unindexed_texts)
                    logger.info(f"Added {len(unindexed_texts)} texts to index")
            
            result = await super().aquery(query_text, **kwargs)
            
            logger.info(f"Query completed with {len(result.contexts)} contexts")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            raise

async def main():
    try:
        logger.info("Starting script execution...")
        
        # Initialize PineconeDocs
        docs = PineconeDocs(
            index_name=os.getenv('PINECONE_INDEX_NAME'),
            api_key=os.getenv('PINECONE_API_KEY'),
            environment=os.getenv('PINECONE_ENVIRONMENT'),
            name="my-docs"
        )
        logger.info("PineconeDocs instance created")

        # Change docs.add() to await docs.aadd()
        file_path = "../../data/RAW/DeepSipred A deep-learning-based approach on siRNA.pdf"
        logger.info(f"Adding document: {file_path}")
        docname = await docs.aadd(file_path)  # Changed from docs.add to await docs.aadd
        logger.info(f"Added document with name: {docname}")

        # Print current state
        logger.info(f"Number of documents: {len(docs.docs)}")
        logger.info(f"Number of texts: {len(docs.texts)}")
        logger.info(f"Size of texts_index: {len(docs.texts_index)}")

        # Execute query
        query_text = "What is the main topic of the document?"
        logger.info(f"Executing query: {query_text}")
        answer = await docs.aquery(query_text)
        
        # Log answer details
        logger.info(f"Answer contains {len(answer.contexts)} contexts")
        logger.info(f"Answer text length: {len(answer.answer)}")
        print("\nAnswer:", answer)

    except Exception as e:
        logger.error("Script failed with error:", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())