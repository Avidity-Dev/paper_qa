from typing import List, Optional, Dict
import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_environment_variables(env_file: str = '.env') -> dict:
    """
    Load environment variables from a specified .env file.
    Args:
        env_file (str): Path to the .env file
    Returns:
        dict: Dictionary containing required environment variables
    """
    load_dotenv(env_file)
    
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

class CreatePineconeIndex:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        """
        Initialize Pinecone index with proper error handling and validation.
        """
        self.index_name = index_name
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {index_name}")
                
                # Create index with specified configuration
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,  # OpenAI embedding dimension
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=pinecone_environment
                    )
                )
                
                # Wait for index to be ready with timeout
                max_wait_time = 300  # 5 minutes timeout
                start_time = time.time()
                
                while index_name not in self.pc.list_indexes().names():
                    if time.time() - start_time > max_wait_time:
                        raise TimeoutError(f"Index creation timed out after {max_wait_time} seconds")
                    logger.info("Waiting for index to be created...")
                    time.sleep(5)
                
                logger.info(f"Successfully created Pinecone index: {index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {index_name}")
            
            # Verify index configuration
            index_description = self.pc.describe_index(index_name)
            if index_description.dimension != 1536:
                raise ValueError(f"Existing index has incorrect dimension: {index_description.dimension}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone: {str(e)}")
            raise

def main():
    try:
        env_vars = load_environment_variables()
        index = CreatePineconeIndex(
            pinecone_api_key=env_vars['PINECONE_API_KEY'],
            pinecone_environment=env_vars['PINECONE_ENVIRONMENT'],
            index_name=env_vars['PINECONE_INDEX_NAME']
        )
        return index
    except Exception as e:
        logger.error(f"Failed to create Pinecone index: {str(e)}")
        raise

if __name__ == "__main__":
    main()