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
from dotenv import load_dotenv
import time

# Configure logging
logging.getLogger('paper_qa').setLevel(logging.ERROR)

def load_environment_variables(env_file: str = '.env') -> dict:
    """
    Load environment variables from a specified .env file.
    Args:
        env_file (str): Path to the .env file
    Returns:
        dict: Dictionary containing required environment variables
    """
    load_dotenv(env_file)
    
    # Required environment variables
    required_vars = {
        'PINECONE_API_KEY': os.getenv('PINECONE_API_KEY'),
        'PINECONE_ENVIRONMENT': os.getenv('PINECONE_ENVIRONMENT'),
        'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY')
    }
    
    # Check for missing environment variables
    missing_vars = [var for var, value in required_vars.items() if not value]
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return required_vars

class CreatePineconeIndex:
    def __init__(self, pinecone_api_key: str, pinecone_environment: str, index_name: str):
        warnings.filterwarnings('ignore', message='.*API.*')
        warnings.filterwarnings('ignore', message='.*Provider.*')
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=pinecone_api_key)
        
        try:
            # Create index if it doesn't exist
            if index_name not in self.pc.list_indexes().names():
                print(f"Creating new Pinecone index: {index_name}")
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=pinecone_environment
                    )
                )
                
                # Wait for index to be ready
                while index_name not in self.pc.list_indexes().names():
                    print("Waiting for index to be created...")
                    time.sleep(1)
                print(f"Successfully created Pinecone index: {index_name}")
            else:
                print(f"Using existing Pinecone index: {index_name}")
            
        except Exception as e:
            print(f"Error initializing Pinecone: {str(e)}")
            raise  

if __name__ == "__main__":
    env_vars = load_environment_variables()
    index = CreatePineconeIndex(
        pinecone_api_key=env_vars['PINECONE_API_KEY'],
        pinecone_environment=env_vars['PINECONE_ENVIRONMENT'],
        index_name=env_vars['PINECONE_INDEX_NAME']
    )