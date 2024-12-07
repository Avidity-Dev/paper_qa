"""
Adapters for connecting to cloud storage services to store and retrieve documents. 

Great for storing documents in a cloud storage service and then using them in a Pinecone index. 

Great work by Seth.
"""

from abc import ABC, abstractmethod
from io import BytesIO
import logging
import os
from typing import BinaryIO, Optional, Dict, Any

import aioboto3
from azure.storage.blob import BlobClient, ContentSettings
from azure.storage.blob.aio import BlobServiceClient
import pymupdf

logger = logging.getLogger(__name__)


class CloudObjectStore(ABC):
    """Abstract base class for cloud storage providers."""

    @abstractmethod
    async def upload_file(
        self, file_data: BinaryIO, file_name: str, content_type: Optional[str] = None
    ) -> Dict[str, str]:
        """Upload a file to cloud storage.

        Parameters
        ----------
        file_data : BinaryIO
            File-like object containing the file data
        file_name : str
            Name of the file to store
        content_type : str, optional
            MIME type of the file

        Returns
        -------
        dict
            Dictionary containing upload details (url, path, etc.)
        """
        pass

    @abstractmethod
    async def download_file(self, file_name: str) -> BytesIO:
        """Download a file from cloud storage.

        Parameters
        ----------
        file_name : str
            Name of the file to download

        Returns
        -------
        BytesIO
            BytesIO object containing the file data
        """
        pass

    @abstractmethod
    async def delete_file(self, file_name: str) -> bool:
        """Delete a file from cloud storage.

        Parameters
        ----------
        file_name : str
            Name of the file to delete

        Returns
        -------
        bool
            Boolean indicating success
        """
        pass


class AWSS3Adapter(CloudObjectStore):
    """AWS S3 implementation of cloud storage adapter."""

    def __init__(
        self,
        bucket_name: str,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None,
    ):
        """Initialize AWS S3 client."""
        self.bucket_name = bucket_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.session = aioboto3.Session()

    async def _get_client(self):
        """Get an async S3 client."""
        return await self.session.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            region_name=self.region_name,
        ).__aenter__()

    async def upload_file(
        self, file_data: BinaryIO, file_name: str, content_type: Optional[str] = None
    ) -> Dict[str, str]:
        try:
            client = await self._get_client()
            extra_args = {"ContentType": content_type} if content_type else {}

            await client.upload_fileobj(
                file_data, self.bucket_name, file_name, ExtraArgs=extra_args
            )

            url = f"https://{self.bucket_name}.s3.amazonaws.com/{file_name}"
            return {
                "url": url,
                "provider": "aws",
                "bucket": self.bucket_name,
                "path": file_name,
            }
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

    async def download_file(self, file_name: str) -> BytesIO:
        try:
            client = await self._get_client()
            file_obj = BytesIO()
            await client.download_fileobj(self.bucket_name, file_name, file_obj)
            file_obj.seek(0)
            return file_obj
        except Exception as e:
            logger.error(f"Error downloading from S3: {str(e)}")
            raise

    async def delete_file(self, file_name: str) -> bool:
        try:
            client = await self._get_client()
            await client.delete_object(Bucket=self.bucket_name, Key=file_name)
            return True
        except Exception as e:
            logger.error(f"Error deleting from S3: {str(e)}")
            return False


class AzureBlobAdapter(CloudObjectStore):
    """Azure Blob Storage implementation of cloud storage adapter."""

    def __init__(self, connection_string: str, container_name: str):
        """Initialize Azure Blob Storage client."""
        self.container_name = container_name
        self.connection_string = connection_string
        self.blob_service_client = None
        self.container_client = None

    async def _ensure_client(self):
        """Ensure blob service and container clients are initialized."""
        if not self.blob_service_client:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )

    async def upload_file(
        self, file_data: BinaryIO, file_name: str, content_type: Optional[str] = None
    ) -> Dict[str, str]:
        try:
            await self._ensure_client()
            blob_client = self.container_client.get_blob_client(file_name)

            content_settings = None
            if content_type:

                content_settings = ContentSettings(content_type=content_type)

            await blob_client.upload_blob(
                file_data, content_settings=content_settings, overwrite=True
            )

            return {
                "url": blob_client.url,
                "provider": "azure",
                "container": self.container_name,
                "path": file_name,
            }
        except Exception as e:
            logger.error(f"Error uploading to Azure Blob: {str(e)}")
            raise

    async def download_file(self, file_name: str) -> BytesIO:
        try:
            await self._ensure_client()
            blob_client = self.container_client.get_blob_client(file_name)
            stream = BytesIO()
            download_stream = await blob_client.download_blob()
            stream.write(await download_stream.readall())
            stream.seek(0)
            return stream
        except Exception as e:
            logger.error(f"Error downloading from Azure Blob: {str(e)}")
            raise

    async def delete_file(self, file_name: str) -> bool:
        try:
            await self._ensure_client()
            blob_client = self.container_client.get_blob_client(file_name)
            await blob_client.delete_blob()
            return True
        except Exception as e:
            logger.error(f"Error deleting from Azure Blob: {str(e)}")
            return False


class CloudStorageFactory:
    """Factory for creating cloud storage adapters."""

    @staticmethod
    def create_storage_adapter(
        provider: str, config: Dict[str, Any]
    ) -> CloudObjectStore:
        """Create a cloud storage adapter instance."""
        if provider == "aws":
            return AWSS3Adapter(
                bucket_name=os.getenv('AWS_BUCKET_NAME'),
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_REGION_NAME')
            )
        elif provider == "azure":
            return AzureBlobAdapter(
                connection_string=config["connection_string"],
                container_name=config["container_name"],
            )
        else:
            raise ValueError(f"Unsupported storage provider: {provider}")


async def process_documents():
    """Example of how to upload documents to AWS S3."""
    # Create S3 adapter
    s3_adapter = CloudStorageFactory.create_storage_adapter("aws", {})
    
    # Directory containing your documents
    docs_directory = os.path.join(os.path.dirname(__file__), "../../data/RAW/")
    
    # Get list of PDF files in directory
    pdf_files = [f for f in os.listdir(docs_directory) if f.endswith('.pdf')]
    
    for pdf_file in pdf_files:
        local_path = os.path.join(docs_directory, pdf_file)
        try:
            with open(local_path, "rb") as file:
                # Upload file using original filename
                result = await s3_adapter.upload_file(
                    file_data=file,
                    file_name=pdf_file,  # Using original filename
                    content_type="application/pdf"
                )
                print(f"Successfully uploaded {pdf_file}")
                print(f"Access URL: {result['url']}")
                
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(process_documents())