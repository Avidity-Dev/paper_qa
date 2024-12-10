"""
Adapters for connecting to cloud storage services to store and retrieve documents.
"""

from abc import ABC, abstractmethod
from io import BytesIO
import logging
import os
from typing import BinaryIO, Optional, Dict, Any

import aioboto3
from azure.storage.blob import BlobClient, ContentSettings
from azure.storage.blob.aio import (
    BlobServiceClient,
    ContainerClient as AsyncContainerClient,
)
from google.cloud import storage

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


class AWSS3Store(CloudObjectStore):
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


class AzureBlobStore(CloudObjectStore):
    """Azure Blob Storage implementation of cloud storage adapter."""

    def __init__(
        self,
        connection_string: str,
        container_name: str,
    ):
        """Initialize Azure Blob Storage client."""
        self.container_name = container_name
        self.connection_string = connection_string
        self.blob_service_client: Optional[BlobServiceClient] = None
        self.container_client: Optional[AsyncContainerClient] = None

    async def __aenter__(self) -> "AzureBlobStore":
        """Async context manager entry.

        Returns
        -------
        AzureBlobStore
            The initialized store instance
        """
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.

        Ensures proper cleanup of Azure clients.
        """
        await self.close()

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
            async with blob_client as blob_client:
                download_stream = await blob_client.download_blob()
                async for chunk in download_stream.chunks():
                    stream.write(chunk)
            await blob_client.close()  # need to explicity close the blob client or else aiohttp will log warnings
            stream.seek(0)
            return stream
        except Exception as e:
            logger.error(f"Error downloading from Azure Blob: {str(e)}")
            raise

    async def list_files(self) -> list[str]:
        """List all files in the container."""
        await self._ensure_client()
        blob_list = []
        async for blob in self.container_client.list_blobs():
            blob_list.append(blob)

        return blob_list

    async def delete_file(self, file_name: str) -> bool:
        try:
            await self._ensure_client()
            blob_client = self.container_client.get_blob_client(file_name)
            await blob_client.delete_blob()
            return True
        except Exception as e:
            logger.error(f"Error deleting from Azure Blob: {str(e)}")
            return False

    async def close(self):
        """Close any sockets opened by the client."""
        if self.container_client:
            await self.container_client.close()
        if self.blob_service_client:
            await self.blob_service_client.close()


class CloudStorageFactory:
    """Factory for creating cloud storage adapters."""

    @staticmethod
    def create_storage_adapter(
        provider: str, config: Dict[str, Any]
    ) -> CloudObjectStore:
        """Create a cloud storage adapter instance."""
        if provider == "aws":
            return AWSS3Store(
                bucket_name=config["bucket_name"],
                aws_access_key_id=config.get("aws_access_key_id"),
                aws_secret_access_key=config.get("aws_secret_access_key"),
                region_name=config.get("region_name"),
            )
        elif provider == "azure":
            return AzureBlobStore(
                connection_string=config["connection_string"],
                container_name=config["container_name"],
            )
        else:
            raise ValueError(f"Unsupported storage provider: {provider}")
