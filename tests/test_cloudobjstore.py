from io import BytesIO
import os

import dotenv
import pytest

from src.storage.object.stores import AzureBlobStore

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_azure_blob_connection():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
    blob_store = AzureBlobStore(blob_connection_string, blob_container_name)
    async with blob_store as blob_store:
        assert blob_store.blob_service_client is not None
        assert blob_store.container_client is not None


@pytest.mark.asyncio
async def test_azure_blob_upload_and_delete():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
    blob_store = AzureBlobStore(blob_connection_string, blob_container_name)
    async with blob_store as blob_store:
        await blob_store.upload_file(b"test", "test.txt")

        assert await blob_store.container_client.get_blob_client("test.txt").exists()
        assert await blob_store.delete_file("test.txt")


@pytest.mark.asyncio
async def test_azure_blob_list_files():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
    blob_store = AzureBlobStore(blob_connection_string, blob_container_name)
    async with blob_store as blob_store:
        files = await blob_store.list_files()
        assert len(files) > 0
        assert "AttentionIsAllYouNeed.pdf" in [file.name for file in files]


@pytest.mark.asyncio
async def test_azure_blob_download_files():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
    blob_store = AzureBlobStore(blob_connection_string, blob_container_name)
    async with blob_store as blob_store:
        files = await blob_store.list_files()
        assert len(files) > 0
        # just get first few files
        file_stream = await blob_store.download_file(files[0])
        assert file_stream is not None
        assert file_stream.getvalue() is not None
