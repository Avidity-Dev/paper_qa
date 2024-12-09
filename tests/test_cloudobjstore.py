from io import BytesIO
import os

import dotenv
import pytest

from src.storage.object.stores import AzureBlobAdapter

dotenv.load_dotenv()


@pytest.mark.asyncio
async def test_azure_blob_connection():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_adapter = AzureBlobAdapter(blob_connection_string, "papers")
    await blob_adapter._ensure_client()
    assert blob_adapter.blob_service_client is not None
    assert blob_adapter.container_client is not None


@pytest.mark.asyncio
async def test_azure_blob_upload_and_delete():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_adapter = AzureBlobAdapter(blob_connection_string, "test")
    await blob_adapter._ensure_client()
    await blob_adapter.upload_file(b"test", "test.txt")

    assert await blob_adapter.container_client.get_blob_client("test.txt").exists()
    assert await blob_adapter.delete_file("test.txt")


@pytest.mark.asyncio
async def test_azure_blob_list_files():
    blob_connection_string = os.getenv("AZURE_BLOB_CONNECTION_STRING")
    blob_adapter = AzureBlobAdapter(blob_connection_string, "papers")
    await blob_adapter._ensure_client()
    files = await blob_adapter.list_files()
    assert len(files) > 0
    assert "AttentionIsAllYouNeed.pdf" in files
