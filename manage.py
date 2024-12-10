"""
Commands for application environment management.
"""

import os
from typing import Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
import yaml
import click
from langchain_community.vectorstores import Redis
from langchain_community.vectorstores.redis.schema import read_schema
import redis

from src.config.config import ConfigurationManager

# TODO: Create a function to generate this command from config and be able to run from
# from the command line.
FT_CREATE_COMMAND = """
FT.CREATE idx:docs_vss ON JSON PREFIX 1 docs: SCHEMA $.id AS id TEXT NOSTEM $.title AS title TEXT WEIGHT 1.0 $.authors AS authors TAG SEPARATOR "|" $.doi AS doi TEXT NOSTEM $.published_date AS published_date NUMERIC SORTABLE $.created_at AS created_at NUMERIC SORTABLE $.text AS text TEXT WEIGHT 1.0 $.embedding AS vector VECTOR FLAT 6 TYPE FLOAT32 DIM 1536 DISTANCE_METRIC COSINE
"""


class RedisManager:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        index_name: str = "idx:docs",
        prefix: str = "docs:",
        schema: list[Field] = INDEX_SCHEMA,
        app_config_path: str = "src/config/app.yaml",
        static_config_path: str = "src/config/static.yaml",
    ):
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.config_manager = ConfigurationManager(
            app_config_path=app_config_path, static_config_path=static_config_path
        )

    def create_index(self, config: Union[Dict[str, Any], os.PathLike[str]]) -> None:
        try:
            self.client.execute_command(create_cmd)
            click.echo(f"Successfully created index '{index_name}'")
        except redis.exceptions.ResponseError as e:
            if "Index already exists" in str(e):
                click.echo(f"Index '{index_name}' already exists")
            else:
                raise e

    def delete_index(self, index_name: str) -> None:
        """Deletes a Redis index and all its documents."""
        try:
            self.client.execute_command(f"FT.DROPINDEX {index_name}")
            click.echo(f"Successfully deleted index '{index_name}'")
        except redis.exceptions.ResponseError as e:
            click.echo(f"Error deleting index: {str(e)}")

    def clear_documents(self, prefix: str) -> None:
        """Deletes all documents with the given prefix."""
        try:
            cursor = 0
            while True:
                cursor, keys = self.client.scan(cursor, match=f"{prefix}*", count=100)
                if keys:
                    self.client.delete(*keys)
                if cursor == 0:
                    break
            click.echo(f"Successfully cleared all documents with prefix '{prefix}'")
        except Exception as e:
            click.echo(f"Error clearing documents: {str(e)}")

    # TODO: Complete this function to fully populate the index with embeddings and metadata
    def populate_index_local_pdfs(self, index_name: str, dir_path: str) -> None:
        """Populates a Redis index from a directory of documents

        Notes
        -----
        Currently defaults to embedding with OpenAI's text-embedding-3-small model.
        """
        # Get all the pdfs in the test directory
        pdfs = [f for f in os.listdir(dir_path) if f.endswith(".pdf")]
        pdf_bytes = []

        # Read each pdf into a byte stream
        for pdf in pdfs:
            with open(os.path.join(dir_path, pdf), "rb") as f:
                pdf_bytes.append(f.read())

        # Embed the documents
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        doc_embeddings = embeddings.embed_documents(pdf_bytes)

    def populate_index_from_cloud_pdfs(self, index_name: str, dir_path: str) -> None:
        pass


@click.group()
def cli():
    """Redis Index Management CLI"""
    pass


@cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option(
    "--config_file",
    required=True,
    type=click.Path(exists=True),
    help="Path to index config YAML file",
)
def create_index(host: str, port: int, config: str):
    """Create a Redis index from a YAML configuration file."""
    with open(config, "r") as f:
        config_data = yaml.safe_load(f)

    redis_manager = RedisManager(host=host, port=port)
    redis_manager.create_index(config_data)


@cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--index", required=True, help="Name of the index to delete")
@click.option("--prefix", required=True, help="Document prefix to clear")
def delete_index_and_documents(host: str, port: int, index: str, prefix: str):
    """Delete a Redis index and all its documents."""
    redis_manager = RedisManager(host=host, port=port)
    redis_manager.delete_index(index)
    redis_manager.clear_documents(prefix)


@cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--prefix", required=True, help="Document prefix to clear")
def clear_documents(host: str, port: int, prefix: str):
    """Clear all documents with the given prefix."""
    redis_manager = RedisManager(host=host, port=port)
    redis_manager.clear_documents(prefix)


if __name__ == "__main__":
    cli()
