"""
Commands for application environment management.
"""

import os
from typing import Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
import yaml

import click
from dotenv import load_dotenv
import redis


from src.config.config import (
    ConfigurationManager,
    AppConfig,
    INDEX_SCHEMA_PATH,
    APP_CONFIG_PATH,
    STATIC_CONFIG_PATH,
)
from src.storage.vector.stores import RedisVectorStore, RedisIndexBuilder

load_dotenv()


class RedisManager:
    def __init__(
        self,
        config_manager: ConfigurationManager = ConfigurationManager(),
        environment: str = "local",
    ):
        self.config_manager = config_manager
        self.app_config: AppConfig = config_manager.init_app_config(
            environment=environment
        )

    def create_index(
        self, config: Union[Dict[str, Any], os.PathLike[str]] = INDEX_SCHEMA_PATH
    ) -> None:
        redis_client = redis.Redis.from_url(self.app_config.vector_db_url)
        index_builder = RedisIndexBuilder(
            redis_client=redis_client,
            index_name=self.app_config.vector_db_index_name,
            key_prefix=self.app_config.vector_db_index_prefix,
            store_type="redis",
            schema=config,
        )
        index_builder.build_index()

    def delete_index(self, index_name: str) -> None:
        """Deletes a Redis index and all its documents."""
        try:
            self.client.execute_command(f"FT.DROPINDEX {index_name} DD")
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

    def populate_test_data(self) -> None:
        """Retrieves test documents from cloud storage and populates the index."""


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
