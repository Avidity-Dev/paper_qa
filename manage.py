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
from src.process.metadata import pqa_build_mla, pqa_extract_publication_metadata
from src.process.processors import PQAProcessor
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

    def _get_redis_client(self) -> redis.Redis:
        return redis.Redis.from_url(self.app_config.vector_db_url)

    def create_index(
        self, config: Union[Dict[str, Any], os.PathLike[str]] = INDEX_SCHEMA_PATH
    ) -> None:
        redis_client = self._get_redis_client()
        index_builder = RedisIndexBuilder(
            redis_client=redis_client,
            index_name=self.app_config.index_name,
            key_prefix=self.app_config.index_prefix,
            store_type=self.app_config.index_type,
            schema=config,
        )
        index_builder.build_index(recreate_index=False)

    def delete_index(self, index_name: str, drop_documents: bool = True) -> None:
        """Deletes a Redis index and all its documents."""
        redis_client = self._get_redis_client()
        drop_command = "DD" if drop_documents else ""
        try:
            redis_client.execute_command(f"FT.DROPINDEX {index_name} {drop_command}")
            click.echo(
                f"Successfully deleted index '{index_name}'. Drop documents: {drop_documents}"
            )
        except redis.exceptions.ResponseError as e:
            click.echo(f"Error deleting index: {str(e)}. The index may not exist.")

    def clear_documents(self, prefix: str) -> None:
        """Deletes all documents with the given prefix."""
        redis_client = self._get_redis_client()
        try:
            cursor = 0
            while True:
                cursor, keys = redis_client.scan(cursor, match=f"{prefix}*", count=100)
                if keys:
                    redis_client.delete(*keys)
                if cursor == 0:
                    break
            click.echo(f"Successfully cleared all documents with prefix '{prefix}'")
        except Exception as e:
            click.echo(f"Error clearing documents: {str(e)}")

    async def populate_test_data(self) -> None:
        """Retrieves test documents from cloud storage and populates the index."""
        vector_db = RedisVectorStore(
            redis_url=self.app_config.vector_db_url,
            index_name=self.app_config.index_name,
            key_prefix=self.app_config.index_prefix,
            counter_key=self.app_config.counter_key,
            schema=INDEX_SCHEMA_PATH,
        )


@click.group()
def cli():
    """Redis Index Management CLI"""
    pass


@cli.command()
@click.option(
    "--environment",
    default="local",
    help="Environment to use (local, dev, prod)",
)
@click.option(
    "--config_file",
    default=INDEX_SCHEMA_PATH,
    type=click.Path(exists=True),
    help="Path to index config YAML file",
)
def create_index(environment: str, config_file: str):
    """Create a Redis index from a YAML configuration file."""
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    redis_manager = RedisManager(environment=environment)
    redis_manager.create_index(config_data)


@cli.command()
@click.option(
    "--environment",
    default="local",
    help="Environment to use (local, dev, prod)",
)
@click.argument("index", type=click.STRING, required=True)
@click.option("--dd", default=True, is_flag=True, help="Drop documents from index")
def delete_index(environment: str, index: str, dd: bool):
    """Delete a Redis index and all its documents."""
    redis_manager = RedisManager(environment=environment)
    redis_manager.delete_index(index, dd)


@cli.command()
@click.option(
    "--environment",
    default="local",
    help="Environment to use (local, dev, prod)",
)
@click.option("--prefix", required=True, help="Document prefix to clear")
def clear_documents(environment: str, prefix: str):
    """Clear all documents with the given prefix."""
    redis_manager = RedisManager(environment=environment)
    redis_manager.clear_documents(prefix)


if __name__ == "__main__":
    cli()
