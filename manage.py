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

    def get_redis_client(self) -> redis.Redis:
        return redis.Redis.from_url(self.app_config.vector_db_url)

    def create_index(
        self, config: Union[Dict[str, Any], os.PathLike[str]] = INDEX_SCHEMA_PATH
    ) -> None:
        redis_client = self.get_redis_client()
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
        redis_client = self.get_redis_client()
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
        redis_client = self.get_redis_client()
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
    """
    CLI tool for managing Redis search indexes and documents.
    """
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
    """
    Create a new Redis search index using the specified configuration.

    Parameters
    ----------
    environment : str
        The deployment environment (local, dev, prod). Defaults to "local". This
        will be used to load the appropriate configuration from the config.yaml
        file.
    config_file : str
        Path to YAML file containing index schema configuration
        including field definitions, prefixes, and other settings

    Notes
    -----
    The config file should define the index schema including field types,
    names, and any special index configurations.

    Examples
    --------
    $ python manage.py create-index --environment dev --config_file schema.yaml
    """
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    redis_manager = RedisManager(environment=environment)
    redis_manager.create_index(config_data)


@cli.command()
@click.option(
    "--env",
    default="local",
    help="Environment to use (local, dev, prod)",
)
@click.argument("index", type=click.STRING, required=True)
@click.option("--dd", default=True, is_flag=True, help="Drop documents from index")
def delete_index(env: str, index: str, dd: bool):
    """
    Delete a Redis search index and optionally its associated documents.

    Currently defaults to utilizing the YAML config for determine the Redis
    connection parameters, based on the environment.

    Parameters
    ----------
    environment : str
        The deployment environment (local, dev, prod)
    index : str
        Name of the Redis search index to delete
    dd : bool, optional
        If True, delete all documents associated with the index.
        If False, only delete the index structure.
        Default is True.

    Notes
    -----
    This operation cannot be undone. Use with caution in production environments.

    Examples
    --------
    $ python manage.py delete-index --env local my_index --dd
    """
    redis_manager = RedisManager(environment=env)
    redis_manager.delete_index(index, dd)


@cli.command()
@click.option(
    "--env",
    default="local",
    help="Environment to use (local, dev, prod)",
)
@click.argument("prefix", type=click.STRING, required=True)
def clear_documents(env: str, prefix: str):
    """
    Remove all documents with the specified key prefix from Redis.

    Currently defaults to utilizing the YAML config for determine the Redis
    connection parameters, based on the environment.

    Parameters
    ----------
    environment : str
        The deployment environment (local, dev, prod)
    prefix : str
        Key prefix pattern to match documents for deletion.
        All documents whose keys start with this prefix will be removed.

    Notes
    -----
    This operation performs a scan operation and deletes documents in batches
    to prevent memory issues with large document sets.

    Examples
    --------
    $ python manage.py clear-documents --env local --prefix doc:
    """
    redis_manager = RedisManager(environment=env)
    redis_manager.clear_documents(prefix)


if __name__ == "__main__":
    cli()
