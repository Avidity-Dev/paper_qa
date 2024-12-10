import os
from enum import Enum
from typing import Optional, Tuple


from dataclasses import dataclass, Field
from typing import Optional, Dict, Any
import yaml
import os
from functools import lru_cache

from redis.commands.search.field import NumericField, TagField, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType

# Set defaults for local dev but could change during deployment
INDEX_SCHEMA_PATH = os.getenv("REDIS_VECTOR_CONFIG_PATH")
APP_CONFIG_PATH = os.getenv("APP_CONFIG_PATH")
STATIC_CONFIG_PATH = os.getenv("STATIC_CONFIG_PATH")


class Environment(Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EmbeddingConfig:
    name: str
    model: str
    dimension: int
    endpoint: str
    provider: str


@dataclass
class LLMConfig:
    name: str
    model: str
    temperature: float
    max_tokens: int
    endpoint: str
    provider: str
    top_p: float = 1.0


@dataclass
class LiteLLMParams:
    """Configuration for LiteLLM parameters."""

    model: str
    api_key: str
    api_base: str


@dataclass
class AppConfig:
    """Main application configuration

    Attributes
    ----------
    vector_db_url: str
        URL of the Redis vector database
    index_name: str
        Name of the Redis index
    index_prefix: str
        Prefix used for records in the Redis index
    counter_name: str
        Name of the key used to store next available record id
    embedding_config: Tuple[str, EmbeddingConfig]
        Embedding model configuration
    llm_config: Tuple[str, LLMConfig]
        LLM model configuration
    summary_llm_config: Tuple[str, LLMConfig]
        Summary LLM model configuration
    environment: str
        Environment used to initialize the configuration
    """

    vector_db_url: str
    index_name: str
    index_prefix: str
    counter_name: str
    embedding_config: Tuple[str, EmbeddingConfig]
    llm_config: Tuple[str, LLMConfig]
    summary_llm_config: Tuple[str, LLMConfig]
    environment: str


class ConfigurationManager:
    """Configuration manager for the application."""

    def __init__(
        self,
        app_config_path: str = APP_CONFIG_PATH,
        index_schema_path: str = INDEX_SCHEMA_PATH,
        static_config_path: str = STATIC_CONFIG_PATH,
    ):
        # Uninitialized configs
        self._llm_configs: Dict[str, LLMConfig]
        self._embedding_configs: Dict[str, EmbeddingConfig]

        # Store the paths used to initialize the configs
        self._init_app_config_fp = app_config_path
        self._init_index_schema_fp = index_schema_path

        # We can load static configs without an app config
        self.init_static_configs(static_config_path)

    @property
    def app_config(self) -> AppConfig:
        """Get the current application configuration."""
        return self._app_config

    @app_config.setter
    def app_config(self, app_config: AppConfig) -> None:
        """Set the current application configuration."""
        self._app_config = app_config

    @property
    def embedding_config(self) -> Dict[str, EmbeddingConfig]:
        """Get the embedding configuration.

        Returns
        -------
        Dict[str, EmbeddingConfig]
            Model short name, Embedding config dictionary
        """
        return self._app_config.embedding_config

    @property
    def llm_config(self) -> Dict[str, LLMConfig]:
        """Get the LLM configuration."""
        return self._app_config.llm_config

    @property
    def summary_llm_config(self) -> Dict[str, LLMConfig]:
        """Get the summary LLM configuration."""
        return self._app_config.summary_llm_config

    def init_static_configs(self, static_config_path: str) -> None:
        """Load static configuration options from YAML file."""
        with open(static_config_path, "r") as f:
            self._static_configs = yaml.safe_load(f)

        self._llm_configs = {
            name: LLMConfig(**config)
            for name, config in self._static_configs["llm_models"].items()
        }
        self._embedding_configs = {
            name: EmbeddingConfig(**config)
            for name, config in self._static_configs["embedding_models"].items()
        }

    def init_app_config(self, environment: str = "local") -> AppConfig:
        """Load application configuration from YAML file."""
        with open(self._app_config_path, "r") as f:
            self._app_config = yaml.safe_load(f)

        llm_model = self._app_config[environment]["llm_model"]
        summary_llm_model = self._app_config[environment]["summary_llm_model"]
        embedding_model = self._app_config[environment]["embedding_model"]

        try:
            self._app_config.llm_config = self._llm_configs[llm_model]
            self._app_config.summary_llm_config = self._llm_configs[summary_llm_model]
            self._app_config.embedding_config = self._embedding_configs[embedding_model]
        except KeyError as e:
            raise ValueError(
                f"Error during default configuration initialization: {e}"
            ) from e

        return self.app_config

    def _get_api_key(self, provider: str) -> str:
        """Helper method to retrieve API secrets based on a specific config."""
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")

    def get_pqa_settings_dict(self) -> dict:
        """Build dictionary to be used for unpacking into Settings object for paper-qa."""
        return {
            "llm": self._app_config.summary_llm_config.model,
            "summary_llm": self._app_config.summary_llm_config.model,
            "llm_config": {
                "model_list": [
                    {
                        "model_name": self._app_config.llm_config.model,
                        "litellm_params": {
                            "model": self._app_config.llm_config.model,
                            "api_key": self._get_api_key(
                                self._app_config.llm_config.provider
                            ),
                            "api_base": self._app_config.llm_config.endpoint,
                        },
                    }
                ]
            },
            "summary_llm_config": {
                "model": self._app_config.summary_llm_config.model,
                "litellm_params": {
                    "model": self._app_config.summary_llm_config.model,
                    "api_key": self._get_api_key(
                        self._app_config.summary_llm_config.provider
                    ),
                    "api_base": self._app_config.summary_llm_config.endpoint,
                },
            },
            "embedding": self._app_config.embedding_config.model,
        }


# class Config:
#     """Configuration management for both local and Streamlit deployment"""

#     # Load .env file for local development
#     load_dotenv()

#     def __init__(self):
#         """Initialize configuration values"""
#         # API Keys
#         self.ANTHROPIC_API_KEY = self._get_env_var("ANTHROPIC_API_KEY")

#     def _get_env_var(self, key: str, default: Optional[str] = None) -> str:
#         """Get an environment variable with optional Streamlit secrets fallback."""
#         value = None
#         try:
#             # Try getting from Streamlit secrets first
#             import streamlit as st

#             value = st.secrets.get(key)
#         except ImportError:
#             pass  # Streamlit not installed or not in Streamlit environment

#         # Fall back to environment variables
#         if value is None:
#             value = os.getenv(key, default)

#         # Raise an error for required keys with no value
#         if value is None and default is None:
#             raise ValueError(
#                 f"Required configuration '{key}' not found in environment or secrets."
#             )

#         return value


# # Create and export the singleton instance
# config = Config()

# # Export the instance for other modules to import
# __all__ = ["config"]
