import os
from enum import Enum
from typing import Optional, Tuple


from dataclasses import dataclass
from typing import Optional, Dict, Any
import yaml
import os
from functools import lru_cache


# Set defaults for local dev but could change during deployment
VEC_IDX_NAME = os.getenv("REDIS_INDEX_NAME", "idx:docs")
VEC_IDX_PREFIX = os.getenv("REDIS_INDEX_PREFIX", "docs:")


class Environment(Enum):
    LOCAL = "local"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EmbeddingConfig:
    model: str
    dimension: int
    endpoint: str
    provider: str


@dataclass
class LLMConfig:
    model: str
    temperature: float
    max_tokens: int
    top_p: float = 1.0
    endpoint: str
    provider: str


@dataclass
class LiteLLMParams:
    """Configuration for LiteLLM parameters."""

    model: str
    api_key: str
    api_base: str


@dataclass
class AppConfig:
    """Main application configuration."""

    embedding_config: Tuple[str, EmbeddingConfig]
    llm_config: Tuple[str, LLMConfig]
    summary_llm_config: Tuple[str, LLMConfig]
    environment: str


class ConfigurationManager:
    def __init__(
        self,
        app_config_path: str = "src/config/app.yaml",
        static_config_path: str = "src/config/static.yaml",
    ):
        # Uninitialized configs
        self._static_configs: Dict[str, Dict[str, Any]]
        self._app_config: AppConfig
        self._llm_configs: Dict[str, LLMConfig]
        self._embedding_configs: Dict[str, EmbeddingConfig]

        # Config paths
        self._static_config_path = static_config_path or os.getenv(
            "STATIC_CONFIG_PATH", "src/config/static.yaml"
        )
        self._app_config_path = app_config_path or os.getenv(
            "APP_CONFIG_PATH", "src/config/app.yaml"
        )

        self.init_static_configs()
        self.init_app_config()

    def init_static_configs(self) -> None:
        """Load static configuration options from YAML file."""
        with open(self._static_config_path, "r") as f:
            self._static_configs = yaml.safe_load(f)

        self._llm_configs = {
            name: LLMConfig(**config)
            for name, config in self._static_configs["llm_models"].items()
        }
        self._embedding_configs = {
            name: EmbeddingConfig(**config)
            for name, config in self._static_configs["embedding_models"].items()
        }

    def init_app_config(self) -> None:
        """Load application configuration from YAML file."""
        with open(self._app_config_path, "r") as f:
            self._app_config = yaml.safe_load(f)

        default_llm_model = self._app_config["default_llm_model"]
        default_summary_llm_model = self._app_config["default_llm_model"]
        default_embedding_model = self._app_config["default_embedding_model"]

        try:
            self._app_config.llm_config = self._llm_configs[default_llm_model]
            self._app_config.summary_llm_config = self._llm_configs[
                default_summary_llm_model
            ]
            self._app_config.embedding_config = self._embedding_configs[
                default_embedding_model
            ]
        except KeyError as e:
            raise ValueError(
                f"Error during default configuration initialization: {e}"
            ) from e

    def _get_api_key(self, provider: str) -> str:
        """Helper method to retrieve API secrets based on a specific config."""
        if provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")

    def get_paper_qa_settings(self) -> dict:
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
