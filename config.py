import os
from dotenv import load_dotenv
from typing import Optional


class Config:
    """Configuration management for both local and Streamlit deployment"""

    # Load .env file for local development
    load_dotenv()

    def __init__(self):
        """Initialize configuration values"""
        # API Keys
        self.ANTHROPIC_API_KEY = self._get_env_var("ANTHROPIC_API_KEY")

    def _get_env_var(self, key: str, default: Optional[str] = None) -> str:
        """Get an environment variable with optional Streamlit secrets fallback."""
        value = None
        try:
            # Try getting from Streamlit secrets first
            import streamlit as st

            value = st.secrets.get(key)
        except ImportError:
            pass  # Streamlit not installed or not in Streamlit environment

        # Fall back to environment variables
        if value is None:
            value = os.getenv(key, default)

        # Raise an error for required keys with no value
        if value is None and default is None:
            raise ValueError(
                f"Required configuration '{key}' not found in environment or secrets."
            )

        return value


# Create and export the singleton instance
config = Config()

# Export the instance for other modules to import
__all__ = ["config"]
