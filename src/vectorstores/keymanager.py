"""
Key management for the vector databases.

This module provides abstract and concrete implementations for managing
incremental keys in vector databases.
"""

from abc import ABC, abstractmethod
from typing import Optional

from redis import Redis


class KeyManager(ABC):
    """Abstract base class for key management.

    Provides an interface for managing incremental keys in vector databases.
    """

    @abstractmethod
    def initialize_counter(self) -> None:
        """Initialize the counter if it doesn't exist.

        Creates and initializes a counter in the database if one does not
        already exist.
        """
        pass

    @abstractmethod
    def get_next_key(self) -> str:
        """Get the next available key.

        Returns
        -------
        str
            The next formatted key string.
        """
        pass

    @abstractmethod
    def get_current_counter(self) -> int:
        """Get the current counter value.

        Returns
        -------
        int
            The current counter value.
        """
        pass

    @abstractmethod
    def reset_counter(self, value: int = 0) -> None:
        """Reset the counter to a specific value.

        Parameters
        ----------
        value : int, optional
            Value to set the counter to, by default 0
        """
        pass

    @abstractmethod
    def generate_batch_keys(self, batch_size: int) -> list[str]:
        """Generate multiple keys at once for batch processing.

        Parameters
        ----------
        batch_size : int
            Number of keys to generate

        Returns
        -------
        list[str]
            List of formatted key strings
        """
        pass


class RedisKeyManager:
    """Manages incremental key generation for Redis.

    A concrete implementation of key management for Redis databases.

    Parameters
    ----------
    redis_client : Redis
        Redis client instance
    key_prefix : str
        Prefix for embedding chunk keys
    counter_key : str
        Redis key to store the current counter value
    """

    def __init__(
        self,
        redis_client: Redis,
        key_prefix: str,
        counter_key: str,
        key_padding: int = 4,
    ):
        """Initialize the key manager.

        Parameters
        ----------
        redis_client : Redis
            Redis client instance
        key_prefix : str
            Prefix for embedding chunk keys
        counter_key : str
            Redis key to store the current counter value
        key_padding : int, optional
            Number of digits to pad the key with, by default 4
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.counter_key = counter_key
        self.key_padding = key_padding

        self._init()

    def _init(self) -> None:
        """Initialize the counter if it doesn't exist.

        Creates a counter in Redis with value 1 if it doesn't already exist.
        Uses Redis SETNX for atomic initialization.
        """
        if not self.redis.exists(self.counter_key):
            self.redis.setnx(self.counter_key, 1)

    def get_next_key(self) -> str:
        """Get the next available key using atomic increment.

        Returns
        -------
        str
            Formatted key string (e.g., 'doc_chunk:0001')

        Notes
        -----
        Uses Redis INCR for atomic increment operation.
        The numeric portion is zero-padded based on key_padding value.
        """
        next_id = self.redis.incr(self.counter_key)
        return f"{self.key_prefix}:{next_id:0{self.key_padding}d}"

    def get_current_counter(self) -> int:
        """Get the current counter value without incrementing.

        Returns
        -------
        int
            Current counter value

        Notes
        -----
        Returns 0 if counter doesn't exist.
        """
        counter = self.redis.get(self.counter_key)
        return int(counter) if counter else 0

    def reset_counter(self, value: int = 0) -> None:
        """Reset the counter to a specific value.

        Parameters
        ----------
        value : int, optional
            Value to set the counter to, by default 0

        Notes
        -----
        Uses Redis SET operation to update the counter value.
        """
        self.redis.set(self.counter_key, value)

    def generate_batch_keys(self, batch_size: int) -> list[str]:
        """Generate multiple keys at once for batch processing.

        Parameters
        ----------
        batch_size : int
            Number of keys to generate

        Returns
        -------
        list[str]
            List of formatted key strings, zero-padded based on key_padding value

        Notes
        -----
        Uses Redis INCRBY for atomic batch increment operation.
        Keys are generated sequentially starting from the next available ID.
        """
        new_counter = self.redis.incrby(self.counter_key, batch_size)
        start_id = new_counter - batch_size + 1

        return [
            f"{self.key_prefix}:{i:0{self.key_padding}d}"
            for i in range(start_id, start_id + batch_size)
        ]
