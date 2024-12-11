"""
Rate limiter implementation for API requests.
"""

import asyncio
import time
from typing import Optional, Dict

class RateLimiter:
    """
    Async rate limiter using token bucket algorithm.
    Ensures requests don't exceed specified rate limits.
    """
    def __init__(self, requests_per_second: float = 1.0):
        self.rate = requests_per_second
        self.last_check = time.monotonic()
        self._lock = asyncio.Lock()
        self._tokens = 1.0  
        
    async def acquire(self):
        """
        Acquires permission to make a request, waiting if necessary.
        """
        async with self._lock:
            while self._tokens <= 0:
                now = time.monotonic()
                time_passed = now - self.last_check
                self._tokens = min(1.0, self._tokens + time_passed * self.rate)
                self.last_check = now
                
                if self._tokens <= 0:
                    # Wait until next token would be available
                    wait_time = (1.0 - self._tokens) / self.rate
                    await asyncio.sleep(wait_time)
            
            self._tokens -= 1
            self.last_check = time.monotonic()

class APIRateLimiter:
    """
    Manages rate limits for multiple APIs.
    """
    def __init__(self):
        self.limiters: Dict[str, RateLimiter] = {
            'semantic_scholar': RateLimiter(requests_per_second=0.95),  # Slightly under 1 req/sec to stay under the limit
            'crossref': RateLimiter(requests_per_second=0.5)  # Conservative rate since crossref is the most rate limited API
        }
    
    async def acquire(self, api_name: str):
        """
        Acquires permission to make a request to specified API.
        
        Parameters:
        -----------
        api_name : str
            Name of the API to rate limit ('semantic_scholar' or 'crossref')
        """
        if api_name in self.limiters:
            await self.limiters[api_name].acquire()

# Global rate limiter instance
rate_limiter = APIRateLimiter()