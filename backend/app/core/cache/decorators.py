"""
Cache Decorators

Provides decorators for easy caching integration with functions and methods.
"""

import functools
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable

from .manager import CacheManager

logger = logging.getLogger(__name__)

# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def set_cache_manager(manager: CacheManager):
    """Set global cache manager for decorators"""
    global _cache_manager
    _cache_manager = manager


def get_cache_manager() -> Optional[CacheManager]:
    """Get global cache manager"""
    return _cache_manager


def cached(key_func: Optional[Callable] = None,
          ttl: Optional[float] = None,
          levels: Optional[List[str]] = None,
          key_prefix: str = "",
          unless: Optional[Callable] = None,
          condition: Optional[Callable] = None):
    """
    Cache decorator for functions and methods

    Args:
        key_func: Function to generate cache key from arguments
        ttl: Time to live in seconds
        levels: Cache levels to use
        key_prefix: Prefix for cache keys
        unless: Function to determine when NOT to cache
        condition: Function to determine when to cache
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if _cache_manager is None:
                logger.warning("No cache manager configured")
                return await func(*args, **kwargs)

            # Check conditions
            if unless and unless(*args, **kwargs):
                return await func(*args, **kwargs)

            if condition and not condition(*args, **kwargs):
                return await func(*args, **kwargs)

            # Generate cache key
            cache_key = _generate_cache_key(func, key_func, key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await _cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            try:
                await _cache_manager.set(cache_key, result, ttl, levels)
            except Exception as e:
                logger.error(f"Failed to cache result for {func.__name__}: {e}")

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if _cache_manager is None:
                logger.warning("No cache manager configured")
                return func(*args, **kwargs)

            # This is a simplified sync wrapper
            # In practice, you might want to use a different approach
            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


def cache_result(key_func: Optional[Callable] = None,
                ttl: Optional[float] = None,
                levels: Optional[List[str]] = None,
                key_prefix: str = "",
                hash_result: bool = False):
    """
    Cache function results with optional result hashing

    Args:
        key_func: Function to generate cache key
        ttl: Time to live
        levels: Cache levels
        key_prefix: Key prefix
        hash_result: Whether to hash the result for storage
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if _cache_manager is None:
                logger.warning("No cache manager configured")
                return await func(*args, **kwargs)

            cache_key = _generate_cache_key(func, key_func, key_prefix, *args, **kwargs)

            # Try to get from cache
            cached_value = await _cache_manager.get(cache_key)
            if cached_value is not None:
                if hash_result:
                    # If result was hashed, this would need unhashing logic
                    pass
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            try:
                if hash_result:
                    import hashlib
                    result_hash = hashlib.sha256(str(result).encode()).hexdigest()
                    await _cache_manager.set(cache_key, result_hash, ttl, levels)
                else:
                    await _cache_manager.set(cache_key, result, ttl, levels)

            except Exception as e:
                logger.error(f"Failed to cache result for {func.__name__}: {e}")

            return result

        return wrapper

    return decorator


def invalidate_cache(key_func: Optional[Callable] = None,
                    key_prefix: str = "",
                    pattern: Optional[str] = None):
    """
    Decorator to invalidate cache after function execution

    Args:
        key_func: Function to generate cache key to invalidate
        key_prefix: Key prefix
        pattern: Pattern to invalidate (alternative to key_func)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            if _cache_manager is None:
                return result

            try:
                if pattern:
                    await _cache_manager.invalidate_pattern(pattern)
                elif key_func:
                    cache_key = _generate_cache_key(func, key_func, key_prefix, *args, **kwargs)
                    await _cache_manager.delete(cache_key)
                else:
                    # Try to generate key from function name and args
                    cache_key = _generate_cache_key(func, None, key_prefix, *args, **kwargs)
                    await _cache_manager.delete(cache_key)

            except Exception as e:
                logger.error(f"Failed to invalidate cache for {func.__name__}: {e}")

            return result

        return wrapper

    return decorator


def cache_bypass(bypass_func: Callable):
    """
    Decorator to bypass cache based on conditions

    Args:
        bypass_func: Function that returns True to bypass cache
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            if bypass_func(*args, **kwargs):
                return await func(*args, **kwargs)

            # Use cached decorator
            cached_decorator = cached()(func)
            return await cached_decorator(*args, **kwargs)

        return wrapper

    return decorator


def memoize(ttl: Optional[float] = None, maxsize: Optional[int] = None):
    """
    Simple memoization decorator

    Args:
        ttl: Time to live
        maxsize: Maximum number of cached items
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_order = []  # For LRU if maxsize is set

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key
            key = _make_hashable_key(args, kwargs)

            # Check cache
            if key in cache:
                if maxsize:
                    # Move to end for LRU
                    cache_order.remove(key)
                    cache_order.append(key)
                return cache[key]

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            if maxsize and len(cache) >= maxsize:
                # Remove oldest item
                oldest_key = cache_order.pop(0)
                del cache[oldest_key]

            cache[key] = result
            if maxsize:
                cache_order.append(key)

            return result

        return wrapper

    return decorator


def _generate_cache_key(func: Callable,
                       key_func: Optional[Callable],
                       key_prefix: str,
                       *args, **kwargs) -> str:
    """Generate cache key for function call"""
    if key_func:
        # Use custom key function
        key_part = key_func(*args, **kwargs)
    else:
        # Generate key from function name and arguments
        key_part = _make_hashable_key(args, kwargs)

    # Include function name and module
    full_key = f"{key_prefix}{func.__module__}.{func.__name__}:{key_part}"
    return full_key


def _make_hashable_key(args: tuple, kwargs: dict) -> str:
    """Create hashable key from arguments"""
    try:
        # Try to create a JSON-serializable key
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()

    except (TypeError, ValueError):
        # Fallback to string representation
        key_str = str(args) + str(sorted(kwargs.items()))
        import hashlib
        return hashlib.md5(key_str.encode()).hexdigest()


class CacheKeyBuilder:
    """Helper class for building cache keys"""

    @staticmethod
    def user_key(user_id: Union[str, int], suffix: str = "") -> str:
        """Build user-specific cache key"""
        return f"user:{user_id}:{suffix}" if suffix else f"user:{user_id}"

    @staticmethod
    def document_key(document_id: Union[str, int], suffix: str = "") -> str:
        """Build document-specific cache key"""
        return f"document:{document_id}:{suffix}" if suffix else f"document:{document_id}"

    @staticmethod
    def query_key(query_hash: str, suffix: str = "") -> str:
        """Build query-specific cache key"""
        return f"query:{query_hash}:{suffix}" if suffix else f"query:{query_hash}"

    @staticmethod
    def session_key(session_id: str, suffix: str = "") -> str:
        """Build session-specific cache key"""
        return f"session:{session_id}:{suffix}" if suffix else f"session:{session_id}"

    @staticmethod
    def search_key(search_params: Dict[str, Any], user_id: Optional[Union[str, int]] = None) -> str:
        """Build search-specific cache key"""
        import hashlib
        params_str = json.dumps(search_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()

        if user_id:
            return f"search:user:{user_id}:{params_hash}"
        return f"search:global:{params_hash}"

    @staticmethod
    def function_key(func_name: str, args: tuple, kwargs: dict) -> str:
        """Build function-specific cache key"""
        import hashlib
        key_data = {
            'args': args,
            'kwargs': sorted(kwargs.items())
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        return f"func:{func_name}:{key_hash}"


# Example usage of cache decorators

@cached(ttl=300, key_prefix="user:")
async def get_user_profile(user_id: int):
    """Example function with caching"""
    # Simulate database call
    await asyncio.sleep(0.1)
    return {"id": user_id, "name": f"User {user_id}"}


@invalidate_cache(pattern="user:*")
async def update_user_profile(user_id: int, data: Dict[str, Any]):
    """Example function that invalidates cache"""
    # Simulate database update
    await asyncio.sleep(0.1)
    return True


@cache_result(ttl=600, key_prefix="search:")
async def search_documents(query: str, filters: Dict[str, Any] = None):
    """Example search function with result caching"""
    # Simulate search operation
    await asyncio.sleep(0.2)
    return {"results": [f"Document for {query}"], "count": 1}


@memoize(ttl=60, maxsize=100)
async def expensive_computation(x: int, y: int) -> int:
    """Example function with memoization"""
    await asyncio.sleep(0.05)
    return x * y