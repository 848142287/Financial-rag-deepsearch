"""
缓存模块初始化
"""

from .cache_manager import CacheManager, cache_manager

# 暂时提供空的装饰器和函数
def cached(*args, **kwargs):
    """临时缓存装饰器"""
    def decorator(func):
        return func
    return decorator

def cache_rag_result(*args, **kwargs):
    """临时RAG缓存装饰器"""
    def decorator(func):
        return func
    return decorator

def get_cached_rag_result(*args, **kwargs):
    """临时获取缓存的RAG结果"""
    return None

__all__ = [
    'cache_manager',
    'cached',
    'cache_rag_result',
    'get_cached_rag_result'
]