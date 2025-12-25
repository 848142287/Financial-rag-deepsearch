"""
Unified Configuration Management

Provides centralized configuration management with support for:
- Environment-specific configurations
- Configuration validation
- Dynamic configuration updates
"""

from .manager import (
    ConfigManager,
    get_config_manager,
    initialize_config
)

from .database import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
    MILVUS_HOST,
    MILVUS_PORT,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_DB
)

__all__ = [
    # Core
    "ConfigManager",
    "get_config_manager",
    "initialize_config",

    # Database
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
    "MILVUS_HOST",
    "MILVUS_PORT",
    "REDIS_HOST",
    "REDIS_PORT",
    "REDIS_DB"
]