"""
数据库配置文件
统一管理所有数据库连接配置
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DatabaseConfig:
    """数据库配置类"""

    # MySQL配置
    mysql_host: str = "mysql"
    mysql_port: int = 3306
    mysql_user: str = "rag_user"
    mysql_password: str = "rag_pass"
    mysql_database: str = "financial_rag"

    # MongoDB配置
    mongodb_host: str = "mongodb"
    mongodb_port: int = 27017
    mongodb_user: str = "admin"
    mongodb_password: str = "password"
    mongodb_database: str = "financial_rag"

    # Neo4j配置
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j123"
    neo4j_database: str = "neo4j"

    # Milvus配置
    milvus_host: str = "milvus"
    milvus_port: int = 19530
    milvus_collection: str = "document_embeddings"

    # Redis配置
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_database: int = 0

    # MinIO配置
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "financial-rag-documents"

# 创建全局配置实例
db_config = DatabaseConfig()

def get_database_config() -> DatabaseConfig:
    """获取数据库配置"""
    return db_config

def get_mysql_connection_string():
    """获取MySQL连接字符串"""
    return f"mysql+pymysql://{db_config.mysql_user}:{db_config.mysql_password}@{db_config.mysql_host}:{db_config.mysql_port}/{db_config.mysql_database}"

def get_mongodb_connection_string():
    """获取MongoDB连接字符串"""
    return f"mongodb://{db_config.mongodb_user}:{db_config.mongodb_password}@{db_config.mongodb_host}:{db_config.mongodb_port}/{db_config.mongodb_database}"

def get_neo4j_connection_string():
    """获取Neo4j连接字符串"""
    return db_config.neo4j_uri

def get_milvus_connection_params():
    """获取Milvus连接参数"""
    return {
        'host': db_config.milvus_host,
        'port': db_config.milvus_port
    }

def get_redis_connection_string():
    """获取Redis连接字符串"""
    if db_config.redis_password:
        return f"redis://:{db_config.redis_password}@{db_config.redis_host}:{db_config.redis_port}/{db_config.redis_database}"
    else:
        return f"redis://{db_config.redis_host}:{db_config.redis_port}/{db_config.redis_database}"


# 导出常量，供其他模块使用
MYSQL_HOST = db_config.mysql_host
MYSQL_PORT = db_config.mysql_port
MYSQL_USER = db_config.mysql_user
MYSQL_PASSWORD = db_config.mysql_password
MYSQL_DATABASE = db_config.mysql_database

MILVUS_HOST = db_config.milvus_host
MILVUS_PORT = db_config.milvus_port

REDIS_HOST = db_config.redis_host
REDIS_PORT = db_config.redis_port
REDIS_DB = db_config.redis_database