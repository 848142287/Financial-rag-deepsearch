"""
Factory Classes for Data Access Layer

Provides factory methods for creating and configuring data access objects,
repositories, and units of work based on configuration.
"""

from typing import Any, Dict, Optional, Type, Union
import logging
from functools import lru_cache

from .base import (
    BaseDAO,
    BaseRepository,
    UnitOfWork,
    DatabaseConfig,
    DataSource
)
from .repositories import (
    DocumentRepository,
    VectorRepository,
    KnowledgeGraphRepository,
    UserRepository,
    TaskRepository
)
from .daos import (
    MySQLDAO,
    MilvusDAO,
    Neo4jDAO,
    RedisDAO,
    MinioDAO
)
from .unit_of_work import (
    DataUnitOfWork,
    DocumentUnitOfWork,
    RAGUnitOfWork
)

logger = logging.getLogger(__name__)


class DAOFactory:
    """
    Factory for creating DAO instances
    """

    _dao_classes = {
        DataSource.MYSQL: MySQLDAO,
        DataSource.MILVUS: MilvusDAO,
        DataSource.NEO4J: Neo4jDAO,
        DataSource.REDIS: RedisDAO,
        DataSource.MINIO: MinioDAO
    }

    @classmethod
    def create_dao(cls, source: DataSource, config: DatabaseConfig) -> BaseDAO:
        """
        Create a DAO instance for the specified data source

        Args:
            source: The data source type
            config: Database configuration

        Returns:
            Configured DAO instance
        """
        if source not in cls._dao_classes:
            raise ValueError(f"Unsupported data source: {source}")

        dao_class = cls._dao_classes[source]
        return dao_class(config)

    @classmethod
    def register_dao(cls, source: DataSource, dao_class: Type[BaseDAO]) -> None:
        """
        Register a new DAO class for a data source

        Args:
            source: Data source type
            dao_class: DAO class to register
        """
        cls._dao_classes[source] = dao_class


class RepositoryFactory:
    """
    Factory for creating repository instances
    """

    @staticmethod
    def create_document_repository(mysql_dao: MySQLDAO,
                                 milvus_dao: MilvusDAO,
                                 minio_dao: MinioDAO,
                                 neo4j_dao: Neo4jDAO) -> DocumentRepository:
        """
        Create document repository

        Args:
            mysql_dao: MySQL DAO for metadata
            milvus_dao: Milvus DAO for vectors
            minio_dao: MinIO DAO for files
            neo4j_dao: Neo4j DAO for relationships

        Returns:
            Configured document repository
        """
        return DocumentRepository(
            mysql_dao=mysql_dao,
            milvus_dao=milvus_dao,
            minio_dao=minio_dao,
            neo4j_dao=neo4j_dao
        )

    @staticmethod
    def create_vector_repository(milvus_dao: MilvusDAO,
                               document_repository: DocumentRepository) -> VectorRepository:
        """
        Create vector repository

        Args:
            milvus_dao: Milvus DAO
            document_repository: Document repository for enrichment

        Returns:
            Configured vector repository
        """
        return VectorRepository(
            milvus_dao=milvus_dao,
            document_repository=document_repository
        )

    @staticmethod
    def create_knowledge_graph_repository(neo4j_dao: Neo4jDAO) -> KnowledgeGraphRepository:
        """
        Create knowledge graph repository

        Args:
            neo4j_dao: Neo4j DAO

        Returns:
            Configured knowledge graph repository
        """
        return KnowledgeGraphRepository(neo4j_dao=neo4j_dao)

    @staticmethod
    def create_user_repository(mysql_dao: MySQLDAO, redis_dao: RedisDAO) -> UserRepository:
        """
        Create user repository

        Args:
            mysql_dao: MySQL DAO for user data
            redis_dao: Redis DAO for caching

        Returns:
            Configured user repository
        """
        return UserRepository(
            mysql_dao=mysql_dao,
            redis_dao=redis_dao
        )

    @staticmethod
    def create_task_repository(mysql_dao: MySQLDAO, redis_dao: RedisDAO) -> TaskRepository:
        """
        Create task repository

        Args:
            mysql_dao: MySQL DAO for task persistence
            redis_dao: Redis DAO for task queue

        Returns:
            Configured task repository
        """
        return TaskRepository(
            mysql_dao=mysql_dao,
            redis_dao=redis_dao
        )


class UnitOfWorkFactory:
    """
    Factory for creating Unit of Work instances
    """

    @staticmethod
    def create_data_uow(configs: Dict[DataSource, DatabaseConfig]) -> DataUnitOfWork:
        """
        Create generic data Unit of Work

        Args:
            configs: Database configurations by data source

        Returns:
            Configured data Unit of Work
        """
        return DataUnitOfWork(configs)

    @staticmethod
    def create_document_uow(mysql_config: DatabaseConfig,
                          milvus_config: DatabaseConfig,
                          minio_config: DatabaseConfig,
                          neo4j_config: DatabaseConfig) -> DocumentUnitOfWork:
        """
        Create document-specific Unit of Work

        Args:
            mysql_config: MySQL configuration
            milvus_config: Milvus configuration
            minio_config: MinIO configuration
            neo4j_config: Neo4j configuration

        Returns:
            Configured document Unit of Work
        """
        return DocumentUnitOfWork(
            mysql_config=mysql_config,
            milvus_config=milvus_config,
            minio_config=minio_config,
            neo4j_config=neo4j_config
        )

    @staticmethod
    def create_rag_uow(configs: Dict[DataSource, DatabaseConfig]) -> RAGUnitOfWork:
        """
        Create RAG-specific Unit of Work

        Args:
            configs: Database configurations by data source

        Returns:
            Configured RAG Unit of Work
        """
        return RAGUnitOfWork(configs)


class DataAccessManager:
    """
    High-level manager for data access operations

    Provides a simplified interface for working with the data access layer,
    managing connections, and creating appropriate objects based on context.
    """

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize data access manager

        Args:
            config_dict: Configuration dictionary with database settings
        """
        self.config_dict = config_dict
        self._database_configs = self._parse_configs(config_dict)
        self._dao_cache: Dict[str, BaseDAO] = {}
        self._repository_cache: Dict[str, BaseRepository] = {}

    def _parse_configs(self, config_dict: Dict[str, Any]) -> Dict[DataSource, DatabaseConfig]:
        """Parse configuration dictionary into DatabaseConfig objects"""
        configs = {}

        # MySQL configuration
        if 'mysql' in config_dict:
            mysql_cfg = config_dict['mysql']
            configs[DataSource.MYSQL] = DatabaseConfig(
                source=DataSource.MYSQL,
                host=mysql_cfg.get('host', 'localhost'),
                port=mysql_cfg.get('port', 3306),
                database=mysql_cfg.get('database'),
                username=mysql_cfg.get('username'),
                password=mysql_cfg.get('password'),
                connection_params=mysql_cfg.get('connection_params', {})
            )

        # Milvus configuration
        if 'milvus' in config_dict:
            milvus_cfg = config_dict['milvus']
            configs[DataSource.MILVUS] = DatabaseConfig(
                source=DataSource.MILVUS,
                host=milvus_cfg.get('host', 'localhost'),
                port=milvus_cfg.get('port', 19530),
                database=milvus_cfg.get('collection'),
                connection_params=milvus_cfg.get('connection_params', {})
            )

        # Neo4j configuration
        if 'neo4j' in config_dict:
            neo4j_cfg = config_dict['neo4j']
            configs[DataSource.NEO4J] = DatabaseConfig(
                source=DataSource.NEO4J,
                host=neo4j_cfg.get('uri', 'bolt://localhost:7687'),
                port=0,  # Port is in URI
                database=neo4j_cfg.get('database'),
                username=neo4j_cfg.get('user', 'neo4j'),
                password=neo4j_cfg.get('password'),
                connection_params=neo4j_cfg.get('connection_params', {})
            )

        # Redis configuration
        if 'redis' in config_dict:
            redis_cfg = config_dict['redis']
            configs[DataSource.REDIS] = DatabaseConfig(
                source=DataSource.REDIS,
                host=redis_cfg.get('host', 'localhost'),
                port=redis_cfg.get('port', 6379),
                database=redis_cfg.get('db', 0),
                username=redis_cfg.get('username'),
                password=redis_cfg.get('password'),
                connection_params=redis_cfg.get('connection_params', {})
            )

        # MinIO configuration
        if 'minio' in config_dict:
            minio_cfg = config_dict['minio']
            configs[DataSource.MINIO] = DatabaseConfig(
                source=DataSource.MINIO,
                host=minio_cfg.get('endpoint', 'localhost:9000'),
                port=0,  # Port is in endpoint
                database=minio_cfg.get('bucket'),
                username=minio_cfg.get('access_key'),
                password=minio_cfg.get('secret_key'),
                connection_params=minio_cfg.get('connection_params', {})
            )

        return configs

    @lru_cache(maxsize=10)
    def get_dao(self, source: DataSource) -> BaseDAO:
        """
        Get DAO instance (cached)

        Args:
            source: Data source type

        Returns:
            DAO instance
        """
        cache_key = source.value

        if cache_key not in self._dao_cache:
            config = self._database_configs.get(source)
            if not config:
                raise ValueError(f"Configuration not found for {source}")

            self._dao_cache[cache_key] = DAOFactory.create_dao(source, config)

        return self._dao_cache[cache_key]

    def get_repository(self, repository_type: str) -> BaseRepository:
        """
        Get repository instance

        Args:
            repository_type: Type of repository

        Returns:
            Repository instance
        """
        if repository_type in self._repository_cache:
            return self._repository_cache[repository_type]

        # Create repository based on type
        if repository_type == "document":
            repository = RepositoryFactory.create_document_repository(
                mysql_dao=self.get_dao(DataSource.MYSQL),
                milvus_dao=self.get_dao(DataSource.MILVUS),
                minio_dao=self.get_dao(DataSource.MINIO),
                neo4j_dao=self.get_dao(DataSource.NEO4J)
            )
        elif repository_type == "vector":
            doc_repo = self.get_repository("document")
            repository = RepositoryFactory.create_vector_repository(
                milvus_dao=self.get_dao(DataSource.MILVUS),
                document_repository=doc_repo
            )
        elif repository_type == "knowledge_graph":
            repository = RepositoryFactory.create_knowledge_graph_repository(
                neo4j_dao=self.get_dao(DataSource.NEO4J)
            )
        elif repository_type == "user":
            repository = RepositoryFactory.create_user_repository(
                mysql_dao=self.get_dao(DataSource.MYSQL),
                redis_dao=self.get_dao(DataSource.REDIS)
            )
        elif repository_type == "task":
            repository = RepositoryFactory.create_task_repository(
                mysql_dao=self.get_dao(DataSource.MYSQL),
                redis_dao=self.get_dao(DataSource.REDIS)
            )
        else:
            raise ValueError(f"Unknown repository type: {repository_type}")

        self._repository_cache[repository_type] = repository
        return repository

    def create_unit_of_work(self, uow_type: str = "data") -> UnitOfWork:
        """
        Create Unit of Work instance

        Args:
            uow_type: Type of Unit of Work ("data", "document", "rag")

        Returns:
            Unit of Work instance
        """
        if uow_type == "data":
            return UnitOfWorkFactory.create_data_uow(self._database_configs)
        elif uow_type == "document":
            return UnitOfWorkFactory.create_document_uow(
                mysql_config=self._database_configs[DataSource.MYSQL],
                milvus_config=self._database_configs[DataSource.MILVUS],
                minio_config=self._database_configs[DataSource.MINIO],
                neo4j_config=self._database_configs[DataSource.NEO4J]
            )
        elif uow_type == "rag":
            return UnitOfWorkFactory.create_rag_uow(self._database_configs)
        else:
            raise ValueError(f"Unknown Unit of Work type: {uow_type}")

    async def initialize_all(self):
        """Initialize all connections"""
        for dao in self._dao_cache.values():
            await dao.connect()

    async def close_all(self):
        """Close all connections"""
        for dao in self._dao_cache.values():
            await dao.disconnect()

    @property
    def configs(self) -> Dict[DataSource, DatabaseConfig]:
        """Get all database configurations"""
        return self._database_configs.copy()


# Global instance for application-wide use
_data_access_manager: Optional[DataAccessManager] = None


def get_data_access_manager() -> Optional[DataAccessManager]:
    """Get global data access manager instance"""
    return _data_access_manager


def initialize_data_access(config_dict: Dict[str, Any]) -> DataAccessManager:
    """
    Initialize global data access manager

    Args:
        config_dict: Configuration dictionary

    Returns:
        Initialized data access manager
    """
    global _data_access_manager
    _data_access_manager = DataAccessManager(config_dict)
    return _data_access_manager


def create_dao(source: DataSource) -> BaseDAO:
    """Convenience function to create DAO"""
    manager = get_data_access_manager()
    if not manager:
        raise RuntimeError("Data access manager not initialized")

    return manager.get_dao(source)


def create_repository(repository_type: str) -> BaseRepository:
    """Convenience function to create repository"""
    manager = get_data_access_manager()
    if not manager:
        raise RuntimeError("Data access manager not initialized")

    return manager.get_repository(repository_type)


def create_unit_of_work(uow_type: str = "data") -> UnitOfWork:
    """Convenience function to create Unit of Work"""
    manager = get_data_access_manager()
    if not manager:
        raise RuntimeError("Data access manager not initialized")

    return manager.create_unit_of_work(uow_type)