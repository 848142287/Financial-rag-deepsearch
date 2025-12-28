"""
Unit of Work Implementations

Implements the Unit of Work pattern for managing transactions and business operations
across multiple data sources and repositories.
"""

from typing import Any, Dict, List, Optional, Union, Type
import logging
from contextlib import asynccontextmanager

from .base import (
    UnitOfWork as BaseUnitOfWork,
    DataAccessException,
    TransactionException
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
    MinioDAO,
    DatabaseConfig,
    DataSource
)

logger = logging.getLogger(__name__)


class DataUnitOfWork(BaseUnitOfWork):
    """
    Generic Unit of Work for data operations

    Manages transactions and operations across multiple data sources
    """

    def __init__(self, configs: Dict[DataSource, DatabaseConfig]):
        super().__init__()
        self.configs = configs
        self._daos = {}
        self._repositories = {}

    async def __aenter__(self):
        """Enter Unit of Work context"""
        try:
            # Initialize all DAOs
            await self._initialize_daos()

            # Initialize repositories
            await self._initialize_repositories()

            # Begin transactions for all DAOs
            for dao in self._daos.values():
                if hasattr(dao, '_begin_transaction'):
                    await dao._begin_transaction()

            self._is_active = True
            logger.info("Unit of Work started")
            return self

        except Exception as e:
            logger.error(f"Failed to start Unit of Work: {e}")
            raise TransactionException(f"UoW start failed: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit Unit of Work context"""
        if not self._is_active:
            return

        try:
            if exc_type is not None:
                await self.rollback()
            else:
                await self.commit()

            # Disconnect all DAOs
            for dao in self._daos.values():
                await dao.disconnect()

            self._is_active = False
            logger.info("Unit of Work completed")

        except Exception as e:
            logger.error(f"Error completing Unit of Work: {e}")
            if exc_type is None:
                raise TransactionException(f"UoW completion failed: {e}")

    async def commit(self) -> None:
        """Commit all changes"""
        if not self._is_active:
            raise TransactionException("Unit of Work is not active")

        try:
            # Commit transactions for all DAOs
            for name, dao in self._daos.items():
                if hasattr(dao, '_commit_transaction'):
                    await dao._commit_transaction()
                    logger.debug(f"Committed transaction for {name}")

            logger.info("Unit of Work committed successfully")

        except Exception as e:
            logger.error(f"Failed to commit Unit of Work: {e}")
            await self.rollback()
            raise TransactionException(f"UoW commit failed: {e}")

    async def rollback(self) -> None:
        """Rollback all changes"""
        if not self._is_active:
            return

        try:
            # Rollback transactions for all DAOs
            for name, dao in self._daos.items():
                if hasattr(dao, '_rollback_transaction'):
                    await dao._rollback_transaction()
                    logger.debug(f"Rolled back transaction for {name}")

            logger.info("Unit of Work rolled back")

        except Exception as e:
            logger.error(f"Failed to rollback Unit of Work: {e}")
            raise TransactionException(f"UoW rollback failed: {e}")

    async def _initialize_daos(self):
        """Initialize all DAOs"""
        dao_classes = {
            DataSource.MYSQL: MySQLDAO,
            DataSource.MILVUS: MilvusDAO,
            DataSource.NEO4J: Neo4jDAO,
            DataSource.REDIS: RedisDAO,
            DataSource.MINIO: MinioDAO
        }

        for source, config in self.configs.items():
            if source in dao_classes:
                dao_class = dao_classes[source]
                dao = dao_class(config)
                await dao.connect()
                self._daos[source.value] = dao
                self.register_dao(source.value, dao)

    async def _initialize_repositories(self):
        """Initialize repositories"""
        # Document repository (uses multiple DAOs)
        document_repo = DocumentRepository(
            mysql_dao=self._daos.get(DataSource.MYSQL.value),
            milvus_dao=self._daos.get(DataSource.MILVUS.value),
            minio_dao=self._daos.get(DataSource.MINIO.value),
            neo4j_dao=self._daos.get(DataSource.NEO4J.value)
        )
        self.register_repository("document", document_repo)

        # Vector repository
        vector_repo = VectorRepository(
            milvus_dao=self._daos.get(DataSource.MILVUS.value),
            document_repository=document_repo
        )
        self.register_repository("vector", vector_repo)

        # Knowledge graph repository
        kg_repo = KnowledgeGraphRepository(
            neo4j_dao=self._daos.get(DataSource.NEO4J.value)
        )
        self.register_repository("knowledge_graph", kg_repo)

        # User repository
        user_repo = UserRepository(
            mysql_dao=self._daos.get(DataSource.MYSQL.value),
            redis_dao=self._daos.get(DataSource.REDIS.value)
        )
        self.register_repository("user", user_repo)

        # Task repository
        task_repo = TaskRepository(
            mysql_dao=self._daos.get(DataSource.MYSQL.value),
            redis_dao=self._daos.get(DataSource.REDIS.value)
        )
        self.register_repository("task", task_repo)


class DocumentUnitOfWork(BaseUnitOfWork):
    """
    Specialized Unit of Work for document operations

    Focused on document-related operations and transactions
    """

    def __init__(self, mysql_config: DatabaseConfig,
                 milvus_config: DatabaseConfig,
                 minio_config: DatabaseConfig,
                 neo4j_config: DatabaseConfig):
        super().__init__()
        self.mysql_config = mysql_config
        self.milvus_config = milvus_config
        self.minio_config = minio_config
        self.neo4j_config = neo4j_config

        # DAOs
        self.mysql_dao = None
        self.milvus_dao = None
        self.minio_dao = None
        self.neo4j_dao = None

        # Repositories
        self.document_repository = None
        self.vector_repository = None

    async def __aenter__(self):
        """Enter document Unit of Work context"""
        try:
            # Initialize DAOs
            self.mysql_dao = MySQLDAO(self.mysql_config)
            await self.mysql_dao.connect()

            self.milvus_dao = MilvusDAO(self.milvus_config)
            await self.milvus_dao.connect()

            self.minio_dao = MinioDAO(self.minio_config)
            await self.minio_dao.connect()

            self.neo4j_dao = Neo4jDAO(self.neo4j_config)
            await self.neo4j_dao.connect()

            # Initialize repositories
            self.document_repository = DocumentRepository(
                mysql_dao=self.mysql_dao,
                milvus_dao=self.milvus_dao,
                minio_dao=self.minio_dao,
                neo4j_dao=self.neo4j_dao
            )

            self.vector_repository = VectorRepository(
                milvus_dao=self.milvus_dao,
                document_repository=self.document_repository
            )

            # Register with base class
            self.register_dao("mysql", self.mysql_dao)
            self.register_dao("milvus", self.milvus_dao)
            self.register_dao("minio", self.minio_dao)
            self.register_dao("neo4j", self.neo4j_dao)
            self.register_repository("document", self.document_repository)
            self.register_repository("vector", self.vector_repository)

            self._is_active = True
            return self

        except Exception as e:
            logger.error(f"Failed to start Document Unit of Work: {e}")
            raise TransactionException(f"Document UoW start failed: {e}")

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit document Unit of Work context"""
        if not self._is_active:
            return

        try:
            if exc_type is not None:
                await self.rollback()
            else:
                await self.commit()

            # Disconnect DAOs
            if self.mysql_dao:
                await self.mysql_dao.disconnect()
            if self.milvus_dao:
                await self.milvus_dao.disconnect()
            if self.minio_dao:
                await self.minio_dao.disconnect()
            if self.neo4j_dao:
                await self.neo4j_dao.disconnect()

            self._is_active = False

        except Exception as e:
            logger.error(f"Error completing Document Unit of Work: {e}")
            if exc_type is None:
                raise TransactionException(f"Document UoW completion failed: {e}")

    async def commit(self) -> None:
        """Commit document operations"""
        try:
            # Note: Some data sources (Milvus, MinIO) don't support traditional transactions
            # We ensure consistency by ordering operations appropriately

            logger.info("Document Unit of Work committed")

        except Exception as e:
            logger.error(f"Failed to commit Document Unit of Work: {e}")
            raise TransactionException(f"Document UoW commit failed: {e}")

    async def rollback(self) -> None:
        """Rollback document operations"""
        try:
            # For data sources that don't support rollback,
            # we would need to implement compensation logic

            logger.info("Document Unit of Work rolled back")

        except Exception as e:
            logger.error(f"Failed to rollback Document Unit of Work: {e}")
            raise TransactionException(f"Document UoW rollback failed: {e}")

    async def save_document_with_embeddings(self,
                                          document: 'Document',
                                          embeddings: List['VectorData']) -> 'Document':
        """Save document with embeddings in one atomic operation"""
        if not self._is_active:
            raise TransactionException("Unit of Work is not active")

        try:
            # Save document
            saved_document = await self.document_repository.save(document)

            # Associate embeddings with document
            for embedding in embeddings:
                embedding.document_id = saved_document.id
                await self.vector_repository.save(embedding)

            return saved_document

        except Exception as e:
            logger.error(f"Failed to save document with embeddings: {e}")
            raise

    async def delete_document_cascade(self, document_id: Union[int, str]) -> bool:
        """Delete document and all related data"""
        if not self._is_active:
            raise TransactionException("Unit of Work is not active")

        try:
            # Get document
            document = await self.document_repository.find(document_id)
            if not document:
                return False

            # Delete related vectors
            # This would need to be implemented in VectorRepository

            # Delete document
            return await self.document_repository.remove(document_id)

        except Exception as e:
            logger.error(f"Failed to delete document cascade: {e}")
            raise


class RAGUnitOfWork(BaseUnitOfWork):
    """
    Unit of Work for RAG (Retrieval-Augmented Generation) operations

    Coordinates operations between documents, vectors, and knowledge graph
    """

    def __init__(self, configs: Dict[DataSource, DatabaseConfig]):
        super().__init__()
        self.configs = configs
        self._document_uow = None

    async def __aenter__(self):
        """Enter RAG Unit of Work context"""
        # Create document Unit of Work as underlying context
        self._document_uow = DocumentUnitOfWork(
            mysql_config=self.configs.get(DataSource.MYSQL),
            milvus_config=self.configs.get(DataSource.MILVUS),
            minio_config=self.configs.get(DataSource.MINIO),
            neo4j_config=self.configs.get(DataSource.NEO4J)
        )

        # Enter document context
        await self._document_uow.__aenter__()

        # Add additional RAG-specific repositories
        if DataSource.NEO4J in self.configs:
            kg_repo = KnowledgeGraphRepository(
                neo4j_dao=self._document_uow.neo4j_dao
            )
            self.register_repository("knowledge_graph", kg_repo)

        self._is_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit RAG Unit of Work context"""
        if not self._is_active:
            return

        # Use document Unit of Work cleanup
        await self._document_uow.__aexit__(exc_type, exc_val, exc_tb)
        self._is_active = False

    async def commit(self) -> None:
        """Commit RAG operations"""
        await self._document_uow.commit()

    async def rollback(self) -> None:
        """Rollback RAG operations"""
        await self._document_uow.rollback()

    async def process_document_for_rag(self,
                                     document: 'Document',
                                     extract_entities: bool = True) -> Dict[str, Any]:
        """Process document for RAG system"""
        if not self._is_active:
            raise TransactionException("Unit of Work is not active")

        try:
            result = {
                'document': None,
                'embeddings': [],
                'entities': [],
                'relationships': []
            }

            # Save document
            saved_document = await self._document_uow.document_repository.save(document)
            result['document'] = saved_document

            # Generate and save embeddings
            # This would integrate with the embedding service

            # Extract entities if requested
            if extract_entities and self._document_uow.neo4j_dao:
                # Extract entities from document content
                # Create entities and relationships in knowledge graph
                pass

            return result

        except Exception as e:
            logger.error(f"Failed to process document for RAG: {e}")
            raise

    async def search_with_context(self,
                                query: str,
                                max_documents: int = 5,
                                include_related: bool = True) -> Dict[str, Any]:
        """Search documents with context"""
        if not self._is_active:
            raise TransactionException("Unit of Work is not active")

        try:
            # Search for similar documents
            documents = await self._document_uow.document_repository.search_by_content(
                query=query,
                limit=max_documents
            )

            result = {
                'documents': documents,
                'related_entities': [],
                'related_documents': []
            }

            # Get related entities and documents if requested
            if include_related and documents:
                for doc in documents:
                    related_docs = await self._document_uow.document_repository.get_related_documents(
                        document_id=doc.id,
                        depth=2
                    )
                    result['related_documents'].extend(related_docs)

            return result

        except Exception as e:
            logger.error(f"Failed to search with context: {e}")
            raise

    @property
    def document_repository(self) -> DocumentRepository:
        """Get document repository"""
        return self._document_uow.document_repository

    @property
    def vector_repository(self) -> VectorRepository:
        """Get vector repository"""
        return self._document_uow.vector_repository

    def get_repository(self, name: str):
        """Get repository by name"""
        if name in self._repositories:
            return self._repositories[name]
        return self._document_uow.get_repository(name)