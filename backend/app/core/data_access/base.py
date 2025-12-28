"""
数据访问抽象层

Defines the foundational abstractions for the data access layer including
Repository pattern, DAO pattern, and Unit of Work pattern implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar, Generic, Union, AsyncContextManager
from contextlib import asynccontextmanager
from enum import Enum
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')
ID = TypeVar('ID')


class DataSource(str, Enum):
    """Data source types"""
    MYSQL = "mysql"
    MILVUS = "milvus"
    NEO4J = "neo4j"
    REDIS = "redis"
    MINIO = "minio"
    ELASTICSEARCH = "elasticsearch"


@dataclass
class DatabaseConfig:
    """Database connection configuration"""
    source: DataSource
    host: str
    port: int
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.connection_params is None:
            self.connection_params = {}


class BaseDAO(ABC, Generic[T, ID]):
    """
    Data Access Object - provides low-level data access operations

    DAOs encapsulate the complexity of data access for a specific data source
    and handle the raw CRUD operations without business logic.
    """

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._connection = None
        self._is_connected = False

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source"""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass

    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity"""
        pass

    @abstractmethod
    async def get_by_id(self, entity_id: ID) -> Optional[T]:
        """Retrieve entity by ID"""
        pass

    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity"""
        pass

    @abstractmethod
    async def delete(self, entity_id: ID) -> bool:
        """Delete entity by ID"""
        pass

    @abstractmethod
    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[T]:
        """List entities with optional filtering"""
        pass

    @property
    def is_connected(self) -> bool:
        """Check if DAO is connected"""
        return self._is_connected

    @asynccontextmanager
    async def transaction(self):
        """Transaction context manager"""
        try:
            await self._begin_transaction()
            yield self
            await self._commit_transaction()
        except Exception as e:
            await self._rollback_transaction()
            logger.error(f"Transaction failed: {e}")
            raise

    @abstractmethod
    async def _begin_transaction(self) -> None:
        """Begin a transaction"""
        pass

    @abstractmethod
    async def _commit_transaction(self) -> None:
        """Commit a transaction"""
        pass

    @abstractmethod
    async def _rollback_transaction(self) -> None:
        """Rollback a transaction"""
        pass


class BaseRepository(ABC, Generic[T, ID]):
    """
    Repository - provides domain-oriented data access operations

    Repositories work with domain models and may coordinate multiple DAOs
    to provide a cohesive interface for the domain layer.
    """

    def __init__(self, dao: BaseDAO[T, ID]):
        self.dao = dao

    @abstractmethod
    async def save(self, entity: T) -> T:
        """Save entity (create or update)"""
        pass

    @abstractmethod
    async def find(self, entity_id: ID) -> Optional[T]:
        """Find entity by ID"""
        pass

    @abstractmethod
    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List[T]:
        """Find all entities matching filters"""
        pass

    @abstractmethod
    async def remove(self, entity_id: ID) -> bool:
        """Remove entity by ID"""
        pass

    @abstractmethod
    async def exists(self, entity_id: ID) -> bool:
        """Check if entity exists"""
        pass

    @abstractmethod
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities matching filters"""
        pass


class UnitOfWork(ABC, AsyncContextManager):
    """
    Unit of Work - manages transactions and business operations

    Coordinates multiple repositories and ensures atomic operations
    across different data sources.
    """

    def __init__(self):
        self._repositories: Dict[str, BaseRepository] = {}
        self._daos: Dict[str, BaseDAO] = {}
        self._is_active = False

    @abstractmethod
    async def __aenter__(self):
        """Enter Unit of Work context"""
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit Unit of Work context"""
        if exc_type is not None:
            await self.rollback()
        else:
            await self.commit()

    @abstractmethod
    async def commit(self) -> None:
        """Commit all changes"""
        pass

    @abstractmethod
    async def rollback(self) -> None:
        """Rollback all changes"""
        pass

    @property
    def is_active(self) -> bool:
        """Check if Unit of Work is active"""
        return self._is_active

    def get_repository(self, name: str) -> BaseRepository:
        """Get repository by name"""
        if name not in self._repositories:
            raise ValueError(f"Repository '{name}' not found")
        return self._repositories[name]

    def get_dao(self, name: str) -> BaseDAO:
        """Get DAO by name"""
        if name not in self._daos:
            raise ValueError(f"DAO '{name}' not found")
        return self._daos[name]

    def register_repository(self, name: str, repository: BaseRepository) -> None:
        """Register a repository"""
        self._repositories[name] = repository

    def register_dao(self, name: str, dao: BaseDAO) -> None:
        """Register a DAO"""
        self._daos[name] = dao


class DataAccessException(Exception):
    """Base exception for data access layer"""
    pass


class ConnectionException(DataAccessException):
    """Connection-related exceptions"""
    pass


class TransactionException(DataAccessException):
    """Transaction-related exceptions"""
    pass


class ValidationException(DataAccessException):
    """Data validation exceptions"""
    pass


class OptimisticLockException(DataAccessException):
    """Optimistic locking exceptions"""
    pass


# Mixins for common operations
class TimestampMixin:
    """Mixin for timestamp tracking"""

    created_at: Any = None
    updated_at: Any = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class SoftDeleteMixin:
    """Mixin for soft delete functionality"""

    is_deleted: bool = False
    deleted_at: Any = None

    def soft_delete(self):
        """Mark entity as deleted"""
        from datetime import datetime
        self.is_deleted = True
        self.deleted_at = datetime.utcnow()


class AuditableMixin(TimestampMixin):
    """Mixin for audit trail functionality"""

    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    version: int = 0

    def increment_version(self):
        """Increment version for optimistic locking"""
        self.version += 1


# Query builder for complex queries
class QueryBuilder:
    """Builder for constructing complex queries"""

    def __init__(self):
        self._filters: List[Dict[str, Any]] = []
        self._sort: List[Dict[str, str]] = []
        self._pagination: Dict[str, int] = {}
        self._fields: Optional[List[str]] = None
        self._includes: List[str] = []

    def filter(self, field: str, operator: str, value: Any) -> 'QueryBuilder':
        """Add filter condition"""
        self._filters.append({
            'field': field,
            'operator': operator,
            'value': value
        })
        return self

    def order_by(self, field: str, direction: str = 'asc') -> 'QueryBuilder':
        """Add sort condition"""
        self._sort.append({
            'field': field,
            'direction': direction
        })
        return self

    def paginate(self, page: int, per_page: int) -> 'QueryBuilder':
        """Add pagination"""
        self._pagination = {
            'page': page,
            'per_page': per_page,
            'offset': (page - 1) * per_page
        }
        return self

    def select(self, fields: List[str]) -> 'QueryBuilder':
        """Select specific fields"""
        self._fields = fields
        return self

    def include(self, relationships: List[str]) -> 'QueryBuilder':
        """Include related entities"""
        self._includes.extend(relationships)
        return self

    def build(self) -> Dict[str, Any]:
        """Build query dict"""
        return {
            'filters': self._filters,
            'sort': self._sort,
            'pagination': self._pagination,
            'fields': self._fields,
            'includes': self._includes
        }


# Result wrapper for paginated responses
@dataclass
class PaginatedResult(Generic[T]):
    """Wrapper for paginated query results"""

    items: List[T]
    total: int
    page: int
    per_page: int
    total_pages: int
    has_next: bool
    has_prev: bool

    @classmethod
    def create(cls, items: List[T], total: int, page: int, per_page: int) -> 'PaginatedResult[T]':
        """Create paginated result"""
        total_pages = (total + per_page - 1) // per_page
        return cls(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1
        )