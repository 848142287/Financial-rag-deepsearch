"""
Data Access Layer

Provides unified abstractions for data access operations across different storage systems.
Implements Repository pattern, Unit of Work pattern, and DAO pattern.
"""

from .base import (
    BaseRepository,
    BaseDAO,
    UnitOfWork,
    DataSource,
    DatabaseConfig
)

from .repositories import (
    DocumentRepository,
    VectorRepository,
    KnowledgeGraphRepository,
    UserRepository,
    TaskRepository
)

from .daos import (
    DocumentDAO,
    VectorDAO,
    KnowledgeGraphDAO,
    UserDAO,
    TaskDAO
)

from .unit_of_work import (
    DataUnitOfWork,
    DocumentUnitOfWork,
    RAGUnitOfWork
)

from .factories import (
    RepositoryFactory,
    DAOFactory,
    UnitOfWorkFactory
)

__all__ = [
    # Base classes
    "BaseRepository",
    "BaseDAO",
    "UnitOfWork",
    "DataSource",
    "DatabaseConfig",

    # Repositories
    "DocumentRepository",
    "VectorRepository",
    "KnowledgeGraphRepository",
    "UserRepository",
    "TaskRepository",

    # DAOs
    "DocumentDAO",
    "VectorDAO",
    "KnowledgeGraphDAO",
    "UserDAO",
    "TaskDAO",

    # Unit of Work
    "DataUnitOfWork",
    "DocumentUnitOfWork",
    "RAGUnitOfWork",

    # Factories
    "RepositoryFactory",
    "DAOFactory",
    "UnitOfWorkFactory"
]