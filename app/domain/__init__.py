"""
Domain Layer

Contains the core business logic, domain models, and domain services.
This layer is independent of infrastructure concerns like databases,
external services, or UI frameworks.

Structure:
- entities/: Core domain entities with business logic
- value_objects/: Value objects that are immutable and have no identity
- aggregates/: Aggregates that group related entities and value objects
- repositories/: Abstract repository interfaces (no implementations)
- services/: Domain services for business logic that doesn't fit entities
- events/: Domain events for decoupled communication
"""

from .entities import (
    Document,
    User,
    Task,
    Query,
    KnowledgeEntity,
    Fragment,
    Report
)

from .value_objects import (
    DocumentStatus,
    TaskStatus,
    QueryType,
    ContentType,
    EmbeddingVector,
    TextChunk,
    Metadata,
    ConfidenceScore
)

from .aggregates import (
    DocumentAggregate,
    QueryAggregate,
    TaskAggregate,
    KnowledgeAggregate
)

from .repositories import (
    DocumentRepository,
    UserRepository,
    TaskRepository,
    QueryRepository,
    KnowledgeRepository
)

from .services import (
    DocumentProcessingService,
    QueryProcessingService,
    KnowledgeExtractionService,
    EmbeddingService,
    RAGService
)

from .events import (
    DocumentCreatedEvent,
    DocumentUpdatedEvent,
    DocumentDeletedEvent,
    QueryExecutedEvent,
    TaskCompletedEvent,
    KnowledgeUpdatedEvent
)

__all__ = [
    # Entities
    "Document",
    "User",
    "Task",
    "Query",
    "KnowledgeEntity",
    "Fragment",
    "Report",

    # Value Objects
    "DocumentStatus",
    "TaskStatus",
    "QueryType",
    "ContentType",
    "EmbeddingVector",
    "TextChunk",
    "Metadata",
    "ConfidenceScore",

    # Aggregates
    "DocumentAggregate",
    "QueryAggregate",
    "TaskAggregate",
    "KnowledgeAggregate",

    # Repository Interfaces
    "DocumentRepository",
    "UserRepository",
    "TaskRepository",
    "QueryRepository",
    "KnowledgeRepository",

    # Domain Services
    "DocumentProcessingService",
    "QueryProcessingService",
    "KnowledgeExtractionService",
    "EmbeddingService",
    "RAGService",

    # Events
    "DocumentCreatedEvent",
    "DocumentUpdatedEvent",
    "DocumentDeletedEvent",
    "QueryExecutedEvent",
    "TaskCompletedEvent",
    "KnowledgeUpdatedEvent"
]