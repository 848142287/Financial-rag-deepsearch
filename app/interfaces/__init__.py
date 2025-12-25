"""
Service Interfaces

Defines clear contracts and interfaces for application services.
This layer provides the interface definitions that are implemented
by the application services and used by the infrastructure layer.
"""

from .services import (
    IDocumentService,
    IQueryService,
    ITaskService,
    ICacheService,
    IEmbeddingService,
    IStorageService,
    INotificationService,
    IUserService,
    IKnowledgeService,
    IReportService
)

from .repositories import (
    IDocumentRepository,
    IUserRepository,
    ITaskRepository,
    IQueryRepository,
    IKnowledgeRepository
)

from .external import (
   ILLMService,
    IVectorDatabaseService,
    IGraphDatabaseService,
    IFileStorageService,
    IMessageQueueService
)

from .dto import (
    DocumentDTO,
    QueryDTO,
    TaskDTO,
    UserDTO,
    SearchResultDTO,
    ProcessingStatusDTO,
    PaginationDTO
)

__all__ = [
    # Service Interfaces
    "IDocumentService",
    "IQueryService",
    "ITaskService",
    "ICacheService",
    "IEmbeddingService",
    "IStorageService",
    "INotificationService",
    "IUserService",
    "IKnowledgeService",
    "IReportService",

    # Repository Interfaces
    "IDocumentRepository",
    "IUserRepository",
    "ITaskRepository",
    "IQueryRepository",
    "IKnowledgeRepository",

    # External Service Interfaces
    "ILLMService",
    "IVectorDatabaseService",
    "IGraphDatabaseService",
    "IFileStorageService",
    "IMessageQueueService",

    # DTOs
    "DocumentDTO",
    "QueryDTO",
    "TaskDTO",
    "UserDTO",
    "SearchResultDTO",
    "ProcessingStatusDTO",
    "PaginationDTO"
]