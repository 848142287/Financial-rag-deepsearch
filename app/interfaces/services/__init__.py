"""
Service Interface Definitions

Defines contracts for application services that coordinate
use cases and business operations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union, AsyncIterator
from datetime import datetime
import asyncio

from ..dto import (
    DocumentDTO,
    QueryDTO,
    TaskDTO,
    UserDTO,
    SearchResultDTO,
    ProcessingStatusDTO,
    PaginationDTO
)


class IDocumentService(ABC):
    """Interface for document management service"""

    @abstractmethod
    async def upload_document(self, file_path: str, user_id: int,
                             metadata: Optional[Dict[str, Any]] = None) -> DocumentDTO:
        """Upload and process a new document"""
        pass

    @abstractmethod
    async def get_document(self, document_id: int) -> Optional[DocumentDTO]:
        """Get document by ID"""
        pass

    @abstractmethod
    async def get_user_documents(self, user_id: int,
                                pagination: Optional[PaginationDTO] = None) -> List[DocumentDTO]:
        """Get all documents for a user"""
        pass

    @abstractmethod
    async def search_documents(self, query: str, filters: Optional[Dict[str, Any]] = None,
                             pagination: Optional[PaginationDTO] = None) -> SearchResultDTO:
        """Search documents"""
        pass

    @abstractmethod
    async def update_document(self, document_id: int,
                             updates: Dict[str, Any]) -> DocumentDTO:
        """Update document metadata"""
        pass

    @abstractmethod
    async def delete_document(self, document_id: int) -> bool:
        """Delete a document"""
        pass

    @abstractmethod
    async def get_processing_status(self, document_id: int) -> ProcessingStatusDTO:
        """Get document processing status"""
        pass

    @abstractmethod
    async def retry_processing(self, document_id: int) -> bool:
        """Retry failed document processing"""
        pass

    @abstractmethod
    async def stream_processing_updates(self, document_id: int) -> AsyncIterator[ProcessingStatusDTO]:
        """Stream real-time processing updates"""
        pass


class IQueryService(ABC):
    """Interface for query/RAG service"""

    @abstractmethod
    async def execute_query(self, query: str, user_id: int,
                           filters: Optional[Dict[str, Any]] = None,
                           document_ids: Optional[List[int]] = None) -> QueryDTO:
        """Execute a query with RAG"""
        pass

    @abstractmethod
    async def stream_query(self, query: str, user_id: int,
                          filters: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        """Stream query response in real-time"""
        pass

    @abstractmethod
    async def get_query_history(self, user_id: int,
                              pagination: Optional[PaginationDTO] = None) -> List[QueryDTO]:
        """Get query history for user"""
        pass

    @abstractmethod
    async def get_similar_queries(self, query_id: int, limit: int = 5) -> List[QueryDTO]:
        """Get similar queries to a given query"""
        pass

    @abstractmethod
    async def get_query_suggestions(self, partial_query: str,
                                   user_id: Optional[int] = None) -> List[str]:
        """Get query suggestions based on partial input"""
        pass

    @abstractmethod
    async def explain_query(self, query_id: int) -> Dict[str, Any]:
        """Get explanation of how query was processed"""
        pass


class ITaskService(ABC):
    """Interface for background task management"""

    @abstractmethod
    async def create_task(self, task_type: str, data: Dict[str, Any],
                         user_id: Optional[int] = None) -> TaskDTO:
        """Create a new background task"""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[TaskDTO]:
        """Get task by ID"""
        pass

    @abstractmethod
    async def get_user_tasks(self, user_id: int,
                            status: Optional[str] = None,
                            pagination: Optional[PaginationDTO] = None) -> List[TaskDTO]:
        """Get tasks for a user"""
        pass

    @abstractmethod
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        pass

    @abstractmethod
    async def retry_task(self, task_id: str) -> bool:
        """Retry a failed task"""
        pass

    @abstractmethod
    async def get_task_logs(self, task_id: str) -> List[str]:
        """Get task execution logs"""
        pass

    @abstractmethod
    async def monitor_task(self, task_id: str) -> AsyncIterator[TaskDTO]:
        """Monitor task progress in real-time"""
        pass


class ICacheService(ABC):
    """Interface for caching service"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

    @abstractmethod
    async def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache keys matching pattern"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        pass

    @abstractmethod
    async def get_ttl(self, key: str) -> Optional[int]:
        """Get remaining TTL for key"""
        pass

    @abstractmethod
    async def increment(self, key: str, delta: int = 1) -> Optional[int]:
        """Increment numeric value"""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        pass


class IEmbeddingService(ABC):
    """Interface for text embedding service"""

    @abstractmethod
    async def generate_embedding(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embedding for text"""
        pass

    @abstractmethod
    async def batch_generate_embeddings(self, texts: List[str],
                                       model: Optional[str] = None) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        pass

    @abstractmethod
    async def calculate_similarity(self, embedding1: List[float],
                                  embedding2: List[float]) -> float:
        """Calculate similarity between two embeddings"""
        pass

    @abstractmethod
    async def find_similar_embeddings(self, query_embedding: List[float],
                                     limit: int = 10,
                                     threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find similar embeddings"""
        pass

    @abstractmethod
    async def get_embedding_info(self, model: Optional[str] = None) -> Dict[str, Any]:
        """Get embedding model information"""
        pass


class IStorageService(ABC):
    """Interface for file storage service"""

    @abstractmethod
    async def upload_file(self, file_path: str, content: bytes,
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload file and return storage key"""
        pass

    @abstractmethod
    async def download_file(self, storage_key: str) -> Optional[bytes]:
        """Download file by storage key"""
        pass

    @abstractmethod
    async def delete_file(self, storage_key: str) -> bool:
        """Delete file by storage key"""
        pass

    @abstractmethod
    async def file_exists(self, storage_key: str) -> bool:
        """Check if file exists"""
        pass

    @abstractmethod
    async def get_file_info(self, storage_key: str) -> Optional[Dict[str, Any]]:
        """Get file metadata"""
        pass

    @abstractmethod
    async def list_files(self, prefix: Optional[str] = None) -> List[str]:
        """List files with optional prefix"""
        pass


class INotificationService(ABC):
    """Interface for notification service"""

    @abstractmethod
    async def send_notification(self, user_id: int, message: str,
                               notification_type: str = "info",
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Send notification to user"""
        pass

    @abstractmethod
    async def broadcast_notification(self, message: str,
                                    channels: Optional[List[str]] = None) -> int:
        """Broadcast notification to channels"""
        pass

    @abstractmethod
    async def get_user_notifications(self, user_id: int,
                                   unread_only: bool = False,
                                   pagination: Optional[PaginationDTO] = None) -> List[Dict[str, Any]]:
        """Get user notifications"""
        pass

    @abstractmethod
    async def mark_notification_read(self, notification_id: int, user_id: int) -> bool:
        """Mark notification as read"""
        pass

    @abstractmethod
    async def get_notification_preferences(self, user_id: int) -> Dict[str, bool]:
        """Get user notification preferences"""
        pass

    @abstractmethod
    async def update_notification_preferences(self, user_id: int,
                                            preferences: Dict[str, bool]) -> bool:
        """Update user notification preferences"""
        pass


class IUserService(ABC):
    """Interface for user management service"""

    @abstractmethod
    async def create_user(self, email: str, password: str,
                         metadata: Optional[Dict[str, Any]] = None) -> UserDTO:
        """Create new user"""
        pass

    @abstractmethod
    async def authenticate_user(self, email: str, password: str) -> Optional[UserDTO]:
        """Authenticate user credentials"""
        pass

    @abstractmethod
    async def get_user(self, user_id: int) -> Optional[UserDTO]:
        """Get user by ID"""
        pass

    @abstractmethod
    async def update_user(self, user_id: int, updates: Dict[str, Any]) -> UserDTO:
        """Update user information"""
        pass

    @abstractmethod
    async def delete_user(self, user_id: int) -> bool:
        """Delete user account"""
        pass

    @abstractmethod
    async def change_password(self, user_id: int, old_password: str,
                            new_password: str) -> bool:
        """Change user password"""
        pass

    @abstractmethod
    async def reset_password(self, email: str) -> bool:
        """Initiate password reset"""
        pass

    @abstractmethod
    async def get_user_activity(self, user_id: int,
                              activity_type: Optional[str] = None,
                              pagination: Optional[PaginationDTO] = None) -> List[Dict[str, Any]]:
        """Get user activity log"""
        pass


class IKnowledgeService(ABC):
    """Interface for knowledge graph service"""

    @abstractmethod
    async def extract_entities(self, text: str, document_id: int) -> List[Dict[str, Any]]:
        """Extract entities from text"""
        pass

    @abstractmethod
    async def create_entity(self, entity_type: str, name: str,
                           properties: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create knowledge entity"""
        pass

    @abstractmethod
    async def create_relationship(self, from_entity_id: str, to_entity_id: str,
                                relationship_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create relationship between entities"""
        pass

    @abstractmethod
    async def find_related_entities(self, entity_id: str,
                                  relationship_types: Optional[List[str]] = None,
                                  depth: int = 1) -> List[Dict[str, Any]]:
        """Find entities related to given entity"""
        pass

    @abstractmethod
    async def search_entities(self, query: str, entity_type: Optional[str] = None,
                           limit: int = 10) -> List[Dict[str, Any]]:
        """Search knowledge entities"""
        pass

    @abstractmethod
    async def get_entity_graph(self, entity_ids: List[str],
                             max_depth: int = 2) -> Dict[str, Any]:
        """Get knowledge graph for entities"""
        pass

    @abstractmethod
    async def update_entity(self, entity_id: str,
                           updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update entity properties"""
        pass

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity"""
        pass


class IReportService(ABC):
    """Interface for report generation service"""

    @abstractmethod
    async def generate_document_report(self, document_id: int,
                                     report_type: str = "summary") -> str:
        """Generate document report and return file key"""
        pass

    @abstractmethod
    async def generate_user_report(self, user_id: int,
                                 report_type: str = "activity",
                                 date_range: Optional[Dict[str, datetime]] = None) -> str:
        """Generate user report and return file key"""
        pass

    @abstractmethod
    async def generate_system_report(self, report_type: str = "usage",
                                   date_range: Optional[Dict[str, datetime]] = None) -> str:
        """Generate system report and return file key"""
        pass

    @abstractmethod
    async def get_report_status(self, report_id: str) -> Dict[str, Any]:
        """Get report generation status"""
        pass

    @abstractmethod
    async def download_report(self, report_id: str) -> Optional[bytes]:
        """Download generated report"""
        pass

    @abstractmethod
    async def get_available_reports(self) -> List[Dict[str, Any]]:
        """Get list of available report types"""
        pass