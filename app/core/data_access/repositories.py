"""
Repository Implementations

Concrete implementations of repositories that work with domain models
and coordinate multiple DAOs for complex operations.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from datetime import datetime

from .base import (
    BaseRepository,
    PaginatedResult,
    QueryBuilder,
    ID,
    T
)

logger = logging.getLogger(__name__)


class DocumentRepository(BaseRepository):
    """
    Repository for Document domain objects

    Coordinates document storage across multiple systems:
    - MySQL for metadata
    - Milvus for embeddings
    - MinIO for files
    - Neo4j for relationships
    """

    def __init__(self, mysql_dao, milvus_dao, minio_dao, neo4j_dao):
        # Store DAOs for different data sources
        self.mysql_dao = mysql_dao
        self.milvus_dao = milvus_dao
        self.minio_dao = minio_dao
        self.neo4j_dao = neo4j_dao

    async def save(self, document: 'Document') -> 'Document':
        """Save document across all storage systems"""
        try:
            # Save metadata to MySQL
            saved_document = await self.mysql_dao.create(document)

            # Save file to MinIO
            if document.file_content:
                await self.minio_dao.upload(
                    key=f"documents/{document.id}/original",
                    content=document.file_content
                )

            # Create document node in Neo4j
            await self.neo4j_dao.create_document_node(
                document_id=document.id,
                title=document.title,
                metadata=document.metadata or {}
            )

            logger.info(f"Document saved successfully: {document.id}")
            return saved_document

        except Exception as e:
            logger.error(f"Failed to save document: {e}")
            raise

    async def find(self, document_id: ID) -> Optional['Document']:
        """Find document by ID"""
        # Get metadata from MySQL
        document = await self.mysql_dao.get_by_id(document_id)
        if not document:
            return None

        # Enrich with additional data if needed
        return document

    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List['Document']:
        """Find all documents matching filters"""
        return await self.mysql_dao.list(filters=filters)

    async def remove(self, document_id: ID) -> bool:
        """Remove document from all storage systems"""
        try:
            # Remove from MinIO
            await self.minio_dao.delete_prefix(f"documents/{document_id}")

            # Remove from Neo4j
            await self.neo4j_dao.delete_document_node(document_id)

            # Remove from MySQL
            return await self.mysql_dao.delete(document_id)

        except Exception as e:
            logger.error(f"Failed to remove document {document_id}: {e}")
            raise

    async def exists(self, document_id: ID) -> bool:
        """Check if document exists"""
        return await self.mysql_dao.get_by_id(document_id) is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count documents matching filters"""
        # This would need to be implemented in the DAO
        return len(await self.find_all(filters))

    async def search_by_content(self, query: str, limit: int = 10) -> List['Document']:
        """Search documents by content using vector similarity"""
        # Convert query to embedding
        query_embedding = await self._get_query_embedding(query)

        # Search in Milvus
        similar_docs = await self.milvus_dao.search(
            query_embedding=query_embedding,
            limit=limit
        )

        # Get full document details from MySQL
        document_ids = [doc['id'] for doc in similar_docs]
        documents = await self.mysql_dao.get_by_ids(document_ids)

        return documents

    async def get_related_documents(self, document_id: ID, depth: int = 2) -> List['Document']:
        """Get documents related through knowledge graph"""
        related_ids = await self.neo4j_dao.get_related_documents(
            document_id=document_id,
            depth=depth
        )

        if related_ids:
            return await self.mysql_dao.get_by_ids(related_ids)
        return []

    async def _get_query_embedding(self, query: str) -> List[float]:
        """Get embedding for query string"""
        # This would use the embedding service
        from app.services.embedding_service import embedding_service
        return await embedding_service.get_embedding(query)


class VectorRepository(BaseRepository):
    """
    Repository for vector/embedding operations

    Manages vectors in Milvus and coordinates with document metadata
    """

    def __init__(self, milvus_dao, document_repository):
        self.milvus_dao = milvus_dao
        self.document_repository = document_repository

    async def save(self, vector_data: 'VectorData') -> 'VectorData':
        """Save vector data"""
        # Save to Milvus
        await self.milvus_dao.insert(vector_data)
        return vector_data

    async def find(self, vector_id: ID) -> Optional['VectorData']:
        """Find vector by ID"""
        return await self.milvus_dao.get_by_id(vector_id)

    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List['VectorData']:
        """Find vectors matching filters"""
        return await self.milvus_dao.list(filters=filters)

    async def remove(self, vector_id: ID) -> bool:
        """Remove vector"""
        return await self.milvus_dao.delete(vector_id)

    async def exists(self, vector_id: ID) -> bool:
        """Check if vector exists"""
        return await self.milvus_dao.get_by_id(vector_id) is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count vectors"""
        return await self.milvus_dao.count(filters=filters)

    async def search_similar(self, query_vector: List[float],
                           limit: int = 10,
                           filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors"""
        results = await self.milvus_dao.search(
            query_vector=query_vector,
            limit=limit,
            filters=filters
        )

        # Enrich with document metadata
        for result in results:
            if 'document_id' in result:
                document = await self.document_repository.find(result['document_id'])
                if document:
                    result['document'] = document

        return results

    async def batch_insert(self, vectors: List['VectorData']) -> bool:
        """Batch insert vectors"""
        try:
            await self.milvus_dao.batch_insert(vectors)
            return True
        except Exception as e:
            logger.error(f"Failed to batch insert vectors: {e}")
            return False

    async def update_vector(self, vector_id: ID, new_vector: List[float]) -> bool:
        """Update existing vector"""
        try:
            await self.milvus_dao.update(vector_id, new_vector)
            return True
        except Exception as e:
            logger.error(f"Failed to update vector {vector_id}: {e}")
            return False


class KnowledgeGraphRepository(BaseRepository):
    """
    Repository for knowledge graph operations

    Manages entities and relationships in Neo4j
    """

    def __init__(self, neo4j_dao):
        self.neo4j_dao = neo4j_dao

    async def save(self, entity: 'Entity') -> 'Entity':
        """Save entity to knowledge graph"""
        return await self.neo4j_dao.create_entity(entity)

    async def find(self, entity_id: ID) -> Optional['Entity']:
        """Find entity by ID"""
        return await self.neo4j_dao.get_entity_by_id(entity_id)

    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List['Entity']:
        """Find entities matching filters"""
        return await self.neo4j_dao.list_entities(filters=filters)

    async def remove(self, entity_id: ID) -> bool:
        """Remove entity"""
        return await self.neo4j_dao.delete_entity(entity_id)

    async def exists(self, entity_id: ID) -> bool:
        """Check if entity exists"""
        return await self.neo4j_dao.entity_exists(entity_id)

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count entities"""
        return await self.neo4j_dao.count_entities(filters=filters)

    async def create_relationship(self, from_entity: ID,
                                to_entity: ID,
                                relationship_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create relationship between entities"""
        return await self.neo4j_dao.create_relationship(
            from_entity, to_entity, relationship_type, properties
        )

    async def find_related_entities(self, entity_id: ID,
                                  relationship_types: Optional[List[str]] = None,
                                  depth: int = 1) -> List['Entity']:
        """Find entities related through relationships"""
        return await self.neo4j_dao.get_related_entities(
            entity_id, relationship_types, depth
        )

    async def find_path(self, from_entity: ID, to_entity: ID,
                       max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find path between two entities"""
        return await self.neo4j_dao.find_path(
            from_entity, to_entity, max_depth
        )

    async def get_entity_neighbors(self, entity_id: ID) -> Dict[str, List['Entity']]:
        """Get all neighbors of an entity grouped by relationship type"""
        return await self.neo4j_dao.get_entity_neighbors(entity_id)


class UserRepository(BaseRepository):
    """
    Repository for user management
    """

    def __init__(self, mysql_dao, redis_dao):
        self.mysql_dao = mysql_dao
        self.redis_dao = redis_dao

    async def save(self, user: 'User') -> 'User':
        """Save user"""
        # Save to MySQL
        saved_user = await self.mysql_dao.create(user)

        # Cache user session in Redis
        await self.redis_dao.set(
            key=f"user:{user.id}",
            value=user.to_dict(),
            ttl=3600  # 1 hour
        )

        return saved_user

    async def find(self, user_id: ID) -> Optional['User']:
        """Find user by ID"""
        # Try cache first
        cached_user = await self.redis_dao.get(f"user:{user_id}")
        if cached_user:
            return User.from_dict(cached_user)

        # Get from database
        user = await self.mysql_dao.get_by_id(user_id)
        if user:
            # Update cache
            await self.redis_dao.set(
                key=f"user:{user_id}",
                value=user.to_dict(),
                ttl=3600
            )

        return user

    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List['User']:
        """Find users matching filters"""
        return await self.mysql_dao.list(filters=filters)

    async def remove(self, user_id: ID) -> bool:
        """Remove user"""
        # Remove from cache
        await self.redis_dao.delete(f"user:{user_id}")

        # Remove from database
        return await self.mysql_dao.delete(user_id)

    async def exists(self, user_id: ID) -> bool:
        """Check if user exists"""
        return await self.mysql_dao.get_by_id(user_id) is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count users"""
        return len(await self.find_all(filters))

    async def authenticate(self, username: str, password: str) -> Optional['User']:
        """Authenticate user"""
        user = await self.mysql_dao.get_by_username(username)
        if user and user.verify_password(password):
            # Update last login
            user.last_login = datetime.utcnow()
            await self.mysql_dao.update(user)

            # Update cache
            await self.redis_dao.set(
                key=f"user:{user.id}",
                value=user.to_dict(),
                ttl=3600
            )

            return user

        return None


class TaskRepository(BaseRepository):
    """
    Repository for task management and background jobs
    """

    def __init__(self, mysql_dao, redis_dao):
        self.mysql_dao = mysql_dao
        self.redis_dao = redis_dao

    async def save(self, task: 'Task') -> 'Task':
        """Save task"""
        # Save to MySQL for persistence
        saved_task = await self.mysql_dao.create(task)

        # Queue task in Redis
        await self.redis_dao.enqueue_task(task)

        return saved_task

    async def find(self, task_id: ID) -> Optional['Task']:
        """Find task by ID"""
        return await self.mysql_dao.get_by_id(task_id)

    async def find_all(self, filters: Optional[Dict[str, Any]] = None) -> List['Task']:
        """Find tasks matching filters"""
        return await self.mysql_dao.list(filters=filters)

    async def remove(self, task_id: ID) -> bool:
        """Remove task"""
        # Cancel task if running
        await self.redis_dao.cancel_task(task_id)

        # Remove from database
        return await self.mysql_dao.delete(task_id)

    async def exists(self, task_id: ID) -> bool:
        """Check if task exists"""
        return await self.mysql_dao.get_by_id(task_id) is not None

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """Count tasks"""
        return len(await self.find_all(filters))

    async def get_pending_tasks(self, limit: int = 100) -> List['Task']:
        """Get pending tasks from queue"""
        return await self.redis_dao.get_pending_tasks(limit)

    async def update_task_status(self, task_id: ID,
                               status: str,
                               result: Optional[Dict[str, Any]] = None,
                               error: Optional[str] = None) -> bool:
        """Update task status"""
        # Update in database
        task = await self.find(task_id)
        if task:
            task.status = status
            task.result = result
            task.error = error
            await self.mysql_dao.update(task)

        # Update cache/queue
        return await self.redis_dao.update_task_status(task_id, status, result, error)

    async def get_user_tasks(self, user_id: ID,
                           status: Optional[str] = None) -> List['Task']:
        """Get tasks for a specific user"""
        filters = {'user_id': user_id}
        if status:
            filters['status'] = status

        return await self.find_all(filters)