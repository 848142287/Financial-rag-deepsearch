"""
Data Access Object Implementations

Concrete DAO implementations for different data sources.
These provide low-level data access operations without business logic.
"""

from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from .base import (
    BaseDAO,
    DatabaseConfig,
    DataSource,
    ConnectionException,
    TransactionException
)

logger = logging.getLogger(__name__)


class MySQLDAO(BaseDAO):
    """DAO for MySQL database operations"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._pool = None
        self._connection = None

    async def connect(self) -> None:
        """Establish MySQL connection"""
        try:
            import aiomysql

            self._pool = await aiomysql.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                db=self.config.database,
                **self.config.connection_params
            )

            self._is_connected = True
            logger.info(f"Connected to MySQL: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise ConnectionException(f"MySQL connection failed: {e}")

    async def disconnect(self) -> None:
        """Close MySQL connection"""
        if self._pool:
            self._pool.close()
            await self._pool.wait_closed()
            self._is_connected = False
            logger.info("Disconnected from MySQL")

    async def create(self, entity: Any) -> Any:
        """Create new entity"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                # Build INSERT query based on entity
                table_name = entity.__class__.__name__.lower()
                columns = []
                values = []
                placeholders = []

                for key, value in entity.__dict__.items():
                    if not key.startswith('_'):
                        columns.append(key)
                        values.append(value)
                        placeholders.append('%s')

                query = f"""
                    INSERT INTO {table_name} ({', '.join(columns)})
                    VALUES ({', '.join(placeholders)})
                """

                await cursor.execute(query, values)
                entity_id = cursor.lastrowid
                await conn.commit()

                # Set the ID on the entity
                if hasattr(entity, 'id'):
                    entity.id = entity_id

                return entity

    async def get_by_id(self, entity_id: Union[int, str]) -> Optional[Any]:
        """Get entity by ID"""
        if not self._pool:
            await self.connect()

        # This is a simplified implementation
        # In practice, you'd need to know the entity type
        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                query = "SELECT * FROM documents WHERE id = %s"
                await cursor.execute(query, (entity_id,))
                row = await cursor.fetchone()

                if row:
                    # Convert to entity object
                    return self._row_to_entity(row, 'document')

                return None

    async def update(self, entity: Any) -> Any:
        """Update entity"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                table_name = entity.__class__.__name__.lower()
                updates = []
                values = []

                for key, value in entity.__dict__.items():
                    if not key.startswith('_') and key != 'id':
                        updates.append(f"{key} = %s")
                        values.append(value)

                values.append(entity.id)

                query = f"""
                    UPDATE {table_name}
                    SET {', '.join(updates)}
                    WHERE id = %s
                """

                await cursor.execute(query, values)
                await conn.commit()

                return entity

    async def delete(self, entity_id: Union[int, str]) -> bool:
        """Delete entity by ID"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                query = "DELETE FROM documents WHERE id = %s"
                await cursor.execute(query, (entity_id,))
                await conn.commit()

                return cursor.rowcount > 0

    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Any]:
        """List entities with filters"""
        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            async with conn.cursor() as cursor:
                query = "SELECT * FROM documents"
                params = []

                if filters:
                    where_clauses = []
                    for key, value in filters.items():
                        where_clauses.append(f"{key} = %s")
                        params.append(value)
                    query += f" WHERE {' AND '.join(where_clauses)}"

                if limit:
                    query += f" LIMIT {limit}"
                if offset:
                    query += f" OFFSET {offset}"

                await cursor.execute(query, params)
                rows = await cursor.fetchall()

                return [self._row_to_entity(row, 'document') for row in rows]

    async def _begin_transaction(self) -> None:
        """Begin transaction"""
        if not self._connection:
            self._connection = await self._pool.acquire()
        await self._connection.begin()

    async def _commit_transaction(self) -> None:
        """Commit transaction"""
        if self._connection:
            await self._connection.commit()
            self._connection = None

    async def _rollback_transaction(self) -> None:
        """Rollback transaction"""
        if self._connection:
            await self._connection.rollback()
            self._connection = None

    def _row_to_entity(self, row, entity_type: str):
        """Convert database row to entity object"""
        # Simplified implementation
        # In practice, you'd have proper entity classes
        return {
            'id': row[0],
            'title': row[1],
            'content': row[2],
            'created_at': row[3]
        }


class MilvusDAO(BaseDAO):
    """DAO for Milvus vector database operations"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Establish Milvus connection"""
        try:
            from pymilvus import connections

            connections.connect(
                alias="default",
                host=self.config.host,
                port=self.config.port
            )

            self._is_connected = True
            logger.info(f"Connected to Milvus: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise ConnectionException(f"Milvus connection failed: {e}")

    async def disconnect(self) -> None:
        """Close Milvus connection"""
        try:
            from pymilvus import connections
            connections.disconnect("default")
            self._is_connected = False
            logger.info("Disconnected from Milvus")
        except Exception as e:
            logger.error(f"Failed to disconnect from Milvus: {e}")

    async def create(self, vector_data: 'VectorData') -> 'VectorData':
        """Insert vector data"""
        if not self._is_connected:
            await self.connect()

        try:
            from pymilvus import Collection

            collection = Collection(self.config.database or "vectors")
            entities = [
                [vector_data.id],
                [vector_data.vector],
                [vector_data.metadata],
                [vector_data.document_id]
            ]

            collection.insert(entities)
            return vector_data

        except Exception as e:
            logger.error(f"Failed to insert vector: {e}")
            raise

    async def get_by_id(self, vector_id: Union[int, str]) -> Optional['VectorData']:
        """Get vector by ID"""
        # Implementation for retrieving vector by ID
        pass

    async def update(self, vector_id: Union[int, str], new_vector: List[float]) -> bool:
        """Update vector"""
        try:
            from pymilvus import Collection

            collection = Collection(self.config.database or "vectors")
            expr = f"id == {vector_id}"
            data = {
                "vector": new_vector
            }

            collection.update(data, expr)
            return True

        except Exception as e:
            logger.error(f"Failed to update vector {vector_id}: {e}")
            return False

    async def delete(self, vector_id: Union[int, str]) -> bool:
        """Delete vector"""
        try:
            from pymilvus import Collection

            collection = Collection(self.config.database or "vectors")
            expr = f"id == {vector_id}"
            collection.delete(expr)
            return True

        except Exception as e:
            logger.error(f"Failed to delete vector {vector_id}: {e}")
            return False

    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List['VectorData']:
        """List vectors"""
        # Implementation for listing vectors
        pass

    async def search(self, query_vector: List[float],
                    limit: int = 10,
                    filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search similar vectors"""
        try:
            from pymilvus import Collection

            collection = Collection(self.config.database or "vectors")
            collection.load()

            search_params = {
                "metric_type": "IP",  # Inner product
                "params": {"nprobe": 10}
            }

            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=limit,
                expr=None
            )

            # Convert results to dict format
            formatted_results = []
            for result in results[0]:
                formatted_results.append({
                    "id": result.id,
                    "distance": result.distance,
                    "score": result.score
                })

            return formatted_results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []

    async def batch_insert(self, vectors: List['VectorData']) -> bool:
        """Batch insert vectors"""
        try:
            from pymilvus import Collection

            collection = Collection(self.config.database or "vectors")
            entities = [
                [v.id for v in vectors],
                [v.vector for v in vectors],
                [v.metadata for v in vectors],
                [v.document_id for v in vectors]
            ]

            collection.insert(entities)
            return True

        except Exception as e:
            logger.error(f"Failed to batch insert vectors: {e}")
            return False

    async def _begin_transaction(self) -> None:
        """Milvus doesn't support traditional transactions"""
        pass

    async def _commit_transaction(self) -> None:
        """Milvus doesn't support traditional transactions"""
        pass

    async def _rollback_transaction(self) -> None:
        """Milvus doesn't support traditional transactions"""
        pass


class Neo4jDAO(BaseDAO):
    """DAO for Neo4j graph database operations"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._driver = None

    async def connect(self) -> None:
        """Establish Neo4j connection"""
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                self.config.host,
                auth=(self.config.username, self.config.password)
            )

            # Test connection
            with self._driver.session() as session:
                session.run("RETURN 1")

            self._is_connected = True
            logger.info(f"Connected to Neo4j: {self.config.host}")

        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionException(f"Neo4j connection failed: {e}")

    async def disconnect(self) -> None:
        """Close Neo4j connection"""
        if self._driver:
            self._driver.close()
            self._is_connected = False
            logger.info("Disconnected from Neo4j")

    async def create_document_node(self, document_id: int,
                                 title: str,
                                 metadata: Dict[str, Any]) -> bool:
        """Create document node"""
        try:
            with self._driver.session() as session:
                query = """
                CREATE (d:Document {
                    id: $document_id,
                    title: $title,
                    metadata: $metadata,
                    created_at: datetime()
                })
                """
                session.run(query, document_id=document_id,
                           title=title, metadata=metadata)
                return True

        except Exception as e:
            logger.error(f"Failed to create document node: {e}")
            return False

    async def create_entity(self, entity: 'Entity') -> 'Entity':
        """Create entity node"""
        try:
            with self._driver.session() as session:
                query = """
                CREATE (e:Entity {
                    id: $id,
                    name: $name,
                    type: $type,
                    properties: $properties,
                    created_at: datetime()
                })
                """
                session.run(query,
                           id=entity.id,
                           name=entity.name,
                           type=entity.type,
                           properties=entity.properties)
                return entity

        except Exception as e:
            logger.error(f"Failed to create entity: {e}")
            raise

    async def create_relationship(self, from_entity: Union[int, str],
                                to_entity: Union[int, str],
                                relationship_type: str,
                                properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create relationship between entities"""
        try:
            with self._driver.session() as session:
                query = f"""
                MATCH (a), (b)
                WHERE a.id = $from_entity AND b.id = $to_entity
                CREATE (a)-[r:{relationship_type}]->(b)
                SET r += $properties
                """
                session.run(query,
                           from_entity=from_entity,
                           to_entity=to_entity,
                           properties=properties or {})
                return True

        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False

    async def get_related_documents(self, document_id: int,
                                  depth: int = 2) -> List[int]:
        """Get related documents"""
        try:
            with self._driver.session() as session:
                query = f"""
                MATCH (d1:Document {{id: $document_id}})
                MATCH path = (d1)-[*1..{depth}]-(d2:Document)
                WHERE d1 <> d2
                RETURN DISTINCT d2.id as document_id
                """
                result = session.run(query, document_id=document_id)
                return [record["document_id"] for record in result]

        except Exception as e:
            logger.error(f"Failed to get related documents: {e}")
            return []

    async def get_related_entities(self, entity_id: Union[int, str],
                                 relationship_types: Optional[List[str]] = None,
                                 depth: int = 1) -> List['Entity']:
        """Get related entities"""
        # Implementation for getting related entities
        pass

    async def find_path(self, from_entity: Union[int, str],
                       to_entity: Union[int, str],
                       max_depth: int = 3) -> List[Dict[str, Any]]:
        """Find path between entities"""
        try:
            with self._driver.session() as session:
                query = f"""
                MATCH path = (a)-[*1..{max_depth}]-(b)
                WHERE a.id = $from_entity AND b.id = $to_entity
                RETURN path
                LIMIT 1
                """
                result = session.run(query,
                                   from_entity=from_entity,
                                   to_entity=to_entity)

                # Convert path to list of relationships
                for record in result:
                    path = record["path"]
                    return self._path_to_dict(path)

                return []

        except Exception as e:
            logger.error(f"Failed to find path: {e}")
            return []

    async def create(self, entity: Any) -> Any:
        """Create entity"""
        return await self.create_entity(entity)

    async def get_by_id(self, entity_id: Union[int, str]) -> Optional[Any]:
        """Get entity by ID"""
        try:
            with self._driver.session() as session:
                query = "MATCH (e) WHERE e.id = $entity_id RETURN e"
                result = session.run(query, entity_id=entity_id)
                record = result.single()

                if record:
                    return record["e"]
                return None

        except Exception as e:
            logger.error(f"Failed to get entity by ID: {e}")
            return None

    async def update(self, entity: Any) -> Any:
        """Update entity"""
        # Implementation for updating entity
        pass

    async def delete(self, entity_id: Union[int, str]) -> bool:
        """Delete entity"""
        try:
            with self._driver.session() as session:
                query = "MATCH (e) WHERE e.id = $entity_id DETACH DELETE e"
                session.run(query, entity_id=entity_id)
                return True

        except Exception as e:
            logger.error(f"Failed to delete entity: {e}")
            return False

    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Any]:
        """List entities"""
        # Implementation for listing entities
        pass

    async def _begin_transaction(self) -> None:
        """Begin transaction"""
        pass

    async def _commit_transaction(self) -> None:
        """Commit transaction"""
        pass

    async def _rollback_transaction(self) -> None:
        """Rollback transaction"""
        pass

    def _path_to_dict(self, path) -> List[Dict[str, Any]]:
        """Convert Neo4j path to dictionary"""
        # Implementation for converting path to dict
        return []


class RedisDAO(BaseDAO):
    """DAO for Redis operations"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._redis = None

    async def connect(self) -> None:
        """Establish Redis connection"""
        try:
            import aioredis

            self._redis = await aioredis.from_url(
                f"redis://{self.config.host}:{self.config.port}",
                **self.config.connection_params
            )

            await self._redis.ping()
            self._is_connected = True
            logger.info(f"Connected to Redis: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise ConnectionException(f"Redis connection failed: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection"""
        if self._redis:
            await self._redis.close()
            self._is_connected = False
            logger.info("Disconnected from Redis")

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair"""
        if not self._redis:
            await self.connect()

        try:
            import json
            serialized_value = json.dumps(value, default=str)

            if ttl:
                await self._redis.setex(key, ttl, serialized_value)
            else:
                await self._redis.set(key, serialized_value)

            return True

        except Exception as e:
            logger.error(f"Failed to set key {key}: {e}")
            return False

    async def get(self, key: str) -> Optional[Any]:
        """Get value by key"""
        if not self._redis:
            await self.connect()

        try:
            import json
            value = await self._redis.get(key)

            if value:
                return json.loads(value)
            return None

        except Exception as e:
            logger.error(f"Failed to get key {key}: {e}")
            return None

    async def delete(self, key: str) -> bool:
        """Delete key"""
        if not self._redis:
            await self.connect()

        try:
            result = await self._redis.delete(key)
            return result > 0

        except Exception as e:
            logger.error(f"Failed to delete key {key}: {e}")
            return False

    async def enqueue_task(self, task: 'Task') -> bool:
        """Enqueue task for processing"""
        try:
            import json
            task_data = {
                'id': task.id,
                'type': task.type,
                'data': task.data,
                'created_at': datetime.utcnow().isoformat()
            }

            await self._redis.lpush(
                "task_queue",
                json.dumps(task_data, default=str)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to enqueue task: {e}")
            return False

    async def get_pending_tasks(self, limit: int = 100) -> List['Task']:
        """Get pending tasks from queue"""
        try:
            import json

            # Use BRPOP for blocking pop
            tasks_data = await self._redis.lrange("task_queue", 0, limit - 1)
            tasks = []

            for task_data in tasks_data:
                task_dict = json.loads(task_data)
                # Convert to Task object
                task = Task(
                    id=task_dict['id'],
                    type=task_dict['type'],
                    data=task_dict['data']
                )
                tasks.append(task)

            # Remove tasks from queue
            if tasks_data:
                await self._redis.ltrim("task_queue", len(tasks_data), -1)

            return tasks

        except Exception as e:
            logger.error(f"Failed to get pending tasks: {e}")
            return []

    async def update_task_status(self, task_id: Union[int, str],
                                status: str,
                                result: Optional[Dict[str, Any]] = None,
                                error: Optional[str] = None) -> bool:
        """Update task status"""
        try:
            import json
            status_data = {
                'status': status,
                'updated_at': datetime.utcnow().isoformat()
            }

            if result:
                status_data['result'] = result
            if error:
                status_data['error'] = error

            await self._redis.hset(
                f"task_status:{task_id}",
                mapping=status_data
            )

            return True

        except Exception as e:
            logger.error(f"Failed to update task status: {e}")
            return False

    async def cancel_task(self, task_id: Union[int, str]) -> bool:
        """Cancel task"""
        # Implementation for cancelling task
        pass

    # BaseDAO interface implementation
    async def create(self, entity: Any) -> Any:
        """Generic create"""
        key = f"{entity.__class__.__name__.lower()}:{entity.id}"
        await self.set(key, entity.to_dict())
        return entity

    async def get_by_id(self, entity_id: Union[int, str]) -> Optional[Any]:
        """Generic get by ID"""
        # This would need to know the entity type
        return None

    async def update(self, entity: Any) -> Any:
        """Generic update"""
        key = f"{entity.__class__.__name__.lower()}:{entity.id}"
        await self.set(key, entity.to_dict())
        return entity

    async def delete(self, entity_id: Union[int, str]) -> bool:
        """Generic delete"""
        # This would need to know the entity type and key
        return False

    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Any]:
        """Generic list"""
        return []

    async def _begin_transaction(self) -> None:
        """Redis doesn't support traditional transactions"""
        pass

    async def _commit_transaction(self) -> None:
        """Redis doesn't support traditional transactions"""
        pass

    async def _rollback_transaction(self) -> None:
        """Redis doesn't support traditional transactions"""
        pass


class MinioDAO(BaseDAO):
    """DAO for MinIO object storage operations"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self._client = None

    async def connect(self) -> None:
        """Establish MinIO connection"""
        try:
            from minio import Minio

            self._client = Minio(
                f"{self.config.host}:{self.config.port}",
                access_key=self.config.username,
                secret_key=self.config.password,
                secure=self.config.connection_params.get('secure', False)
            )

            # Test connection by listing buckets
            self._client.list_buckets()
            self._is_connected = True
            logger.info(f"Connected to MinIO: {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to connect to MinIO: {e}")
            raise ConnectionException(f"MinIO connection failed: {e}")

    async def disconnect(self) -> None:
        """Close MinIO connection"""
        # MinIO client doesn't need explicit disconnection
        self._is_connected = False
        logger.info("Disconnected from MinIO")

    async def upload(self, key: str, content: bytes,
                    bucket: str = "documents") -> bool:
        """Upload object to MinIO"""
        if not self._client:
            await self.connect()

        try:
            from io import BytesIO

            # Ensure bucket exists
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)

            # Upload object
            self._client.put_object(
                bucket,
                key,
                BytesIO(content),
                length=len(content)
            )

            return True

        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            return False

    async def download(self, key: str,
                       bucket: str = "documents") -> Optional[bytes]:
        """Download object from MinIO"""
        if not self._client:
            await self.connect()

        try:
            response = self._client.get_object(bucket, key)
            content = response.read()
            response.close()
            response.release_conn()

            return content

        except Exception as e:
            logger.error(f"Failed to download {key}: {e}")
            return None

    async def delete(self, key: str,
                     bucket: str = "documents") -> bool:
        """Delete object from MinIO"""
        if not self._client:
            await self.connect()

        try:
            self._client.remove_object(bucket, key)
            return True

        except Exception as e:
            logger.error(f"Failed to delete {key}: {e}")
            return False

    async def delete_prefix(self, prefix: str,
                           bucket: str = "documents") -> bool:
        """Delete objects with prefix"""
        if not self._client:
            await self.connect()

        try:
            objects = self._client.list_objects(bucket, prefix=prefix)
            for obj in objects:
                self._client.remove_object(bucket, obj.object_name)
            return True

        except Exception as e:
            logger.error(f"Failed to delete prefix {prefix}: {e}")
            return False

    async def list_objects(self, prefix: str = "",
                          bucket: str = "documents") -> List[str]:
        """List objects"""
        if not self._client:
            await self.connect()

        try:
            objects = self._client.list_objects(bucket, prefix=prefix)
            return [obj.object_name for obj in objects]

        except Exception as e:
            logger.error(f"Failed to list objects: {e}")
            return []

    # BaseDAO interface implementation
    async def create(self, entity: Any) -> Any:
        """Generic create"""
        return entity

    async def get_by_id(self, entity_id: Union[int, str]) -> Optional[Any]:
        """Generic get by ID"""
        return None

    async def update(self, entity: Any) -> Any:
        """Generic update"""
        return entity

    async def delete(self, entity_id: Union[int, str]) -> bool:
        """Generic delete"""
        return False

    async def list(self, filters: Optional[Dict[str, Any]] = None,
                   limit: Optional[int] = None,
                   offset: Optional[int] = None) -> List[Any]:
        """Generic list"""
        return []

    async def _begin_transaction(self) -> None:
        """MinIO doesn't support transactions"""
        pass

    async def _commit_transaction(self) -> None:
        """MinIO doesn't support transactions"""
        pass

    async def _rollback_transaction(self) -> None:
        """MinIO doesn't support transactions"""
        pass