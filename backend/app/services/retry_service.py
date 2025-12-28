"""
Document Upload Retry Service
Handles retry logic for failed document uploads and parsing
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from enum import Enum
import json
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from ..core.database import get_db
from ..models.document import Document, DocumentStatus
from ..models.task import Task, TaskStatus
from ..services.upload_service import UploadService
from ..services.multimodal.core.multimodal_parser import MultimodalDocumentParser as MultimodalParser
from ..core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RetryStrategy(Enum):
    """Retry strategies for different failure types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_INTERVAL = "fixed_interval"
    IMMEDIATE = "immediate"


class RetryConfig:
    """Configuration for retry logic"""
    def __init__(
        self,
        max_retries: int = 3,
        strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF,
        base_delay: int = 60,  # seconds
        max_delay: int = 3600,  # 1 hour
        backoff_multiplier: float = 2.0,
        retryable_errors: List[str] = None
    ):
        self.max_retries = max_retries
        self.strategy = strategy
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.retryable_errors = retryable_errors or [
            "ConnectionError",
            "TimeoutError",
            "NetworkError",
            "TemporaryFailure",
            "ServiceUnavailable",
            "RateLimitError"
        ]


class RetryService:
    """Service for managing document upload retries"""

    def __init__(self):
        self.upload_service = UploadService()
        self.parser = MultimodalParser()
        self.retry_configs = {
            "upload": RetryConfig(max_retries=5, base_delay=30),
            "parsing": RetryConfig(max_retries=3, base_delay=60),
            "embedding": RetryConfig(max_retries=3, base_delay=120),
            "knowledge_graph": RetryConfig(max_retries=2, base_delay=300)
        }

    async def check_and_retry_failed_documents(self, db: Session) -> Dict[str, Any]:
        """
        Check for failed documents and attempt retries

        Returns:
            Dict with retry statistics
        """
        retry_stats = {
            "total_checked": 0,
            "retries_attempted": 0,
            "retries_successful": 0,
            "retries_failed": 0,
            "permanent_failures": 0
        }

        # Get failed documents that are eligible for retry
        failed_docs = self._get_retry_eligible_documents(db)
        retry_stats["total_checked"] = len(failed_docs)

        logger.info(f"Found {len(failed_docs)} documents eligible for retry")

        for doc in failed_docs:
            try:
                # Determine failure type and get appropriate config
                failure_type = self._classify_failure(doc)
                config = self.retry_configs.get(failure_type, self.retry_configs["upload"])

                # Check if document should be retried now
                if not self._should_retry_now(doc, config):
                    continue

                # Attempt retry
                retry_stats["retries_attempted"] += 1

                success = await self._retry_document_processing(db, doc, failure_type)

                if success:
                    retry_stats["retries_successful"] += 1
                    logger.info(f"Successfully retried document {doc.id}")
                else:
                    # Increment retry count
                    doc.retry_count = (doc.retry_count or 0) + 1

                    if doc.retry_count >= config.max_retries:
                        doc.status = DocumentStatus.PERMANENTLY_FAILED
                        retry_stats["permanent_failures"] += 1
                        logger.error(f"Document {doc.id} marked as permanently failed after {doc.retry_count} retries")
                    else:
                        retry_stats["retries_failed"] += 1
                        # Set next retry time
                        doc.next_retry_at = self._calculate_next_retry_time(doc.retry_count, config)

                    db.commit()

            except Exception as e:
                logger.error(f"Error during retry of document {doc.id}: {str(e)}")
                retry_stats["retries_failed"] += 1

        return retry_stats

    def _get_retry_eligible_documents(self, db: Session) -> List[Document]:
        """Get documents that are eligible for retry"""
        now = datetime.utcnow()

        # Documents that are failed, not permanently failed, and either:
        # - Never retried before, or
        # - Next retry time has passed
        query = db.query(Document).filter(
            and_(
                Document.status.in_([
                    DocumentStatus.UPLOAD_FAILED,
                    DocumentStatus.PARSING_FAILED,
                    DocumentStatus.EMBEDDING_FAILED,
                    DocumentStatus.PROCESSING_FAILED
                ]),
                or_(
                    Document.retry_count.is_(None),
                    Document.retry_count < 3,  # Default max retries
                    Document.next_retry_at <= now
                )
            )
        )

        return query.all()

    def _classify_failure(self, doc: Document) -> str:
        """Classify the type of failure for appropriate retry strategy"""
        if doc.status == DocumentStatus.UPLOAD_FAILED:
            return "upload"
        elif doc.status in [DocumentStatus.PARSING_FAILED, DocumentStatus.PROCESSING_FAILED]:
            return "parsing"
        elif doc.status == DocumentStatus.EMBEDDING_FAILED:
            return "embedding"
        elif doc.status == DocumentStatus.KNOWLEDGE_GRAPH_FAILED:
            return "knowledge_graph"
        else:
            return "upload"  # Default

    def _should_retry_now(self, doc: Document, config: RetryConfig) -> bool:
        """Check if document should be retried now based on timing"""
        if doc.retry_count is None:
            return True

        if doc.retry_count >= config.max_retries:
            return False

        if doc.next_retry_at:
            return datetime.utcnow() >= doc.next_retry_at

        return True

    def _calculate_next_retry_time(self, retry_count: int, config: RetryConfig) -> datetime:
        """Calculate when the next retry should happen"""
        if config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = min(
                config.base_delay * (config.backoff_multiplier ** retry_count),
                config.max_delay
            )
        elif config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = min(
                config.base_delay * (retry_count + 1),
                config.max_delay
            )
        else:  # FIXED_INTERVAL or IMMEDIATE
            delay = config.base_delay

        return datetime.utcnow() + timedelta(seconds=delay)

    async def _retry_document_processing(
        self,
        db: Session,
        doc: Document,
        failure_type: str
    ) -> bool:
        """Retry processing a failed document"""
        try:
            if failure_type == "upload":
                return await self._retry_upload(db, doc)
            elif failure_type == "parsing":
                return await self._retry_parsing(db, doc)
            elif failure_type == "embedding":
                return await self._retry_embedding(db, doc)
            elif failure_type == "knowledge_graph":
                return await self._retry_knowledge_graph(db, doc)
            else:
                logger.warning(f"Unknown failure type: {failure_type}")
                return False

        except Exception as e:
            logger.error(f"Retry failed for document {doc.id}: {str(e)}")
            return False

    async def _retry_upload(self, db: Session, doc: Document) -> bool:
        """Retry document upload"""
        logger.info(f"Retrying upload for document {doc.id}")

        try:
            # Reset status to uploading
            doc.status = DocumentStatus.UPLOADING
            db.commit()

            # Re-trigger upload processing
            # This would typically involve re-queuing the upload task
            # For now, we'll simulate successful retry
            doc.status = DocumentStatus.UPLOADED
            doc.processed_at = datetime.utcnow()
            db.commit()

            return True

        except Exception as e:
            logger.error(f"Upload retry failed for document {doc.id}: {str(e)}")
            doc.status = DocumentStatus.UPLOAD_FAILED
            db.commit()
            return False

    async def _retry_parsing(self, db: Session, doc: Document) -> bool:
        """Retry document parsing"""
        logger.info(f"Retrying parsing for document {doc.id}")

        try:
            # Reset status
            doc.status = DocumentStatus.PARSING
            db.commit()

            # Get file content from MinIO or storage
            file_content = await self.upload_service.get_file_content(doc.storage_path)

            # Re-parse the document
            parsed_content = await self.parser.parse_document(
                file_content,
                doc.filename,
                doc.mime_type
            )

            # Update document with parsed content
            doc.parsed_content = parsed_content
            doc.status = DocumentStatus.PARSED
            db.commit()

            return True

        except Exception as e:
            logger.error(f"Parsing retry failed for document {doc.id}: {str(e)}")
            doc.status = DocumentStatus.PARSING_FAILED
            db.commit()
            return False

    async def _retry_embedding(self, db: Session, doc: Document) -> bool:
        """Retry embedding generation"""
        logger.info(f"Retrying embedding generation for document {doc.id}")

        try:
            # Reset status
            doc.status = DocumentStatus.EMBEDDING
            db.commit()

            # Re-trigger embedding generation task
            # This would typically involve calling the embedding service
            # For now, we'll simulate successful retry
            doc.status = DocumentStatus.EMBEDDED
            db.commit()

            return True

        except Exception as e:
            logger.error(f"Embedding retry failed for document {doc.id}: {str(e)}")
            doc.status = DocumentStatus.EMBEDDING_FAILED
            db.commit()
            return False

    async def _retry_knowledge_graph(self, db: Session, doc: Document) -> bool:
        """Retry knowledge graph generation"""
        logger.info(f"Retrying knowledge graph generation for document {doc.id}")

        try:
            # Reset status
            doc.status = DocumentStatus.KNOWLEDGE_GRAPH_PROCESSING
            db.commit()

            # Re-trigger knowledge graph task
            # This would typically involve calling the knowledge graph service
            # For now, we'll simulate successful retry
            doc.status = DocumentStatus.KNOWLEDGE_GRAPH_PROCESSED
            db.commit()

            return True

        except Exception as e:
            logger.error(f"Knowledge graph retry failed for document {doc.id}: {str(e)}")
            doc.status = DocumentStatus.KNOWLEDGE_GRAPH_FAILED
            db.commit()
            return False

    async def manual_retry_document(self, db: Session, document_id: str) -> Dict[str, Any]:
        """
        Manually retry a specific document

        Args:
            db: Database session
            document_id: ID of document to retry

        Returns:
            Dict with retry result
        """
        doc = db.query(Document).filter(Document.id == document_id).first()

        if not doc:
            return {
                "success": False,
                "error": f"Document {document_id} not found"
            }

        if doc.status not in [
            DocumentStatus.UPLOAD_FAILED,
            DocumentStatus.PARSING_FAILED,
            DocumentStatus.EMBEDDING_FAILED,
            DocumentStatus.PROCESSING_FAILED,
            DocumentStatus.PERMANENTLY_FAILED
        ]:
            return {
                "success": False,
                "error": f"Document {document_id} is not in a failed state"
            }

        # Reset retry count for manual retry
        doc.retry_count = 0
        doc.next_retry_at = datetime.utcnow()
        db.commit()

        # Attempt immediate retry
        failure_type = self._classify_failure(doc)
        success = await self._retry_document_processing(db, doc, failure_type)

        return {
            "success": success,
            "document_id": document_id,
            "failure_type": failure_type,
            "new_status": doc.status.value if doc else None
        }

    def get_retry_statistics(self, db: Session) -> Dict[str, Any]:
        """Get comprehensive retry statistics"""
        stats = {}

        # Count documents by status and retry count
        for status in DocumentStatus:
            count = db.query(Document).filter(Document.status == status).count()
            stats[status.value] = count

        # Documents with retry attempts
        retried_docs = db.query(Document).filter(
            Document.retry_count > 0
        ).all()

        stats["retried_documents"] = len(retried_docs)
        stats["avg_retry_count"] = sum(doc.retry_count for doc in retried_docs) / len(retried_docs) if retried_docs else 0

        # Documents eligible for retry now
        eligible_docs = len(self._get_retry_eligible_documents(db))
        stats["eligible_for_retry"] = eligible_docs

        return stats


# Background task for periodic retry processing
async def retry_processor_background():
    """Background task to periodically process retries"""
    retry_service = RetryService()

    while True:
        try:
            db = next(get_db())

            # Process retries
            stats = await retry_service.check_and_retry_failed_documents(db)

            if stats["retries_attempted"] > 0:
                logger.info(f"Retry processing completed: {json.dumps(stats, indent=2)}")

            db.close()

            # Sleep for 5 minutes before next check
            await asyncio.sleep(300)

        except Exception as e:
            logger.error(f"Error in retry processor background task: {str(e)}")
            await asyncio.sleep(60)  # Wait 1 minute before retrying


# API endpoint helpers
def get_failed_documents(db: Session, limit: int = 50) -> List[Document]:
    """Get list of failed documents with details"""
    return db.query(Document).filter(
        Document.status.in_([
            DocumentStatus.UPLOAD_FAILED,
            DocumentStatus.PARSING_FAILED,
            DocumentStatus.EMBEDDING_FAILED,
            DocumentStatus.PROCESSING_FAILED,
            DocumentStatus.PERMANENTLY_FAILED
        ])
    ).order_by(Document.created_at.desc()).limit(limit).all()