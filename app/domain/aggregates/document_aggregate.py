"""
Document Aggregate Root

Aggregate root for the Document aggregate, managing related entities
and enforcing business rules at the aggregate level.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.document import Document
from ..value_objects import DocumentStatus, TextChunk, EmbeddingVector, ConfidenceScore
from ..events import DomainEvent


class DocumentAggregate:
    """
    Document aggregate root

    Manages the document entity and its related chunks, embeddings,
    and processing state. Enforces business rules at the aggregate level.
    """

    def __init__(self, document: Document):
        self._document = document
        self._chunks: List[TextChunk] = []
        self._embeddings: List[EmbeddingVector] = []
        self._domain_events: List[DomainEvent] = []

    @property
    def document(self) -> Document:
        """Get the document entity"""
        return self._document

    @property
    def chunks(self) -> List[TextChunk]:
        """Get document chunks"""
        return self._chunks.copy()

    @property
    def embeddings(self) -> List[EmbeddingVector]:
        """Get document embeddings"""
        return self._embeddings.copy()

    @property
    def domain_events(self) -> List[DomainEvent]:
        """Get pending domain events"""
        return self._domain_events.copy()

    def clear_domain_events(self):
        """Clear pending domain events"""
        self._domain_events.clear()

    # Aggregate Business Operations

    def start_document_processing(self) -> None:
        """Start processing the document"""
        if self._document.status != DocumentStatus.UPLOADED:
            raise ValueError(f"Document must be uploaded to start processing, current status: {self._document.status}")

        self._document.start_processing()
        self._add_domain_event(DomainEvent(
            type="document_processing_started",
            aggregate_id=self._document.id,
            data={
                "document_title": self._document.title,
                "file_name": self._document.file_name,
                "content_type": self._document.content_type.value
            }
        ))

    def add_text_chunk(self, content: str, chunk_index: Optional[int] = None,
                      page_number: Optional[int] = None,
                      section_title: Optional[str] = None) -> TextChunk:
        """Add text chunk to document"""
        if self._document.status not in [DocumentStatus.PROCESSING, DocumentStatus.PROCESSED]:
            raise ValueError("Cannot add chunks to document that is not being processed")

        # Auto-generate chunk index if not provided
        if chunk_index is None:
            chunk_index = len(self._chunks)

        # Create chunk
        chunk = TextChunk(
            content=content,
            chunk_index=chunk_index,
            start_char=0,  # Will be calculated if needed
            end_char=len(content),
            page_number=page_number,
            section_title=section_title
        )

        # Validate chunk
        if chunk_index >= 0 and chunk_index < len(self._chunks):
            # Replace existing chunk
            self._chunks[chunk_index] = chunk
        else:
            # Append new chunk
            self._chunks.append(chunk)

        # Update document
        self._document.add_chunk(chunk)

        self._add_domain_event(DomainEvent(
            type="text_chunk_added",
            aggregate_id=self._document.id,
            data={
                "chunk_index": chunk_index,
                "content_length": len(content),
                "page_number": page_number
            }
        ))

        return chunk

    def add_embedding(self, chunk_index: int, embedding: EmbeddingVector) -> None:
        """Add embedding for a specific chunk"""
        if not 0 <= chunk_index < len(self._chunks):
            raise ValueError(f"Invalid chunk index: {chunk_index}")

        # Check if embedding already exists for this chunk
        existing_embedding = next((e for e in self._embeddings if e.vector == embedding.vector), None)

        if existing_embedding is None:
            self._embeddings.append(embedding)

        # Update chunk with embedding reference
        chunk = self._chunks[chunk_index]
        # Note: Since TextChunk is frozen, we would need to create a new instance
        # or modify the implementation to allow updates

        self._document.embedding_count = len(self._embeddings)

        self._add_domain_event(DomainEvent(
            type="embedding_added",
            aggregate_id=self._document.id,
            data={
                "chunk_index": chunk_index,
                "model_name": embedding.model_name,
                "dimension": embedding.dimension
            }
        ))

    def complete_processing(self, quality_score: Optional[float] = None,
                           confidence: float = 1.0) -> None:
        """Complete document processing"""
        if self._document.status != DocumentStatus.PROCESSING:
            raise ValueError("Document is not currently processing")

        # Validate processing completeness
        if len(self._chunks) == 0:
            raise ValueError("Cannot complete processing with no text chunks")

        # Set quality score if provided
        if quality_score is not None:
            self._document.set_quality_score(quality_score, confidence)

        # Mark as complete
        self._document.complete_processing()

        # Update counts
        self._document.entity_count = len([c for c in self._chunks if c.metadata.get("has_entities", False)])

        self._add_domain_event(DomainEvent(
            type="document_processing_completed",
            aggregate_id=self._document.id,
            data={
                "chunk_count": len(self._chunks),
                "embedding_count": len(self._embeddings),
                "quality_score": quality_score,
                "processing_duration": self._document.get_processing_duration()
            }
        ))

    def fail_processing(self, error_message: str) -> None:
        """Mark document processing as failed"""
        if self._document.status != DocumentStatus.PROCESSING:
            raise ValueError("Document is not currently processing")

        self._document.fail_processing(error_message)

        self._add_domain_event(DomainEvent(
            type="document_processing_failed",
            aggregate_id=self._document.id,
            data={
                "error_message": error_message,
                "chunks_processed": len(self._chunks),
                "embeddings_generated": len(self._embeddings)
            }
        ))

    def update_processing_progress(self, progress: float, current_step: str) -> None:
        """Update processing progress"""
        self._document.update_progress(progress, current_step)

        if progress % 25 == 0:  # Emit event at 25%, 50%, 75%, 100%
            self._add_domain_event(DomainEvent(
                type="processing_progress_updated",
                aggregate_id=self._document.id,
                data={
                    "progress": progress,
                    "current_step": current_step,
                    "chunks_processed": len(self._chunks)
                }
            ))

    def analyze_document_quality(self) -> ConfidenceScore:
        """Analyze and set document quality score"""
        if len(self._chunks) == 0:
            quality_score = 0.0
        else:
            # Simple quality calculation based on various factors
            factors = {
                "content_coverage": min(len(self._chunks) / 10.0, 1.0),  # More chunks = better coverage
                "embedding_completeness": len(self._embeddings) / len(self._chunks) if self._chunks else 0,
                "average_chunk_length": self._get_average_chunk_length() / 1000.0,  # Normalized to 1000 chars
                "has_metadata": 1.0 if self._document.metadata.custom_fields else 0.5,
            }

            # Weighted average
            weights = {"content_coverage": 0.3, "embedding_completeness": 0.3, "average_chunk_length": 0.2, "has_metadata": 0.2}
            quality_score = sum(factors[k] * weights[k] for k in factors)

        confidence = ConfidenceScore(quality_score)
        self._document.set_quality_score(quality_score)
        return confidence

    def can_be_queried(self) -> bool:
        """Check if document can be used for queries"""
        return (
            self._document.is_processing_complete() and
            len(self._chunks) > 0 and
            len(self._embeddings) > 0
        )

    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results"""
        return {
            "document_id": self._document.id,
            "title": self._document.title,
            "status": self._document.status.value,
            "processing_progress": self._document.processing_progress,
            "chunk_count": len(self._chunks),
            "embedding_count": len(self._embeddings),
            "total_characters": sum(chunk.get_length() for chunk in self._chunks),
            "total_words": sum(chunk.get_word_count() for chunk in self._chunks),
            "average_chunk_length": self._get_average_chunk_length(),
            "quality_score": self._document.quality_score.value if self._document.quality_score else None,
            "processing_steps": self._document.processing_steps,
            "error_message": self._document.last_error,
            "created_at": self._document.created_at.isoformat(),
            "processed_at": self._document.processed_at.isoformat() if self._document.processed_at else None
        }

    # Aggregate Helper Methods

    def _get_average_chunk_length(self) -> float:
        """Get average length of chunks"""
        if not self._chunks:
            return 0.0
        return sum(chunk.get_length() for chunk in self._chunks) / len(self._chunks)

    def _add_domain_event(self, event: DomainEvent) -> None:
        """Add domain event to be published"""
        self._add_domain_event(event)

    # Business Rule Enforcement

    def validate_business_rules(self) -> List[str]:
        """Validate all business rules and return list of violations"""
        violations = []

        # Rule: Document must have title or file name
        if not self._document.title and not self._document.file_name:
            violations.append("Document must have either title or file name")

        # Rule: Processing progress must be valid
        if self._document.processing_progress < 0 or self._document.processing_progress > 100:
            violations.append("Processing progress must be between 0 and 100")

        # Rule: Chunks must have valid indices
        expected_indices = set(range(len(self._chunks)))
        actual_indices = set(chunk.chunk_index for chunk in self._chunks)
        if expected_indices != actual_indices:
            violations.append("Chunk indices are not sequential or have gaps")

        # Rule: All chunks should have content
        if any(not chunk.content.strip() for chunk in self._chunks):
            violations.append("All chunks must have non-empty content")

        # Rule: If marked as processed, should have quality assessment
        if self._document.status == DocumentStatus.PROCESSED and not self._document.quality_score:
            violations.append("Processed document must have quality score")

        return violations

    def is_consistent(self) -> bool:
        """Check if aggregate is in a consistent state"""
        violations = self.validate_business_rules()
        return len(violations) == 0

    # Factory Methods

    @classmethod
    def create_new(cls, title: str, file_name: str, content_type: str, file_path: Optional[str] = None) -> 'DocumentAggregate':
        """Create new document aggregate"""
        from ..value_objects import detect_content_type

        document = Document(
            title=title,
            file_name=file_name,
            content_type=detect_content_type(file_name),
            file_path=file_path,
            status=DocumentStatus.UPLOADED
        )

        aggregate = cls(document)
        aggregate._add_domain_event(DomainEvent(
            type="document_created",
            aggregate_id=document.id,
            data={
                "title": title,
                "file_name": file_name,
                "content_type": content_type
            }
        ))

        return aggregate

    @classmethod
    def from_existing(cls, document: Document, chunks: List[TextChunk] = None,
                     embeddings: List[EmbeddingVector] = None) -> 'DocumentAggregate':
        """Create aggregate from existing document"""
        aggregate = cls(document)
        aggregate._chunks = chunks or []
        aggregate._embeddings = embeddings or []

        # Update document with chunks
        for chunk in aggregate._chunks:
            document.add_chunk(chunk)

        document.embedding_count = len(aggregate._embeddings)

        return aggregate