"""
Document Entity

Core domain entity representing a document in the financial RAG system.
Contains business rules and invariants related to documents.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from ..value_objects import (
    DocumentStatus,
    ContentType,
    TextChunk,
    Metadata,
    ConfidenceScore
)


class DocumentType(str, Enum):
    """Document types supported in the system"""
    PDF = "pdf"
    WORD = "docx"
    EXCEL = "xlsx"
    POWERPOINT = "pptx"
    TEXT = "txt"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class Document:
    """
    Document entity representing financial documents
    """
    # Identity
    id: Optional[int] = None
    file_id: str = field(default_factory=lambda: f"doc_{datetime.utcnow().timestamp()}")

    # Basic Information
    title: str = ""
    file_path: Optional[str] = None
    file_name: str = ""
    content_type: ContentType = ContentType.PDF
    file_size: int = 0
    checksum: Optional[str] = None

    # Content
    raw_content: Optional[str] = None
    processed_content: Optional[str] = None
    chunks: List[TextChunk] = field(default_factory=list)

    # Metadata
    metadata: Metadata = field(default_factory=Metadata)
    source: str = "manual_upload"
    author: Optional[str] = None
    created_by: Optional[int] = None  # User ID

    # Processing State
    status: DocumentStatus = DocumentStatus.UPLOADING
    processing_progress: float = 0.0
    processing_steps: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    last_error: Optional[str] = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    # Analysis Results
    embedding_count: int = 0
    entity_count: int = 0
    quality_score: Optional[ConfidenceScore] = None
    is_confidential: bool = False
    tags: List[str] = field(default_factory=list)

    # Financial Specific
    document_type: DocumentType = DocumentType.PDF
    financial_data: Dict[str, Any] = field(default_factory=dict)
    company_symbols: List[str] = field(default_factory=list)
    report_date: Optional[datetime] = None
    period: Optional[str] = None  # Q1, Q2, Q3, Q4, Annual, etc.

    def __post_init__(self):
        """Post-initialization validation"""
        self.validate()

    def validate(self):
        """Validate document invariants"""
        if not self.title and not self.file_name:
            raise ValueError("Document must have either title or file name")

        if self.processing_progress < 0 or self.processing_progress > 100:
            raise ValueError("Processing progress must be between 0 and 100")

        if self.file_size < 0:
            raise ValueError("File size cannot be negative")

    # Business Logic Methods

    def start_processing(self) -> None:
        """Start document processing"""
        if self.status != DocumentStatus.UPLOADED:
            raise ValueError(f"Cannot start processing document in status: {self.status}")

        self.status = DocumentStatus.PROCESSING
        self.processing_progress = 0.0
        self.processing_steps = []
        self.error_message = None
        self.last_error = None
        self.updated_at = datetime.utcnow()

    def update_progress(self, progress: float, step: Optional[str] = None) -> None:
        """Update processing progress"""
        if self.status != DocumentStatus.PROCESSING:
            raise ValueError("Cannot update progress for non-processing document")

        self.processing_progress = max(0, min(100, progress))
        self.updated_at = datetime.utcnow()

        if step:
            if step not in self.processing_steps:
                self.processing_steps.append(step)
            else:
                # Update existing step timestamp
                pass

    def complete_processing(self) -> None:
        """Mark document processing as complete"""
        if self.status != DocumentStatus.PROCESSING:
            raise ValueError(f"Cannot complete document in status: {self.status}")

        self.status = DocumentStatus.PROCESSED
        self.processing_progress = 100.0
        self.processed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

    def fail_processing(self, error_message: str) -> None:
        """Mark document processing as failed"""
        self.status = DocumentStatus.FAILED
        self.last_error = error_message
        self.processing_progress = 0.0
        self.updated_at = datetime.utcnow()

    def add_chunk(self, chunk: TextChunk) -> None:
        """Add text chunk to document"""
        self.chunks.append(chunk)
        self.updated_at = datetime.utcnow()

    def remove_chunk(self, chunk_index: int) -> None:
        """Remove text chunk from document"""
        if 0 <= chunk_index < len(self.chunks):
            del self.chunks[chunk_index]
            self.updated_at = datetime.utcnow()

    def update_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """Update document metadata"""
        self.metadata.update(new_metadata)
        self.updated_at = datetime.utcnow()

    def add_tag(self, tag: str) -> None:
        """Add tag to document"""
        if tag not in self.tags:
            self.tags.append(tag)
            self.updated_at = datetime.utcnow()

    def remove_tag(self, tag: str) -> None:
        """Remove tag from document"""
        if tag in self.tags:
            self.tags.remove(tag)
            self.updated_at = datetime.utcnow()

    def add_company_symbol(self, symbol: str) -> None:
        """Add company symbol to document"""
        symbol = symbol.upper()
        if symbol not in self.company_symbols:
            self.company_symbols.append(symbol)
            self.updated_at = datetime.utcnow()

    def set_quality_score(self, score: float, confidence: float = 1.0) -> None:
        """Set document quality score"""
        self.quality_score = ConfidenceScore(score, confidence)
        self.updated_at = datetime.utcnow()

    def mark_confidential(self, is_confidential: bool = True) -> None:
        """Mark document as confidential"""
        self.is_confidential = is_confidential
        self.updated_at = datetime.utcnow()

    # Query Methods

    def get_total_chunks(self) -> int:
        """Get total number of chunks"""
        return len(self.chunks)

    def get_total_characters(self) -> int:
        """Get total character count from chunks"""
        return sum(len(chunk.content) for chunk in self.chunks)

    def get_chunk_by_index(self, index: int) -> Optional[TextChunk]:
        """Get chunk by index"""
        if 0 <= index < len(self.chunks):
            return self.chunks[index]
        return None

    def has_embedding(self) -> bool:
        """Check if document has embeddings"""
        return any(chunk.embedding is not None for chunk in self.chunks)

    def get_processing_duration(self) -> Optional[float]:
        """Get processing duration in seconds"""
        if self.processed_at and self.created_at:
            return (self.processed_at - self.created_at).total_seconds()
        return None

    def is_processing_complete(self) -> bool:
        """Check if processing is complete"""
        return self.status == DocumentStatus.PROCESSED

    def has_processing_failed(self) -> bool:
        """Check if processing has failed"""
        return self.status == DocumentStatus.FAILED

    def can_be_queried(self) -> bool:
        """Check if document can be used for queries"""
        return self.is_processing_complete() and len(self.chunks) > 0

    # Domain Events

    def get_domain_events(self) -> List[Dict[str, Any]]:
        """Get domain events that need to be published"""
        events = []
        current_time = datetime.utcnow()

        # Document creation event
        if self.status == DocumentStatus.UPLOADED:
            events.append({
                "type": "document_created",
                "document_id": self.id,
                "title": self.title,
                "content_type": self.content_type.value,
                "file_size": self.file_size,
                "created_by": self.created_by,
                "timestamp": current_time
            })

        # Document processing complete event
        if self.status == DocumentStatus.PROCESSED:
            events.append({
                "type": "document_processed",
                "document_id": self.id,
                "chunk_count": len(self.chunks),
                "quality_score": self.quality_score.value if self.quality_score else None,
                "processing_duration": self.get_processing_duration(),
                "timestamp": current_time
            })

        # Document processing failed event
        if self.status == DocumentStatus.FAILED:
            events.append({
                "type": "document_processing_failed",
                "document_id": self.id,
                "error_message": self.last_error,
                "timestamp": current_time
            })

        return events

    # Business Rules

    def can_be_deleted(self) -> bool:
        """Check if document can be deleted"""
        # Business rule: Documents being processed cannot be deleted
        return self.status != DocumentStatus.PROCESSING

    def can_be_edited(self) -> bool:
        """Check if document can be edited"""
        # Business rule: Only uploaded or failed documents can be edited
        return self.status in [DocumentStatus.UPLOADED, DocumentStatus.FAILED]

    def requires_reprocessing(self) -> bool:
        """Check if document needs reprocessing"""
        # Business rule: Documents with errors or low quality need reprocessing
        if self.status == DocumentStatus.FAILED:
            return True

        if self.quality_score and self.quality_score.value < 0.5:
            return True

        if not self.has_embedding() and self.content_type in [ContentType.PDF, ContentType.TEXT]:
            return True

        return False

    def is_financial_report(self) -> bool:
        """Check if document is a financial report"""
        financial_keywords = ["report", "annual", "quarterly", "financial", "earnings", "revenue", "profit"]
        title_lower = self.title.lower()
        return any(keyword in title_lower for keyword in financial_keywords)

    def get_summary(self) -> Dict[str, Any]:
        """Get document summary for API responses"""
        return {
            "id": self.id,
            "title": self.title,
            "file_name": self.file_name,
            "content_type": self.content_type.value,
            "file_size": self.file_size,
            "status": self.status.value,
            "processing_progress": self.processing_progress,
            "chunk_count": len(self.chunks),
            "quality_score": self.quality_score.value if self.quality_score else None,
            "is_confidential": self.is_confidential,
            "tags": self.tags,
            "company_symbols": self.company_symbols,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None
        }