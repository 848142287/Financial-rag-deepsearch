"""
Document Value Objects

Immutable value objects related to documents.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import re


class DocumentStatus(str, Enum):
    """Document processing status"""
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"
    DELETED = "deleted"


class ContentType(str, Enum):
    """Content types for documents"""
    PDF = "application/pdf"
    WORD = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    EXCEL = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    POWERPOINT = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    TEXT = "text/plain"
    HTML = "text/html"
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    AUDIO_MP3 = "audio/mpeg"
    VIDEO_MP4 = "video/mp4"
    JSON = "application/json"


@dataclass(frozen=True)
class TextChunk:
    """A chunk of text with metadata"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)
    embedding: Optional[List[float]] = None

    def __post_init__(self):
        """Validate chunk invariants"""
        if self.chunk_index < 0:
            raise ValueError("Chunk index cannot be negative")

        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("Character positions cannot be negative")

        if self.end_char < self.start_char:
            raise ValueError("End character must be after start character")

        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty")

    def get_length(self) -> int:
        """Get length of content"""
        return len(self.content)

    def get_word_count(self) -> int:
        """Get word count"""
        return len(self.content.split())

    def contains_text(self, text: str) -> bool:
        """Check if chunk contains specific text"""
        return text.lower() in self.content.lower()

    def has_embedding(self) -> bool:
        """Check if chunk has embedding"""
        return self.embedding is not None and len(self.embedding) > 0

    def get_summary(self) -> Dict[str, any]:
        """Get chunk summary"""
        return {
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "length": self.get_length(),
            "word_count": self.get_word_count(),
            "has_embedding": self.has_embedding(),
            "preview": self.content[:100] + "..." if len(self.content) > 100 else self.content
        }


@dataclass(frozen=True)
class EmbeddingVector:
    """Embedding vector with metadata"""
    vector: List[float]
    model_name: str
    dimension: int
    created_at: datetime = field(default_factory=datetime.utcnow)
    similarity_threshold: float = 0.7

    def __post_init__(self):
        """Validate embedding invariants"""
        if not self.vector:
            raise ValueError("Embedding vector cannot be empty")

        if self.dimension != len(self.vector):
            raise ValueError(f"Dimension mismatch: expected {self.dimension}, got {len(self.vector)}")

        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("Similarity threshold must be between 0 and 1")

    def get_norm(self) -> float:
        """Calculate vector norm"""
        return sum(x ** 2 for x in self.vector) ** 0.5

    def normalize(self) -> 'EmbeddingVector':
        """Create normalized version of this embedding"""
        norm = self.get_norm()
        if norm == 0:
            raise ValueError("Cannot normalize zero vector")

        normalized_vector = [x / norm for x in self.vector]
        return EmbeddingVector(
            vector=normalized_vector,
            model_name=self.model_name,
            dimension=self.dimension,
            created_at=self.created_at,
            similarity_threshold=self.similarity_threshold
        )

    def calculate_similarity(self, other: 'EmbeddingVector') -> float:
        """Calculate cosine similarity with another embedding"""
        if self.dimension != other.dimension:
            raise ValueError("Cannot calculate similarity between vectors of different dimensions")

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(self.vector, other.vector))

        # Calculate norms
        norm_a = self.get_norm()
        norm_b = other.get_norm()

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Calculate cosine similarity
        return dot_product / (norm_a * norm_b)

    def is_similar_to(self, other: 'EmbeddingVector') -> bool:
        """Check if similar to another embedding based on threshold"""
        similarity = self.calculate_similarity(other)
        return similarity >= self.similarity_threshold


@dataclass(frozen=True)
class ConfidenceScore:
    """Confidence score for quality assessment"""
    value: float
    confidence: float = 1.0
    source: str = "automatic"
    assessed_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate confidence score invariants"""
        if self.value < 0 or self.value > 1:
            raise ValueError("Confidence score value must be between 0 and 1")

        if self.confidence < 0 or self.confidence > 1:
            raise ValueError("Confidence level must be between 0 and 1")

        if not self.source:
            raise ValueError("Confidence source cannot be empty")

    def is_high_quality(self) -> bool:
        """Check if score indicates high quality"""
        return self.value >= 0.8

    def is_medium_quality(self) -> bool:
        """Check if score indicates medium quality"""
        return 0.5 <= self.value < 0.8

    def is_low_quality(self) -> bool:
        """Check if score indicates low quality"""
        return self.value < 0.5

    def get_quality_level(self) -> str:
        """Get quality level as string"""
        if self.is_high_quality():
            return "high"
        elif self.is_medium_quality():
            return "medium"
        else:
            return "low"

    def combine_with(self, other: 'ConfidenceScore', weight: float = 0.5) -> 'ConfidenceScore':
        """Combine with another confidence score using weighted average"""
        combined_value = (self.value * weight) + (other.value * (1 - weight))
        combined_confidence = (self.confidence * weight) + (other.confidence * (1 - weight))

        return ConfidenceScore(
            value=combined_value,
            confidence=combined_confidence,
            source=f"combined_{self.source}_and_{other.source}"
        )


@dataclass(frozen=True)
class Metadata:
    """Document metadata container"""
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: str = "en"
    page_count: int = 0
    word_count: int = 0
    character_count: int = 0
    custom_fields: Dict[str, Union[str, int, float, bool, datetime]] = field(default_factory=dict)

    def get_summary(self) -> Dict[str, any]:
        """Get metadata summary"""
        return {
            "author": self.author,
            "title": self.title,
            "subject": self.subject,
            "keywords": self.keywords,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": self.modification_date.isoformat() if self.modification_date else None,
            "language": self.language,
            "page_count": self.page_count,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "custom_fields_count": len(self.custom_fields)
        }

    def has_keyword(self, keyword: str) -> bool:
        """Check if metadata contains keyword"""
        return keyword.lower() in [k.lower() for k in self.keywords]

    def add_custom_field(self, key: str, value: Union[str, int, float, bool, datetime]) -> 'Metadata':
        """Add custom field (returns new instance since frozen)"""
        new_custom = self.custom_fields.copy()
        new_custom[key] = value
        return Metadata(
            author=self.author,
            title=self.title,
            subject=self.subject,
            keywords=self.keywords.copy(),
            creation_date=self.creation_date,
            modification_date=self.modification_date,
            language=self.language,
            page_count=self.page_count,
            word_count=self.word_count,
            character_count=self.character_count,
            custom_fields=new_custom
        )


@dataclass(frozen=True)
class ProcessingStep:
    """Processing step information"""
    name: str
    status: str  # pending, running, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Union[str, int, float, bool]] = field(default_factory=dict)

    def start(self) -> 'ProcessingStep':
        """Mark step as started"""
        return ProcessingStep(
            name=self.name,
            status="running",
            started_at=datetime.utcnow(),
            metadata=self.metadata.copy()
        )

    def complete(self) -> 'ProcessingStep':
        """Mark step as completed"""
        now = datetime.utcnow()
        duration = None
        if self.started_at:
            duration = (now - self.started_at).total_seconds()

        return ProcessingStep(
            name=self.name,
            status="completed",
            started_at=self.started_at,
            completed_at=now,
            duration_seconds=duration,
            metadata=self.metadata.copy()
        )

    def fail(self, error_message: str) -> 'ProcessingStep':
        """Mark step as failed"""
        now = datetime.utcnow()
        duration = None
        if self.started_at:
            duration = (now - self.started_at).total_seconds()

        return ProcessingStep(
            name=self.name,
            status="failed",
            started_at=self.started_at,
            completed_at=now,
            duration_seconds=duration,
            error_message=error_message,
            metadata=self.metadata.copy()
        )

    def is_running(self) -> bool:
        """Check if step is currently running"""
        return self.status == "running"

    def is_completed(self) -> bool:
        """Check if step is completed"""
        return self.status == "completed"

    def has_failed(self) -> bool:
        """Check if step has failed"""
        return self.status == "failed"


# Utility functions for working with document value objects

def detect_content_type(file_name: str, file_bytes: Optional[bytes] = None) -> ContentType:
    """Detect content type from file name and optionally file content"""
    # Check file extension
    ext = file_name.lower().split('.')[-1] if '.' in file_name else ''

    extension_map = {
        'pdf': ContentType.PDF,
        'docx': ContentType.WORD,
        'doc': ContentType.WORD,
        'xlsx': ContentType.EXCEL,
        'xls': ContentType.EXCEL,
        'pptx': ContentType.POWERPOINT,
        'ppt': ContentType.POWERPOINT,
        'txt': ContentType.TEXT,
        'html': ContentType.HTML,
        'htm': ContentType.HTML,
        'jpg': ContentType.IMAGE_JPEG,
        'jpeg': ContentType.IMAGE_JPEG,
        'png': ContentType.IMAGE_PNG,
        'mp3': ContentType.AUDIO_MP3,
        'mp4': ContentType.VIDEO_MP4,
        'json': ContentType.JSON
    }

    return extension_map.get(ext, ContentType.TEXT)


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple frequency analysis"""
    # Simple keyword extraction - in practice, you'd use more sophisticated NLP
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Filter out common words
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'has', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}

    word_freq = {}
    for word in words:
        if word not in stop_words:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Sort by frequency and return top keywords
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:max_keywords]]


def create_text_chunks(content: str, chunk_size: int = 1000, overlap: int = 100) -> List[TextChunk]:
    """Split content into overlapping text chunks"""
    if not content:
        return []

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(content):
        end = start + chunk_size
        chunk_content = content[start:end]

        chunks.append(TextChunk(
            content=chunk_content,
            chunk_index=chunk_index,
            start_char=start,
            end_char=end,
            metadata={"length": len(chunk_content), "overlap": overlap if start > 0 else 0}
        ))

        chunk_index += 1
        start = end - overlap  # Overlap with previous chunk

        # Prevent infinite loop with small content
        if start >= len(content):
            break

    return chunks