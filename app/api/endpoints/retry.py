"""
Retry API endpoints
Manage document upload and processing retries
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from pydantic import BaseModel

from ...core.database import get_db
from ...models.document import Document, DocumentStatus
from ...services.retry_service import RetryService, get_failed_documents
from ...core.auth import get_current_user

router = APIRouter(prefix="/retry", tags=["retry"])


class RetryRequest(BaseModel):
    document_id: str


class RetryResponse(BaseModel):
    success: bool
    document_id: str
    failure_type: str
    new_status: str
    error: str = None


class RetryStatistics(BaseModel):
    uploading: int
    uploaded: int
    processing: int
    parsing: int
    parsed: int
    embedding: int
    embedded: int
    knowledge_graph_processing: int
    knowledge_graph_processed: int
    completed: int
    duplicate: int
    upload_failed: int
    parsing_failed: int
    embedding_failed: int
    processing_failed: int
    knowledge_graph_failed: int
    permanently_failed: int
    retried_documents: int
    avg_retry_count: float
    eligible_for_retry: int


class FailedDocument(BaseModel):
    id: int
    title: str
    filename: str
    status: DocumentStatus
    error_message: str
    retry_count: int
    next_retry_at: str
    created_at: str


@router.post("/document", response_model=RetryResponse)
async def retry_document(
    request: RetryRequest,
    db: Session = Depends(get_db),
    current_user: Depends = Depends(get_current_user)
):
    """
    Manually retry a specific failed document
    """
    retry_service = RetryService()
    result = await retry_service.manual_retry_document(db, request.document_id)

    if not result["success"] and "not found" in result.get("error", ""):
        raise HTTPException(status_code=404, detail=result["error"])

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return RetryResponse(**result)


@router.post("/process-all", response_model=Dict[str, Any])
async def process_all_retries(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Trigger background processing of all eligible retries
    """
    retry_service = RetryService()

    # Add background task
    background_tasks.add_task(
        retry_service.check_and_retry_failed_documents,
        db
    )

    # Get immediate statistics
    eligible_count = len(retry_service._get_retry_eligible_documents(db))

    return {
        "message": "Retry processing started",
        "eligible_documents": eligible_count,
        "status": "processing"
    }


@router.get("/statistics", response_model=RetryStatistics)
async def get_retry_statistics(
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive retry statistics
    """
    retry_service = RetryService()
    stats = retry_service.get_retry_statistics(db)

    return RetryStatistics(**stats)


@router.get("/failed-documents", response_model=List[FailedDocument])
async def get_failed_documents_list(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Get list of failed documents with retry information
    """
    failed_docs = get_failed_documents(db, limit)

    result = []
    for doc in failed_docs:
        result.append(FailedDocument(
            id=doc.id,
            title=doc.title,
            filename=doc.filename,
            status=doc.status,
            error_message=doc.error_message or "No error message",
            retry_count=doc.retry_count or 0,
            next_retry_at=doc.next_retry_at.isoformat() if doc.next_retry_at else None,
            created_at=doc.created_at.isoformat()
        ))

    return result


@router.post("/reset-retries/{document_id}")
async def reset_document_retries(
    document_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Reset retry count for a specific document
    """
    doc = db.query(Document).filter(Document.id == document_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.retry_count = 0
    doc.next_retry_at = None
    db.commit()

    return {
        "message": "Retry count reset successfully",
        "document_id": document_id
    }


@router.delete("/permanent/{document_id}")
async def mark_permanently_failed(
    document_id: int,
    db: Session = Depends(get_db),
    current_user = Depends(get_current_user)
):
    """
    Mark a document as permanently failed (no more retries)
    """
    doc = db.query(Document).filter(Document.id == document_id).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    doc.status = DocumentStatus.PERMANENTLY_FAILED
    db.commit()

    return {
        "message": "Document marked as permanently failed",
        "document_id": document_id
    }