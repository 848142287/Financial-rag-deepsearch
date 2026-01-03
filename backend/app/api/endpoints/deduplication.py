"""
文档去重检查API端点
"""

from app.core.structured_logging import get_structured_logger
from typing import Dict, Any
from fastapi import APIRouter, HTTPException

from app.schemas.deduplication import (
    DocumentDuplicateCheck, DocumentDuplicateResponse,
    UploadDeduplicationDecision, DocumentUploadRequest
)
from app.services.document_deduplication import document_deduplication_service

logger = get_structured_logger(__name__)
router = APIRouter()

@router.post("/check-duplicate")
async def check_document_duplicate(
    request: DocumentDuplicateCheck
) -> DocumentDuplicateResponse:
    """
    检查文档重复

    检查文档是否已经存在于系统中
    """
    try:
        logger.info(f"检查文档重复: file_hash={request.file_hash[:8]}")

        result = await document_deduplication_service.get_duplicate_summary(
            request.file_hash, request.content_hash
        )

        return DocumentDuplicateResponse(
            is_duplicate=result['is_duplicate'],
            similarity_score=result['similarity_score'],
            duplicate_sources=result['duplicate_sources'],
            existing_document_id=result['existing_document_id'],
            existing_document_info=result['existing_document_info'] if 'existing_document_info' in result else None,
            recommendations=result['recommendations']
        )

    except Exception as e:
        logger.error(f"检查文档重复失败: {e}")
        raise HTTPException(status_code=500, detail=f"检查文档重复失败: {str(e)}")

@router.post("/deduplication-decision")
async def make_deduplication_decision(
    result: DocumentDuplicateResponse,
    upload_request: DocumentUploadRequest
) -> UploadDeduplicationDecision:
    """
    根据重复检查结果做出上传决策
    """
    try:
        decision = UploadDeduplicationDecision()

        if not result.is_duplicate:
            decision.should_block = False
            decision.message = "文档不重复，可以正常上传"
            decision.severity = "info"
            return decision

        # 根据相似度决定处理方式
        if result.similarity_score >= 0.99:
            decision.should_block = True
            decision.message = f"文档已存在（相似度: {result.similarity_score:.1%}），请勿重复上传。"
            decision.severity = "error"
            decision.allow_override = False
            if result.existing_document_id:
                decision.existing_document_id = result.existing_document_id
        elif result.similarity_score >= 0.95:
            decision.should_block = True
            decision.message = f"文档高度相似（相似度: {result.similarity_score:.1%}），建议查看现有文档后再决定。"
            decision.severity = "warning"
            decision.allow_override = True
            if result.existing_document_id:
                decision.existing_document_id = result.existing_document_id
        elif result.similarity_score >= 0.85:
            decision.should_block = False
            decision.message = f"文档较为相似（相似度: {result.similarity_score:.1%}），可以正常上传但请注意。"
            decision.severity = "info"
            decision.allow_override = True
        else:
            decision.should_block = False
            decision.message = f"文档相似度较低（相似度: {result.similarity_score:.1%}），可以正常上传。"
            decision.severity = "info"
            decision.allow_override = True

        return decision

    except Exception as e:
        logger.error(f"做出重复决策失败: {e}")
        raise HTTPException(status_code=500, detail=f"做出重复决策失败: {str(e)}")

@router.get("/duplicate-summary/{file_hash}/{content_hash}")
async def get_duplicate_summary(
    file_hash: str,
    content_hash: str
) -> Dict[str, Any]:
    """
    获取重复检查摘要
    """
    try:
        result = await document_deduplication_service.get_duplicate_summary(file_hash, content_hash)
        return result
    except Exception as e:
        logger.error(f"获取重复摘要失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取重复摘要失败: {str(e)}")