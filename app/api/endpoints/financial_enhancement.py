"""
金融增强功能API端点
提供文档增强、智能切分、查询增强等金融专用功能
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.financial.document_enhancer import financial_document_enhancer
from app.services.financial.document_chunker import financial_document_chunker
from app.services.financial.query_enhancer import financial_query_enhancer

router = APIRouter(prefix="/financial-enhancement", tags=["Financial Enhancement"])


# 请求和响应模型
class DocumentEnhancementRequest(BaseModel):
    text: str = Field(..., description="要增强的文档文本", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")


class DocumentEnhancementResponse(BaseModel):
    success: bool
    enhanced_document: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DocumentChunkingRequest(BaseModel):
    text: str = Field(..., description="要切分的文档文本", min_length=1)
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据")
    chunk_size: Optional[int] = Field(500, description="目标片段大小")
    chunk_overlap: Optional[int] = Field(50, description="片段重叠大小")


class DocumentChunkingResponse(BaseModel):
    success: bool
    chunks: Optional[List[Dict[str, Any]]] = None
    total_chunks: Optional[int] = None
    error: Optional[str] = None


class QueryEnhancementRequest(BaseModel):
    query: str = Field(..., description="原始查询", min_length=1, max_length=500)


class QueryEnhancementResponse(BaseModel):
    success: bool
    original_query: str
    enhanced_query: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/enhance-document", response_model=DocumentEnhancementResponse)
async def enhance_document(request: DocumentEnhancementRequest, db: Session = Depends(get_db)):
    """
    增强金融文档

    - **text**: 文档文本
    - **metadata**: 可选的文档元数据
    """
    try:
        enhanced_doc = financial_document_enhancer.enhance_document(
            document=request.text,
            metadata=request.metadata
        )

        return DocumentEnhancementResponse(
            success=True,
            enhanced_document=enhanced_doc
        )

    except Exception as e:
        return DocumentEnhancementResponse(
            success=False,
            error=str(e)
        )


@router.post("/chunk-document", response_model=DocumentChunkingResponse)
async def chunk_document(request: DocumentChunkingRequest, db: Session = Depends(get_db)):
    """
    智能切分金融文档

    - **text**: 文档文本
    - **metadata**: 可选的文档元数据
    - **chunk_size**: 目标片段大小
    - **chunk_overlap**: 片段重叠大小
    """
    try:
        # 设置切分参数（如果提供）
        if hasattr(financial_document_chunker, 'default_chunk_size'):
            financial_document_chunker.default_chunk_size = request.chunk_size
        if hasattr(financial_document_chunker, 'chunk_overlap'):
            financial_document_chunker.chunk_overlap = request.chunk_overlap

        chunks = financial_document_chunker.chunk_document(
            text=request.text,
            metadata=request.metadata
        )

        # 转换为字典格式
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "section": chunk.section,
                "sub_section": chunk.sub_section,
                "content_type": chunk.content_type,
                "importance_score": chunk.importance_score,
                "word_count": chunk.word_count,
                "entities": chunk.entities,
                "financial_metrics": chunk.financial_metrics,
                "position": chunk.position
            }
            chunk_dicts.append(chunk_dict)

        return DocumentChunkingResponse(
            success=True,
            chunks=chunk_dicts,
            total_chunks=len(chunk_dicts)
        )

    except Exception as e:
        return DocumentChunkingResponse(
            success=False,
            error=str(e)
        )


@router.post("/enhance-query", response_model=QueryEnhancementResponse)
async def enhance_query(request: QueryEnhancementRequest, db: Session = Depends(get_db)):
    """
    增强金融查询

    - **query**: 原始查询文本
    """
    try:
        # 分析查询
        analysis = financial_query_enhancer.analyze_query(request.query)

        # 增强查询
        enhanced_query = financial_query_enhancer.enhance_query(request.query)

        # 转换分析结果为字典
        analysis_dict = {
            "original_query": analysis.original_query,
            "query_type": analysis.query_type,
            "extracted_entities": analysis.extracted_entities,
            "financial_metrics": analysis.financial_metrics,
            "time_info": analysis.time_info,
            "company_info": analysis.company_info,
            "intent": analysis.intent,
            "complexity": analysis.complexity
        }

        return QueryEnhancementResponse(
            success=True,
            original_query=request.query,
            enhanced_query=enhanced_query,
            analysis=analysis_dict
        )

    except Exception as e:
        return QueryEnhancementResponse(
            success=True,  # 即使增强失败，也返回原始查询
            original_query=request.query,
            error=str(e)
        )


@router.post("/process-financial-document")
async def process_financial_document(
    text: str = Field(..., description="金融文档文本"),
    metadata: Optional[Dict[str, Any]] = Field(None, description="文档元数据"),
    db: Session = Depends(get_db)
):
    """
    完整处理金融文档（增强+切分）

    - **text**: 文档文本
    - **metadata**: 可选的文档元数据
    """
    try:
        # 1. 文档增强
        enhanced_doc = financial_document_enhancer.enhance_document(
            document=text,
            metadata=metadata
        )

        # 2. 文档切分
        chunks = financial_document_chunker.chunk_document(
            text=enhanced_doc["normalized_text"],
            metadata=metadata
        )

        # 转换为响应格式
        chunk_dicts = []
        for chunk in chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "section": chunk.section,
                "sub_section": chunk.sub_section,
                "content_type": chunk.content_type,
                "importance_score": chunk.importance_score,
                "word_count": chunk.word_count,
                "entities": chunk.entities,
                "financial_metrics": chunk.financial_metrics
            }
            chunk_dicts.append(chunk_dict)

        return {
            "success": True,
            "enhanced_document": enhanced_doc,
            "chunks": chunk_dicts,
            "total_chunks": len(chunk_dicts),
            "processing_stats": {
                "entities_extracted": len(enhanced_doc.get("entities", [])),
                "financial_metrics_extracted": len(enhanced_doc.get("financial_metrics", [])),
                "sections_identified": len(set(chunk.section for chunk in chunks)),
                "total_words": sum(chunk.word_count for chunk in chunks)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "enhanced_document": None,
            "chunks": [],
            "total_chunks": 0,
            "processing_stats": {}
        }


@router.get("/financial-dictionaries")
async def get_financial_dictionaries():
    """
    获取金融词典信息
    """
    try:
        return {
            "success": True,
            "dictionaries": {
                "company_names": list(financial_document_enhancer.company_names),
                "financial_terms": list(financial_document_enhancer.financial_terms),
                "financial_metrics": list(financial_document_enhancer.financial_metrics),
                "standard_sections": financial_document_chunker.standard_sections,
                "query_types": list(financial_query_enhancer.query_classifiers.keys()),
                "financial_indicators": list(financial_query_enhancer.financial_indicators.keys())
            },
            "statistics": {
                "company_names_count": len(financial_document_enhancer.company_names),
                "financial_terms_count": len(financial_document_enhancer.financial_terms),
                "financial_metrics_count": len(financial_document_enhancer.financial_metrics)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "dictionaries": {},
            "statistics": {}
        }


@router.post("/analyze-financial-text")
async def analyze_financial_text(
    text: str = Field(..., description="要分析的金融文本"),
    db: Session = Depends(get_db)
):
    """
    分析金融文本的综合特征
    """
    try:
        # 文档增强
        enhanced_doc = financial_document_enhancer.enhance_document(text)

        # 查询分析（模拟）
        analysis = financial_query_enhancer.analyze_query(text)

        return {
            "success": True,
            "text_analysis": {
                "original_text": text,
                "cleaned_text": enhanced_doc.get("cleaned_text", ""),
                "normalized_text": enhanced_doc.get("normalized_text", ""),
                "document_features": enhanced_doc.get("document_features", {})
            },
            "entities": {
                "total_count": len(enhanced_doc.get("entities", [])),
                "companies": [e.text for e in enhanced_doc.get("entities", []) if e.type == "COMPANY"],
                "financial_terms": [e.text for e in enhanced_doc.get("entities", []) if e.type == "FINANCIAL_TERM"],
                "stock_codes": [e.text for e in enhanced_doc.get("entities", []) if e.type == "STOCK_CODE"],
                "metrics": [e.text for e in enhanced_doc.get("entities", []) if e.type == "METRIC"]
            },
            "financial_data": enhanced_doc.get("financial_metrics", []),
            "time_info": enhanced_doc.get("time_info", {}),
            "query_analysis": {
                "query_type": analysis.query_type,
                "intent": analysis.intent,
                "complexity": analysis.complexity,
                "extracted_entities": analysis.extracted_entities,
                "financial_metrics": analysis.financial_metrics
            }
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "text_analysis": {},
            "entities": {},
            "financial_data": [],
            "time_info": {},
            "query_analysis": {}
        }