"""
GLM-4.7 RAG问答API端点
提供基于GLM-4.7的文档问答接口
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from typing import Optional, List

from app.core.database import get_db
from app.core.structured_logging import get_structured_logger
from app.services.rag_glm_service import rag_glm_service

logger = get_structured_logger(__name__)

router = APIRouter(tags=["RAG问答 (GLM-4.7)"])


class QuestionRequest(BaseModel):
    """问题请求"""
    question: str = Field(..., description="用户问题", min_length=1, max_length=500)
    top_k: int = Field(10, description="检索的文档块数量", ge=1, le=50)
    min_confidence: float = Field(0.7, description="最小置信度阈值", ge=0.0, le=1.0)
    enable_rerank: bool = Field(True, description="是否启用重排序")


class SourceInfo(BaseModel):
    """来源信息"""
    chunk_id: str
    document_id: int
    document_title: str
    document_filename: str
    content_preview: str
    confidence: float
    metadata: dict


class QuestionResponse(BaseModel):
    """问题回答响应"""
    success: bool
    question: str
    answer: str
    sources: List[SourceInfo]
    trust_explanation: dict
    retrieval_path: dict
    processing_time: float
    model: str
    timestamp: str
    error: Optional[str] = None


@router.post("/ask-glm", response_model=QuestionResponse)
async def ask_question_with_glm(
    request: QuestionRequest,
    db: Session = Depends(get_db)
):
    """
    使用GLM-4.7模型进行RAG问答

    功能：
    1. 向量检索相关文档片段
    2. 可选的结果重排序
    3. 构建RAG提示词
    4. 调用GLM-4.7生成答案
    5. 返回答案、来源、信任度等完整信息

    参数：
    - question: 用户问题
    - top_k: 检索文档块数量（默认10）
    - min_confidence: 最小置信度（默认0.7）
    - enable_rerank: 是否重排序（默认True）

    返回：
    - success: 是否成功
    - answer: AI生成的答案
    - sources: 来源文档列表
    - trust_explanation: 信任度说明
    - retrieval_path: 检索路径信息
    - processing_time: 处理时间（秒）
    """
    try:
        logger.info(f"收到问题: {request.question}")

        # 调用RAG服务
        result = await rag_glm_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            min_confidence=request.min_confidence,
            enable_rerank=request.enable_rerank,
            db=db
        )

        # 记录结果
        if result['success']:
            logger.info(
                f"✅ 答案生成成功: 答案长度={len(result['answer'])}, "
                f"来源数={len(result['sources'])}, "
                f"处理时间={result['processing_time']:.2f}秒"
            )
        else:
            logger.warning(f"❌ 答案生成失败: {result.get('error', '未知错误')}")

        return QuestionResponse(**result)

    except Exception as e:
        logger.error(f"API调用失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"处理问题时发生错误: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "service": "RAG问答服务 (GLM-4.7)",
        "status": "running",
        "model": "glm-4.7"
    }
