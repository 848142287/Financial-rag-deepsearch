"""
简化的智能搜索API端点
基于文档元数据的搜索和问答功能
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from ...core.database import get_db
from ...services.intelligent_search import EnhancedIntelligentSearchService
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    search_type: str = "hybrid"  # semantic, keyword, hybrid

class SearchResponse(BaseModel):
    query: str
    search_type: str
    intent_analysis: Dict[str, Any]
    results_count: int
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]
    timestamp: str
    error: Optional[str] = None

class SuggestionsResponse(BaseModel):
    query: str
    suggestions: List[str]

@router.post("/intelligent-search", response_model=SearchResponse)
async def intelligent_search(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    智能搜索接口

    Args:
        request: 搜索请求
        db: 数据库会话

    Returns:
        搜索结果
    """
    try:
        search_service = EnhancedIntelligentSearchService(db)
        result = search_service.intelligent_search(
            query=request.query,
            limit=request.limit,
            search_type=request.search_type
        )

        return SearchResponse(**result)

    except Exception as e:
        logger.error(f"智能搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/intelligent-search", response_model=SearchResponse)
async def intelligent_search_get(
    query: str = Query(..., description="搜索查询"),
    limit: int = Query(10, ge=1, le=100, description="返回结果数量"),
    search_type: str = Query("hybrid", regex="^(semantic|keyword|hybrid)$", description="搜索类型"),
    db: Session = Depends(get_db)
):
    """
    智能搜索接口（GET方法）

    Args:
        query: 搜索查询
        limit: 返回结果数量限制
        search_type: 搜索类型 (semantic, keyword, hybrid)
        db: 数据库会话

    Returns:
        搜索结果
    """
    try:
        search_service = EnhancedIntelligentSearchService(db)
        result = search_service.intelligent_search(
            query=query,
            limit=limit,
            search_type=search_type
        )

        return SearchResponse(**result)

    except Exception as e:
        logger.error(f"智能搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/search-suggestions", response_model=SuggestionsResponse)
async def get_search_suggestions(
    query: str = Query(..., description="查询前缀"),
    limit: int = Query(10, ge=1, le=20, description="建议数量"),
    db: Session = Depends(get_db)
):
    """
    获取搜索建议

    Args:
        query: 查询前缀
        limit: 建议数量限制
        db: 数据库会话

    Returns:
        搜索建议列表
    """
    try:
        search_service = EnhancedIntelligentSearchService(db)
        suggestions = search_service.get_search_suggestions(query, limit)

        return SuggestionsResponse(
            query=query,
            suggestions=suggestions
        )

    except Exception as e:
        logger.error(f"获取搜索建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取建议失败: {str(e)}")

@router.get("/search-analytics")
async def get_search_analytics(
    db: Session = Depends(get_db)
):
    """
    获取搜索分析数据

    Args:
        db: 数据库会话

    Returns:
        搜索分析统计
    """
    try:
        from ...models.document import Document

        # 获取文档统计
        total_docs = db.query(Document).count()
        completed_docs = db.query(Document).filter(Document.status == 'completed').count()

        # 获取券商分布
        search_service = EnhancedIntelligentSearchService(db)
        all_docs = db.query(Document).all()

        broker_stats = {}
        concept_stats = {}
        indicator_stats = {}

        for doc in all_docs:
            full_text = f"{doc.title} {doc.filename} {getattr(doc, 'description', '')}"

            # 统计券商
            for broker in search_service.brokers:
                if broker in full_text:
                    broker_stats[broker] = broker_stats.get(broker, 0) + 1

            # 统计概念
            for concept, expansions in search_service.concept_expansions.items():
                if concept.lower() in full_text.lower() or any(exp.lower() in full_text.lower() for exp in expansions):
                    concept_stats[concept] = concept_stats.get(concept, 0) + 1

            # 统计技术指标
            for indicator in search_service.technical_indicators:
                if indicator.lower() in full_text.lower() or indicator in full_text:
                    indicator_stats[indicator] = indicator_stats.get(indicator, 0) + 1

        # 排序并限制数量
        broker_stats = dict(sorted(broker_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        concept_stats = dict(sorted(concept_stats.items(), key=lambda x: x[1], reverse=True)[:10])
        indicator_stats = dict(sorted(indicator_stats.items(), key=lambda x: x[1], reverse=True)[:10])

        return {
            "document_statistics": {
                "total_documents": total_docs,
                "completed_documents": completed_docs,
                "completion_rate": round(completed_docs / total_docs * 100, 2) if total_docs > 0 else 0
            },
            "content_distribution": {
                "brokers": broker_stats,
                "concepts": concept_stats,
                "indicators": indicator_stats
            },
            "search_capabilities": {
                "supported_search_types": ["semantic", "keyword", "hybrid"],
                "supported_entities": {
                    "brokers_count": len(search_service.brokers),
                    "concepts_count": len(search_service.concept_expansions),
                    "indicators_count": len(search_service.technical_indicators)
                }
            }
        }

    except Exception as e:
        logger.error(f"获取搜索分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取分析数据失败: {str(e)}")

@router.post("/question-answering")
async def question_answering(
    question: str = Query(..., description="问题"),
    context_limit: int = Query(5, ge=1, le=20, description="上下文文档数量"),
    db: Session = Depends(get_db)
):
    """
    基于文档的问答功能

    Args:
        question: 用户问题
        context_limit: 上下文文档数量限制
        db: 数据库会话

    Returns:
        问答结果
    """
    try:
        search_service = EnhancedIntelligentSearchService(db)

        # 搜索相关文档
        search_result = search_service.intelligent_search(
            query=question,
            limit=context_limit,
            search_type="semantic"
        )

        if search_result["results_count"] == 0:
            return {
                "question": question,
                "answer": "抱歉，我没有找到相关的文档来回答您的问题。",
                "confidence": 0.0,
                "sources": [],
                "related_topics": []
            }

        # 生成答案
        answer = _generate_answer(question, search_result)

        # 提取相关主题
        related_topics = list(set([
            topic for result in search_result["results"][:3]
            for topic in result["metadata"].get("concepts", [])
        ]))[:5]

        return {
            "question": question,
            "answer": answer,
            "confidence": min(0.9, 0.5 + search_result["results_count"] * 0.1),
            "sources": [
                {
                    "id": result["id"],
                    "title": result["title"],
                    "score": result["score"],
                    "preview": result["preview"]
                }
                for result in search_result["results"][:3]
            ],
            "related_topics": related_topics,
            "search_summary": search_result["summary"]
        }

    except Exception as e:
        logger.error(f"问答失败: {e}")
        raise HTTPException(status_code=500, detail=f"问答失败: {str(e)}")

def _generate_answer(question: str, search_result: Dict[str, Any]) -> str:
    """生成答案"""
    if not search_result["results"]:
        return "抱歉，我没有找到相关信息来回答这个问题。"

    # 分析问题类型
    question_lower = question.lower()
    answer_type = "general"

    if any(word in question_lower for word in ['如何', '怎么', '方法']):
        answer_type = "how_to"
    elif any(word in question_lower for word in ['什么', '定义', '解释']):
        answer_type = "what_is"
    elif any(word in question_lower for word in ['为什么', '原因']):
        answer_type = "why"
    elif any(word in question_lower for word in ['哪个', '哪家', '推荐']):
        answer_type = "which"

    # 提取关键信息
    results = search_result["results"][:3]
    entities = search_result["summary"]["found_entities"]

    # 构建答案
    if answer_type == "how_to":
        answer = f"关于如何{question.replace('如何', '').replace('怎么', '')}，根据相关研究报告：\n\n"
    elif answer_type == "what_is":
        answer = f"关于{question.replace('什么', '').replace('解释', '')}，根据相关资料：\n\n"
    elif answer_type == "why":
        answer = f"关于{question.replace('为什么', '').replace('原因', '')}的原因分析：\n\n"
    elif answer_type == "which":
        answer = f"针对{question.replace('哪个', '').replace('哪家', '').replace('推荐', '')}的选择建议：\n\n"
    else:
        answer = f"关于{question}的相关信息：\n\n"

    # 添加具体内容
    if entities["brokers"]:
        answer += f"相关券商研究：{', '.join(entities['brokers'][:3])}\n"

    if entities["concepts"]:
        answer += f"涉及概念：{', '.join(entities['concepts'][:3])}\n"

    if entities["indicators"]:
        answer += f"技术指标：{', '.join(entities['indicators'][:3])}\n"

    # 添加文档来源
    answer += f"\n参考了 {len(results)} 份相关研究报告，主要来源包括：\n"
    for i, result in enumerate(results, 1):
        answer += f"{i}. {result['title']}\n"

    answer += "\n请注意：以上信息基于公开研究报告，投资决策请结合自身情况并咨询专业建议。"

    return answer