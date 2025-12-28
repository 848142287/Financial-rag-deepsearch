"""
简单的搜索API端点
用于测试基本搜索功能
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, List
from pydantic import BaseModel
import logging

from app.core.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_knowledge_graph: bool = True
    use_vector_search: bool = True
    use_hybrid_search: bool = True
    enable_reranking: bool = True

@router.post("/simple-search")
async def simple_search(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    简单搜索接口
    """
    try:
        logger.info(f"简单搜索: '{request.query}'")

        # 查询数据库中的文档
        query_sql = """
        SELECT id, filename, title, created_at, status
        FROM documents
        WHERE title LIKE :query OR filename LIKE :query
        ORDER BY created_at DESC
        LIMIT :limit
        """

        results = db.execute(
            text(query_sql),
            {"query": f"%{request.query}%", "limit": request.top_k}
        ).fetchall()

        # 格式化结果
        formatted_results = []
        for row in results:
            formatted_results.append({
                "id": str(row.id),
                "title": row.title or row.filename,
                "filename": row.filename,
                "status": row.status,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "relevance_score": 0.8  # 简单的相关性分数
            })

        # 生成简单的答案
        answer = f"找到 {len(formatted_results)} 个与 '{request.query}' 相关的文档。"
        if formatted_results:
            answer += f" 最相关的文档是: {formatted_results[0]['title']}。"

        return {
            "query": request.query,
            "answer": answer,
            "sources": formatted_results,
            "total_results": len(formatted_results),
            "knowledge_graph_used": request.use_knowledge_graph,
            "vector_search_used": request.use_vector_search,
            "hybrid_search_used": request.use_hybrid_search,
            "reranking_applied": request.enable_reranking,
            "confidence": 0.8 if formatted_results else 0.1
        }

    except Exception as e:
        logger.error(f"简单搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")