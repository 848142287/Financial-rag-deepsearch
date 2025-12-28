"""
增强版搜索API端点 v2
提供结构化答案和归一化相似度分数
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import asyncio

from app.core.database import get_db
from app.services.smart_embedding_service import SmartEmbeddingService
from app.services.enhanced_answer_service import EnhancedAnswerService
from pymilvus import connections, Collection

logger = logging.getLogger(__name__)

router = APIRouter()


class EnhancedSearchRequest(BaseModel):
    query: str
    top_k: int = 10
    use_knowledge_graph: bool = True
    use_vector_search: bool = True
    enable_reranking: bool = True
    normalize_scores: bool = True
    user_id: Optional[str] = None


@router.post("/enhanced-search-v2")
async def enhanced_search_v2(
    request: EnhancedSearchRequest,
    db: Session = Depends(get_db)
):
    """
    增强版搜索接口 v2
    
    特点：
    1. 结构化答案（摘要 + 要点 + 详细说明）
    2. 归一化相似度分数（0-1范围）
    3. 百分比相似度展示
    4. 高质量LLM答案生成
    """
    try:
        logger.info(f"🚀 增强版搜索 v2: '{request.query}'")
        
        # 1. 向量检索
        embedding_service = SmartEmbeddingService()
        query_embedding = await embedding_service.encode_single(request.query)
        
        # 连接Milvus
        connections.connect(host='milvus', port='19530')
        collection = Collection("document_embeddings")
        collection.load()
        
        # 向量搜索（使用内积IP）
        search_params = {
            "metric_type": "IP",  # 内积
            "params": {"nprobe": 10}
        }
        
        results = collection.search(
            [query_embedding],
            "embedding",
            search_params,
            limit=request.top_k,
            output_fields=["content", "document_id", "chunk_id", "metadata"]
        )
        
        # 2. 处理搜索结果
        search_results = []
        for hit in results[0]:
            search_results.append({
                "id": str(hit.entity.get("document_id", hit.id)),
                "title": hit.entity.get("metadata", {}).get("title", "文档")[:100],
                "content": hit.entity.get("content", ""),
                "score": float(hit.distance),  # 内积分数
                "chunk_id": hit.entity.get("chunk_id", ""),
                "metadata": hit.entity.get("metadata", {})
            })
        
        # 3. 重排序（如果启用）
        if request.enable_reranking and len(search_results) > 1:
            logger.info("🔄 应用重排序...")
            content_list = [r["content"] for r in search_results]
            rerank_scores = await embedding_service.rerank(request.query, content_list)
            
            reranked_results = []
            for idx, (original_idx, score) in enumerate(rerank_scores):
                if original_idx < len(search_results):
                    result = search_results[original_idx]
                    result["score"] = score
                    reranked_results.append(result)
            
            search_results = reranked_results[:request.top_k]
        
        # 4. 计算分数统计（用于归一化）
        score_stats = {
            'min': min(r['score'] for r in search_results),
            'max': max(r['score'] for r in search_results),
            'avg': sum(r['score'] for r in search_results) / len(search_results)
        }
        
        # 5. 生成结构化答案
        logger.info("📝 生成结构化答案...")
        answer_service = EnhancedAnswerService()
        structured_answer = await answer_service.generate_structured_answer(
            query=request.query,
            search_results=search_results,
            normalize_scores=request.normalize_scores,
            score_stats=score_stats
        )
        
        # 6. 构建响应
        response = {
            "query": request.query,
            "answer": structured_answer,
            "retrieval_info": {
                "total_results": len(search_results),
                "vector_search_used": request.use_vector_search,
                "reranking_applied": request.enable_reranking,
                "score_normalization_enabled": request.normalize_scores,
                "score_statistics": {
                    "min": round(score_stats['min'], 2),
                    "max": round(score_stats['max'], 2),
                    "avg": round(score_stats['avg'], 2)
                }
            },
            "performance_metrics": {
                "embedding_dimension": len(query_embedding),
                "response_time_ms": 0  # 可以添加实际计时
            }
        }
        
        logger.info(f"✅ 搜索完成: {len(search_results)} 个结果, 置信度: {structured_answer.get('confidence', 0)}%")
        
        return response
    
    except Exception as e:
        logger.error(f"❌ 增强版搜索失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")


@router.get("/search-v2-status")
async def search_v2_status():
    """获取增强版搜索系统状态"""
    try:
        return {
            "status": "healthy",
            "version": "v2.0",
            "features": {
                "structured_answer": True,
                "normalized_similarity": True,
                "llm_answer_generation": True,
                "reranking": True
            },
            "message": "增强版搜索系统运行正常"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }
