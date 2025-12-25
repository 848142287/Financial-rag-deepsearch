"""
优化的搜索API端点
优先从MongoDB查询，然后回退到其他存储系统
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging

from app.services.optimized_retrieval_service import optimized_retrieval_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/optimized", tags=["优化搜索"])

# 请求模型
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    use_cache: bool = True

class SystemStatusRequest(BaseModel):
    details: bool = False

# 响应模型
class SearchResponse(BaseModel):
    query: str
    answer: str
    results: list
    summary: dict
    timestamp: str

class SystemStatusResponse(BaseModel):
    mongodb: dict
    neo4j: dict
    milvus: dict
    overall_status: str

@router.post("/search", response_model=SearchResponse)
async def optimized_search(request: SearchRequest):
    """
    优化搜索接口
    优先从MongoDB查询解析后的文档内容
    """
    try:
        logger.info(f"收到优化搜索请求: {request.query}")

        # 执行混合搜索
        result = optimized_retrieval_service.hybrid_search(
            query=request.query,
            limit=request.limit
        )

        logger.info(f"搜索完成，返回 {len(result['results'])} 个结果")
        return result

    except Exception as e:
        logger.error(f"优化搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(details: bool = False):
    """
    获取优化检索系统状态
    """
    try:
        status = optimized_retrieval_service.get_system_status()

        # 确定整体状态
        overall_status = "healthy"
        if not status["mongodb"]["connected"]:
            overall_status = "degraded"
        if not status["neo4j"]["connected"] and not status["milvus"]["connected"]:
            overall_status = "critical"

        status["overall_status"] = overall_status

        return status

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.post("/mongodb-only")
async def mongodb_only_search(request: SearchRequest):
    """
    仅从MongoDB搜索（用于测试和对比）
    """
    try:
        logger.info(f"收到MongoDB仅搜索请求: {request.query}")

        results = optimized_retrieval_service.search_mongodb_parsed_content(
            query=request.query,
            limit=request.limit
        )

        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "source": "mongodb_only"
        }

    except Exception as e:
        logger.error(f"MongoDB搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.post("/clear-cache")
async def clear_search_cache():
    """
    清除搜索缓存
    """
    try:
        if optimized_retrieval_service.mongo_client:
            cache_collection = optimized_retrieval_service.mongo_db['search_cache']
            result = cache_collection.delete_many({})
            return {
                "success": True,
                "cleared_count": result.deleted_count,
                "message": f"已清除 {result.deleted_count} 条缓存记录"
            }
        else:
            return {
                "success": False,
                "message": "MongoDB未连接，无法清除缓存"
            }

    except Exception as e:
        logger.error(f"清除缓存失败: {e}")
        raise HTTPException(status_code=500, detail=f"清除缓存失败: {str(e)}")

@router.get("/test-connections")
async def test_connections():
    """
    测试所有存储连接
    """
    try:
        status = optimized_retrieval_service.get_system_status()

        connection_tests = {
            "mongodb": {
                "connected": status["mongodb"]["connected"],
                "test_result": "✅ 连接正常" if status["mongodb"]["connected"] else "❌ 连接失败"
            },
            "neo4j": {
                "connected": status["neo4j"]["connected"],
                "test_result": "✅ 连接正常" if status["neo4j"]["connected"] else "❌ 连接失败"
            },
            "milvus": {
                "connected": status["milvus"]["connected"],
                "test_result": "✅ 连接正常" if status["milvus"]["connected"] else "❌ 连接失败"
            }
        }

        return {
            "connections": connection_tests,
            "timestamp": "2025-12-21T09:30:00Z"
        }

    except Exception as e:
        logger.error(f"测试连接失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试连接失败: {str(e)}")