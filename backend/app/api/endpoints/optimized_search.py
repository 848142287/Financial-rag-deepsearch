"""
优化的搜索API端点
使用Milvus向量搜索和Neo4j知识图谱搜索
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.structured_logging import get_structured_logger

from app.services.retrieval.unified_retrieval_service import get_unified_retrieval_service

# 获取统一检索服务实例
optimized_retrieval_service = None

async def get_retrieval_service():
    global optimized_retrieval_service
    if optimized_retrieval_service is None:
        optimized_retrieval_service = await get_unified_retrieval_service()
    return optimized_retrieval_service

logger = get_structured_logger(__name__)
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
    neo4j: dict
    milvus: dict
    overall_status: str

@router.post("/search", response_model=SearchResponse)
async def optimized_search(request: SearchRequest):
    """
    优化搜索接口
    结合Milvus向量搜索和Neo4j知识图谱搜索
    """
    try:
        logger.info(f"收到优化搜索请求: {request.query}")

        # 获取检索服务
        retrieval = await get_retrieval_service()

        # 执行混合搜索
        result = await retrieval.search(
            query=request.query,
            top_k=request.limit,
            retrieval_mode="hybrid"
        )

        logger.info(f"搜索完成，返回 {len(result.get('results', []))} 个结果")

        # 转换为响应格式
        return {
            "query": request.query,
            "answer": result.get("answer", ""),
            "results": result.get("results", []),
            "summary": {
                "total_found": result.get("total_found", 0),
                "milvus_count": len(result.get("milvus_results", [])),
                "neo4j_count": len(result.get("neo4j_results", []))
            },
            "timestamp": result.get("timestamp", "")
        }

    except Exception as e:
        logger.error(f"优化搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status(details: bool = False):
    """
    获取优化检索系统状态
    """
    try:
        retrieval = await get_retrieval_service()
        health = await retrieval.health_check()

        # 转换为响应格式
        return {
            "neo4j": {
                "connected": health.get("neo4j_connected", False),
                "status": "healthy" if health.get("neo4j_connected") else "disconnected"
            },
            "milvus": {
                "connected": health.get("milvus_connected", False),
                "status": "healthy" if health.get("milvus_connected") else "disconnected"
            },
            "overall_status": health.get("status", "unknown")
        }

    except Exception as e:
        logger.error(f"获取系统状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")

@router.get("/test-connections")
async def test_connections():
    """
    测试所有存储连接
    """
    try:
        retrieval = await get_retrieval_service()
        health = await retrieval.health_check()

        connection_tests = {
            "neo4j": {
                "connected": health.get("neo4j_connected", False),
                "test_result": "✅ 连接正常" if health.get("neo4j_connected") else "❌ 连接失败"
            },
            "milvus": {
                "connected": health.get("milvus_connected", False),
                "test_result": "✅ 连接正常" if health.get("milvus_connected") else "❌ 连接失败"
            }
        }

        return {
            "connections": connection_tests,
            "timestamp": "2025-12-30T00:00:00Z"
        }

    except Exception as e:
        logger.error(f"测试连接失败: {e}")
        raise HTTPException(status_code=500, detail=f"测试连接失败: {str(e)}")
