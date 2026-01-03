"""
数据去重API端点
提供Milvus向量和Neo4j节点的去重功能
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from app.core.structured_logging import get_structured_logger

from app.services.vector_deduplicator import get_vector_deduplicator
from app.services.neo4j_deduplicator import get_neo4j_deduplicator

logger = get_structured_logger(__name__)
router = APIRouter(prefix="/api/v1/dedup", tags=["数据去重"])


# ============================================================================
# Milvus向量去重端点
# ============================================================================

class VectorDedupRequest(BaseModel):
    """向量去重请求"""
    limit: Optional[int] = Field(default=10000, description="处理的最大向量数量")
    dry_run: bool = Field(default=True, description="是否只分析不删除")


@router.post("/vectors/analyze")
async def analyze_duplicate_vectors(request: VectorDedupRequest):
    """
    分析Milvus中的重复向量
    """
    try:
        deduplicator = await get_vector_deduplicator()
        result = await deduplicator.find_duplicate_vectors(
            limit=request.limit,
            dry_run=request.dry_run
        )
        return {
            "success": True,
            "message": f"分析完成，发现 {result['duplicate_groups']} 组重复",
            "data": result
        }
    except Exception as e:
        logger.error(f"向量去重分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/vectors/delete")
async def delete_duplicate_vectors(request: VectorDedupRequest):
    """
    删除Milvus中的重复向量
    """
    try:
        if request.dry_run:
            raise HTTPException(
                status_code=400,
                detail="请先运行 analyze 并确认后再删除（设置 dry_run=False）"
            )

        deduplicator = await get_vector_deduplicator()
        result = await deduplicator.find_duplicate_vectors(
            limit=request.limit,
            dry_run=False
        )

        return {
            "success": True,
            "message": f"删除完成，删除了 {result['duplicates_deleted']} 个重复向量",
            "data": result
        }
    except Exception as e:
        logger.error(f"向量去重删除失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vectors/stats")
async def get_vector_stats():
    """获取Milvus集合统计信息"""
    try:
        deduplicator = await get_vector_deduplicator()
        stats = await deduplicator.get_collection_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取向量统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Neo4j去重端点
# ============================================================================

class Neo4jDedupRequest(BaseModel):
    """Neo4j去重请求"""
    entity_type: Optional[str] = Field(default=None, description="指定实体类型")
    limit: Optional[int] = Field(default=1000, description="最多检查的节点数")
    dry_run: bool = Field(default=True, description="是否只分析不执行")


@router.post("/neo4j/nodes/analyze")
async def analyze_duplicate_nodes(request: Neo4jDedupRequest):
    """
    分析Neo4j中的重复节点
    """
    try:
        deduplicator = await get_neo4j_deduplicator()
        result = await deduplicator.find_duplicate_nodes(
            entity_type=request.entity_type,
            limit=request.limit
        )
        return {
            "success": True,
            "message": f"分析完成，发现 {result['duplicate_groups']} 组重复节点",
            "data": result
        }
    except Exception as e:
        logger.error(f"Neo4j节点去重分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neo4j/nodes/merge")
async def merge_duplicate_nodes(request: Neo4jDedupRequest):
    """
    合并Neo4j中的重复节点
    """
    try:
        deduplicator = await get_neo4j_deduplicator()
        result = await deduplicator.merge_duplicate_nodes(dry_run=request.dry_run)
        return {
            "success": True,
            "message": f"合并完成，处理了 {result['merged_groups']} 组重复节点",
            "data": result
        }
    except Exception as e:
        logger.error(f"合并Neo4j节点失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neo4j/relationships/analyze")
async def analyze_duplicate_relationships():
    """分析Neo4j中的重复关系"""
    try:
        deduplicator = await get_neo4j_deduplicator()
        result = await deduplicator.find_duplicate_relationships()
        return {
            "success": True,
            "message": f"分析完成，发现 {result['duplicate_groups']} 组重复关系",
            "data": result
        }
    except Exception as e:
        logger.error(f"Neo4j关系去重分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neo4j/relationships/delete")
async def delete_duplicate_relationships(dry_run: bool = True):
    """删除Neo4j中的重复关系"""
    try:
        deduplicator = await get_neo4j_deduplicator()
        result = await deduplicator.delete_duplicate_relationships(dry_run=dry_run)
        return {
            "success": True,
            "message": f"删除完成，删除了 {result['relationships_deleted']} 条重复关系",
            "data": result
        }
    except Exception as e:
        logger.error(f"删除Neo4j关系失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/neo4j/setup-constraints")
async def setup_unique_constraints():
    """
    设置Neo4j唯一约束
    这是最有效的防止重复的方法
    """
    try:
        deduplicator = await get_neo4j_deduplicator()
        result = await deduplicator.setup_unique_constraints()
        return {
            "success": True,
            "message": f"设置完成：{len(result['constraints_created'])} 个约束，{len(result['indexes_created'])} 个索引",
            "data": result
        }
    except Exception as e:
        logger.error(f"设置唯一约束失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/neo4j/stats")
async def get_neo4j_stats():
    """获取Neo4j图统计信息"""
    try:
        deduplicator = await get_neo4j_deduplicator()
        stats = await deduplicator.get_graph_stats()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        logger.error(f"获取Neo4j统计失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# 综合去重端点
# ============================================================================

class FullDedupRequest(BaseModel):
    """完整去重请求"""
    vector_limit: Optional[int] = Field(default=10000)
    node_limit: Optional[int] = Field(default=1000)
    dry_run: bool = Field(default=True, description="是否只分析不执行")


@router.post("/full/analyze")
async def full_dedup_analysis(request: FullDedupRequest):
    """
    完整去重分析：向量和节点
    """
    try:
        results = {}

        # 1. 向量去重分析
        vector_dedup = await get_vector_deduplicator()
        results["vectors"] = await vector_dedup.find_duplicate_vectors(
            limit=request.vector_limit,
            dry_run=True
        )

        # 2. Neo4j节点去重分析
        neo4j_dedup = await get_neo4j_deduplicator()
        results["neo4j_nodes"] = await neo4j_dedup.find_duplicate_nodes(
            limit=request.node_limit
        )

        # 3. Neo4j关系去重分析
        results["neo4j_relationships"] = await neo4j_dedup.find_duplicate_relationships()

        summary = {
            "total_duplicate_vectors": results["vectors"]["duplicates_found"],
            "total_duplicate_nodes": results["neo4j_nodes"]["total_duplicates"],
            "total_duplicate_relationships": results["neo4j_relationships"]["total_duplicates"],
            "dry_run": request.dry_run
        }

        return {
            "success": True,
            "message": "完整去重分析完成",
            "summary": summary,
            "data": results
        }

    except Exception as e:
        logger.error(f"完整去重分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/full/execute")
async def full_dedup_execute(request: FullDedupRequest, background_tasks: BackgroundTasks):
    """
    执行完整去重（后台任务）
    """
    try:
        if request.dry_run:
            raise HTTPException(
                status_code=400,
                detail="请先运行分析并确认后再执行（设置 dry_run=False）"
            )

        # 添加后台任务
        background_tasks.add_task(execute_full_dedup, request)

        return {
            "success": True,
            "message": "去重任务已提交，正在后台执行"
        }

    except Exception as e:
        logger.error(f"提交去重任务失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def execute_full_dedup(request: FullDedupRequest):
    """后台执行完整去重"""
    try:
        logger.info("🚀 开始后台去重任务")

        # 1. 向量去重
        vector_dedup = await get_vector_deduplicator()
        vector_result = await vector_dedup.find_duplicate_vectors(
            limit=request.vector_limit,
            dry_run=False
        )
        logger.info(f"✅ 向量去重完成: {vector_result['duplicates_deleted']} 个")

        # 2. Neo4j节点去重
        neo4j_dedup = await get_neo4j_deduplicator()
        node_result = await neo4j_dedup.merge_duplicate_nodes(dry_run=False)
        logger.info(f"✅ 节点去重完成: {node_result['nodes_deleted']} 个")

        # 3. Neo4j关系去重
        rel_result = await neo4j_dedup.delete_duplicate_relationships(dry_run=False)
        logger.info(f"✅ 关系去重完成: {rel_result['relationships_deleted']} 条")

        logger.info("🎉 后台去重任务全部完成")

    except Exception as e:
        logger.error(f"❌ 后台去重任务失败: {e}")


# ============================================================================
# 快速去重端点（推荐使用）
# ============================================================================

@router.post("/quick-setup")
async def quick_setup_and_dedup():
    """
    快速设置和去重（推荐）
    1. 设置唯一约束
    2. 分析重复数据
    3. 返回去重建议
    """
    try:
        results = {}

        # 1. 设置约束
        neo4j_dedup = await get_neo4j_deduplicator()
        constraint_result = await neo4j_dedup.setup_unique_constraints()
        results["constraints"] = constraint_result

        # 2. 分析向量
        vector_dedup = await get_vector_deduplicator()
        vector_result = await vector_dedup.find_duplicate_vectors(limit=5000, dry_run=True)
        results["vectors_analysis"] = {
            "duplicate_groups": vector_result["duplicate_groups"],
            "duplicates_found": vector_result["duplicates_found"]
        }

        # 3. 分析节点
        node_result = await neo4j_dedup.find_duplicate_nodes(limit=500)
        results["nodes_analysis"] = {
            "duplicate_groups": node_result["duplicate_groups"],
            "total_duplicates": node_result["total_duplicates"]
        }

        # 4. 分析关系
        rel_result = await neo4j_dedup.find_duplicate_relationships()
        results["relationships_analysis"] = {
            "duplicate_groups": rel_result["duplicate_groups"],
            "total_duplicates": rel_result["total_duplicates"]
        }

        # 5. 统计
        stats = {
            "neo4j": await neo4j_dedup.get_graph_stats(),
            "milvus": await vector_dedup.get_collection_stats()
        }
        results["current_stats"] = stats

        # 6. 建议
        recommendations = []
        if vector_result["duplicates_found"] > 0:
            recommendations.append(f"发现 {vector_result['duplicates_found']} 个重复向量，建议清理")
        if node_result["total_duplicates"] > 0:
            recommendations.append(f"发现 {node_result['total_duplicates']} 个重复节点，建议合并")
        if rel_result["total_duplicates"] > 0:
            recommendations.append(f"发现 {rel_result['total_duplicates']} 条重复关系，建议删除")

        results["recommendations"] = recommendations

        return {
            "success": True,
            "message": "快速设置和分析完成",
            "data": results
        }

    except Exception as e:
        logger.error(f"快速设置失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
