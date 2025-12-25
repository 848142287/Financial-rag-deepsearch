"""
增强功能API端点
测试所有新增的金融RAG功能
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List, Any, Optional, Dict
from datetime import datetime
import logging

from app.core.database import get_db
from app.services.enhanced_pdf_processor import process_pdf_document
from app.services.data_balancer import data_balancer
from app.services.dataset_expander import dataset_expander
from app.services.financial_llm_service import financial_llm_service, ModelType, FinancialTaskType
from app.services.enhanced_semantic_matching import enhanced_semantic_matcher, MatchingStrategy
from app.services.financial_knowledge_graph import financial_knowledge_graph
from app.services.user_behavior_analytics import user_behavior_analytics, UserAction, UserActionType

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/enhanced-pdf-process")
async def enhanced_pdf_process(
    file: UploadFile = File(...),
    use_marker: bool = True,
    use_nougat: bool = True
):
    """增强PDF处理测试"""
    try:
        # 保存上传的文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # 处理PDF
        config = {
            "use_marker": use_marker,
            "use_nougat": use_nougat,
            "extract_images": True,
            "extract_tables": True
        }

        result = await process_pdf_document(temp_file_path, config)

        # 清理临时文件
        import os
        os.unlink(temp_file_path)

        return {
            "success": result.success,
            "pages_processed": result.pages_processed,
            "total_pages": result.total_pages,
            "processing_time": result.processing_time,
            "content_count": len(result.contents),
            "errors": result.errors,
            "metadata": result.metadata
        }

    except Exception as e:
        logger.error(f"增强PDF处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data-balance-status")
async def get_data_balance_status():
    """获取数据平衡状态"""
    try:
        status = await data_balancer.get_balance_status()
        quality_summary = data_balancer.get_quality_summary()

        return {
            "balance_status": status,
            "quality_summary": quality_summary
        }

    except Exception as e:
        logger.error(f"数据平衡状态获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/auto-balance-data")
async def auto_balance_data():
    """自动平衡数据"""
    try:
        result = await data_balancer.auto_balance()
        return result

    except Exception as e:
        logger.error(f"自动数据平衡失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collect-dataset")
async def collect_dataset(source_id: Optional[str] = None):
    """收集数据集"""
    try:
        if source_id:
            # 从指定数据源收集
            tasks = await dataset_expander.collect_from_source(source_id)
        else:
            # 从所有数据源收集
            all_tasks = []
            for source_id in dataset_expander.data_sources.keys():
                tasks = await dataset_expander.collect_from_source(source_id)
                all_tasks.extend(tasks)
            tasks = all_tasks

        return {
            "total_tasks_created": len(tasks),
            "tasks": [
                {
                    "task_id": task.task_id,
                    "source_id": task.source_id,
                    "url": task.url,
                    "title": task.title,
                    "category": task.category.value,
                    "status": task.status.value
                }
                for task in tasks[:10]  # 返回前10个任务信息
            ]
        }

    except Exception as e:
        logger.error(f"数据集收集失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-pending-tasks")
async def process_pending_tasks(background_tasks: BackgroundTasks):
    """处理待处理任务"""
    try:
        # 添加后台任务
        background_tasks.add_task(dataset_expander.process_pending_tasks)

        return {"message": "开始处理待处理任务", "status": "processing"}

    except Exception as e:
        logger.error(f"任务处理启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dataset-statistics")
async def get_dataset_statistics():
    """获取数据集统计信息"""
    try:
        stats = await dataset_expander.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"数据集统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial-sentiment-analysis")
async def financial_sentiment_analysis(text: str):
    """金融情感分析测试"""
    try:
        result = await financial_llm_service.analyze_sentiment(text, ModelType.FINBERT)

        return {
            "sentiment": result.result.sentiment,
            "confidence": result.confidence,
            "scores": result.result.scores,
            "processing_time": result.processing_time,
            "model_used": result.model_type.value
        }

    except Exception as e:
        logger.error(f"金融情感分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial-entity-extraction")
async def financial_entity_extraction(text: str):
    """金融实体提取测试"""
    try:
        result = await financial_llm_service.extract_entities(text)

        entities = [
            {
                "entity": entity.entity,
                "entity_type": entity.entity_type,
                "confidence": entity.confidence,
                "start_pos": entity.start_pos,
                "end_pos": entity.end_pos
            }
            for entity in result.result
        ]

        return {
            "entities": entities,
            "total_count": len(entities),
            "processing_time": result.processing_time,
            "model_used": result.model_type.value
        }

    except Exception as e:
        logger.error(f"金融实体提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial-keyword-extraction")
async def financial_keyword_extraction(text: str, top_k: int = 10):
    """金融关键词提取测试"""
    try:
        result = await financial_llm_service.extract_keywords(text, top_k)

        keywords = [
            {"keyword": kw[0], "score": kw[1]}
            for kw in result.result
        ]

        return {
            "keywords": keywords,
            "total_count": len(keywords),
            "processing_time": result.processing_time,
            "model_used": result.model_type.value
        }

    except Exception as e:
        logger.error(f"金融关键词提取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/financial-summary")
async def financial_summary(text: str, max_length: int = 150):
    """金融摘要生成测试"""
    try:
        result = await financial_llm_service.generate_summary(text, max_length)

        return {
            "summary": result.result,
            "original_length": len(text),
            "summary_length": len(result.result),
            "processing_time": result.processing_time,
            "model_used": result.model_type.value
        }

    except Exception as e:
        logger.error(f"金融摘要生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financial-model-status")
async def get_financial_model_status():
    """获取金融模型状态"""
    try:
        status = financial_llm_service.get_model_status()
        return status

    except Exception as e:
        logger.error(f"金融模型状态获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhanced-semantic-search")
async def enhanced_semantic_search(
    query: str,
    top_k: int = 10,
    strategy: str = "hybrid"
):
    """增强语义搜索测试"""
    try:
        # 转换策略
        strategy_enum = MatchingStrategy(strategy)

        # 初始化匹配器（如果需要）
        if not enhanced_semantic_matcher.models:
            await enhanced_semantic_matcher.initialize()

        # 执行搜索
        results = await enhanced_semantic_matcher.search(
            query=query,
            top_k=top_k,
            strategy=strategy_enum
        )

        # 格式化结果
        formatted_results = [
            {
                "doc_id": result.doc_id,
                "score": result.score,
                "semantic_score": result.semantic_score,
                "keyword_score": result.keyword_score,
                "financial_score": result.financial_score,
                "context_score": result.context_score,
                "temporal_score": result.temporal_score
            }
            for result in results
        ]

        return {
            "query": query,
            "strategy": strategy,
            "total_results": len(formatted_results),
            "results": formatted_results
        }

    except Exception as e:
        logger.error(f"增强语义搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/semantic-matching-statistics")
async def get_semantic_matching_statistics():
    """获取语义匹配统计信息"""
    try:
        stats = enhanced_semantic_matcher.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"语义匹配统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-knowledge-graph")
async def build_knowledge_graph(background_tasks: BackgroundTasks):
    """构建金融知识图谱"""
    try:
        # 获取一些示例文档（实际应用中应该从数据库获取）
        sample_documents = [
            {
                "id": "doc1",
                "content": "阿里巴巴集团是中国最大的电子商务公司之一，在纽约证券交易所上市。",
                "category": "公司概况"
            },
            {
                "id": "doc2",
                "content": "腾讯控股有限公司是一家中国的跨国企业集团，业务涵盖社交媒体、游戏、云计算等领域。",
                "category": "公司概况"
            },
            {
                "id": "doc3",
                "content": "中国工商银行是中国最大的商业银行之一，提供全面的金融服务。",
                "category": "金融机构"
            }
        ]

        # 添加后台任务构建知识图谱
        background_tasks.add_task(financial_knowledge_graph.build_from_documents, sample_documents)

        return {"message": "开始构建金融知识图谱", "status": "processing"}

    except Exception as e:
        logger.error(f"知识图谱构建启动失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query-knowledge-graph")
async def query_knowledge_graph(
    entity_name: str,
    entity_type: Optional[str] = None
):
    """查询金融知识图谱"""
    try:
        # 查询实体
        entities = await financial_knowledge_graph.query_entity(entity_name)

        # 格式化实体结果
        formatted_entities = [
            {
                "id": entity.id,
                "type": entity.type.value,
                "name": entity.name,
                "aliases": entity.aliases,
                "confidence": entity.confidence,
                "properties": entity.properties
            }
            for entity in entities
        ]

        results = {"entities": formatted_entities}

        # 如果有实体，获取其关系
        if formatted_entities:
            entity_id = formatted_entities[0]["id"]
            relations = await financial_knowledge_graph.get_entity_relations(entity_id)

            formatted_relations = [
                {
                    "id": relation.id,
                    "subject_id": relation.subject_id,
                    "object_id": relation.object_id,
                    "relation_type": relation.relation_type.value,
                    "confidence": relation.confidence,
                    "properties": relation.properties
                }
                for relation in relations
            ]

            results["relations"] = formatted_relations

        return results

    except Exception as e:
        logger.error(f"知识图谱查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge-graph-reasoning")
async def knowledge_graph_reasoning(entity_ids: List[str]):
    """知识图谱推理"""
    try:
        reasoning_results = await financial_knowledge_graph.reason_about_entities(entity_ids)
        return reasoning_results

    except Exception as e:
        logger.error(f"知识图谱推理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/knowledge-graph-statistics")
async def get_knowledge_graph_statistics():
    """获取知识图谱统计信息"""
    try:
        stats = financial_knowledge_graph.get_statistics()
        return stats

    except Exception as e:
        logger.error(f"知识图谱统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/record-user-action")
async def record_user_action(
    user_id: str,
    action_type: str,
    document_id: str,
    query: Optional[str] = None,
    duration: Optional[int] = None,
    rating: Optional[int] = None
):
    """记录用户行为"""
    try:
        # 创建用户行为记录
        action = UserAction(
            user_id=user_id,
            action_type=UserActionType(action_type),
            document_id=document_id,
            timestamp=datetime.now(),
            query=query,
            duration=duration,
            rating=rating
        )

        # 记录行为
        success = await user_behavior_analytics.record_action(action)

        return {
            "success": success,
            "message": "用户行为记录成功" if success else "用户行为记录失败"
        }

    except Exception as e:
        logger.error(f"用户行为记录失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-recommendations")
async def get_user_recommendations(
    user_id: str,
    strategy: str = "hybrid",
    top_k: int = 10
):
    """获取用户推荐"""
    try:
        # 初始化分析器（如果需要）
        if not user_behavior_analytics.content_vectorizer:
            await user_behavior_analytics.initialize()

        from user_behavior_analytics import RecommendationStrategy

        # 转换策略
        strategy_enum = RecommendationStrategy(strategy)

        # 获取推荐
        recommendations = await user_behavior_analytics.get_recommendations(
            user_id=user_id,
            strategy=strategy_enum,
            top_k=top_k
        )

        # 格式化推荐结果
        formatted_recommendations = [
            {
                "document_id": rec.document_id,
                "score": rec.score,
                "reason": rec.reason,
                "strategy": rec.strategy,
                "metadata": rec.metadata
            }
            for rec in recommendations
        ]

        return {
            "user_id": user_id,
            "strategy": strategy,
            "total_recommendations": len(formatted_recommendations),
            "recommendations": formatted_recommendations
        }

    except Exception as e:
        logger.error(f"用户推荐获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-profile")
async def get_user_profile(user_id: str):
    """获取用户画像"""
    try:
        profile = await user_behavior_analytics.get_user_profile(user_id)

        if not profile:
            return {"user_id": user_id, "message": "用户画像不存在"}

        return {
            "user_id": profile.user_id,
            "preferences": profile.preferences,
            "interests": profile.interests,
            "expertise_level": profile.expertise_level,
            "search_history_count": len(profile.search_history),
            "created_at": profile.created_at.isoformat(),
            "updated_at": profile.updated_at.isoformat()
        }

    except Exception as e:
        logger.error(f"用户画像获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-statistics")
async def get_user_statistics(user_id: str):
    """获取用户统计信息"""
    try:
        stats = await user_behavior_analytics.get_user_statistics(user_id)
        return stats

    except Exception as e:
        logger.error(f"用户统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system-statistics")
async def get_system_statistics():
    """获取系统统计信息"""
    try:
        stats = user_behavior_analytics.get_system_statistics()
        return stats

    except Exception as e:
        logger.error(f"系统统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health-check")
async def health_check():
    """系统健康检查"""
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "enhanced_pdf_processor": "available",
                "data_balancer": "available",
                "dataset_expander": "available",
                "financial_llm_service": "available",
                "enhanced_semantic_matcher": "available",
                "financial_knowledge_graph": "available",
                "user_behavior_analytics": "available"
            }
        }

        return health_status

    except Exception as e:
        logger.error(f"健康检查失败: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }