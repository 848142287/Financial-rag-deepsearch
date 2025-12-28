"""
智能搜索API端点 - 使用修复后的嵌入服务
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import asyncio

from app.core.database import get_db
from app.services.smart_embedding_service import SmartEmbeddingService
from app.services.llm_service import LLMService
from pymilvus import connections, Collection

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    use_knowledge_graph: bool = True
    use_vector_search: bool = True
    enable_reranking: bool = True
    user_id: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    title: str
    content: str
    score: float
    metadata: Dict[str, Any]

@router.post("/intelligent-search")
async def intelligent_search(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    智能搜索接口 - 使用向量搜索
    """
    try:
        logger.info(f"智能搜索: '{request.query}'")

        # 初始化嵌入服务
        embedding_service = SmartEmbeddingService()

        # 生成查询向量
        query_embedding = await embedding_service.encode_single(request.query)
        logger.info(f"生成查询向量成功，维度: {len(query_embedding)}")

        # 连接Milvus
        connections.connect(host='milvus', port='19530')
        collection = Collection("document_embeddings")

        # 向量搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": 64}
        }

        results = collection.search(
            [query_embedding],
            "embedding",
            search_params,
            limit=request.top_k,
            output_fields=["content", "document_id", "metadata"]
        )

        # 处理搜索结果
        search_results = []
        for hit in results[0]:
            search_results.append(SearchResult(
                id=str(hit.entity.get("document_id", "")),
                title=hit.entity.get("content", "")[:100],
                content=hit.entity.get("content", ""),
                score=float(hit.distance),
                metadata=hit.entity.get("metadata", {})
            ))

        # 如果启用重排序
        if request.enable_reranking and len(search_results) > 1:
            logger.info("应用重排序...")
            content_list = [result.content for result in search_results]
            rerank_scores = await embedding_service.rerank(request.query, content_list)

            # 重新排序结果
            reranked_results = []
            for idx, (original_idx, score) in enumerate(rerank_scores):
                if original_idx < len(search_results):
                    result = search_results[original_idx]
                    result.score = score
                    reranked_results.append(result)

            search_results = reranked_results[:request.top_k]

        # 生成答案（使用LLM）
        answer = await generate_answer_with_llm(request.query, search_results)

        return {
            "query": request.query,
            "answer": answer,
            "sources": [result.dict() for result in search_results],
            "total_results": len(search_results),
            "vector_search_used": request.use_vector_search,
            "reranking_applied": request.enable_reranking,
            "embedding_dimension": len(query_embedding),
            "confidence": calculate_confidence(search_results)
        }

    except Exception as e:
        logger.error(f"智能搜索失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

def generate_answer(query: str, results: List[SearchResult]) -> str:
    """生成答案（简单版本，不使用LLM）"""
    if not results:
        return f"抱歉，没有找到与 '{query}' 相关的文档。"

    # 统计唯一文档数量
    unique_docs = set()
    for result in results:
        # 从文档路径或ID中提取唯一标识
        doc_id = result.id.split('_')[0] if '_' in str(result.id) else str(result.id)
        unique_docs.add(doc_id)

    unique_doc_count = len(unique_docs)
    total_results = len(results)

    if unique_doc_count == 1:
        return f"从 1 个文档中找到 {total_results} 个相关片段。最相关的是: {results[0].title}，匹配度为 {results[0].score:.2f}。"
    else:
        return f"从 {unique_doc_count} 个文档中找到 {total_results} 个相关片段。最相关的是: {results[0].title}，匹配度为 {results[0].score:.2f}。"


async def generate_answer_with_llm(query: str, results: List[SearchResult]) -> str:
    """使用LLM生成答案"""
    if not results:
        return f"抱歉，没有找到与 '{query}' 相关的文档。"

    # 构建上下文
    unique_docs = {}
    for result in results:
        doc_id = result.id.split('_')[0] if '_' in str(result.id) else str(result.id)
        if doc_id not in unique_docs:
            unique_docs[doc_id] = []
        unique_docs[doc_id].append(result)

    # 准备上下文内容
    context_parts = []
    for i, (doc_id, doc_results) in enumerate(unique_docs.items(), 1):
        best_result = doc_results[0]
        context_parts.append(f"""
【文档{i}】(ID: {doc_id}, 相关度: {best_result.score:.2f})
标题: {best_result.title}
内容: {best_result.content[:500]}
""")

    context = "\n".join(context_parts)

    # 构建LLM提示
    system_prompt = """你是一个专业的金融研究助手。基于提供的文档内容，准确回答用户的问题。

要求：
1. 答案必须严格基于提供的文档内容
2. 如果文档中没有相关信息，明确说明"文档中未提及"
3. 答案要准确、简洁、有条理
4. 引用具体的文档ID和数据
5. 不要编造信息"""

    user_prompt = f"""基于以下文档内容回答问题：

问题: {query}

文档内容:
{context}

请直接回答问题，不要重复问题本身。"""

    try:
        # 调用LLM
        llm_service = LLMService()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = await llm_service.chat_completion(
            messages=messages,
            temperature=0.3,
            max_tokens=2000,
            use_qwen=False
        )

        llm_answer = response.get("content", "").strip()

        # 添加来源信息
        unique_doc_count = len(unique_docs)
        total_frags = len(results)

        source_info = f"\n\n（以上答案基于 {unique_doc_count} 个文档中的 {total_frags} 个相关片段生成）"

        return llm_answer + source_info

    except Exception as e:
        logger.error(f"LLM生成答案失败: {e}")
        # 降级到简单答案
        return generate_answer(query, results)

def calculate_confidence(results: List[SearchResult]) -> float:
    """计算置信度"""
    if not results:
        return 0.1

    # 基于最高分数计算置信度
    max_score = max(result.score for result in results)
    return min(max_score * 100, 100.0)

@router.get("/search-status")
async def search_status():
    """获取搜索系统状态"""
    try:
        # 连接Milvus
        connections.connect(host='milvus', port='19530')
        collection = Collection("document_embeddings")

        # 测试嵌入服务
        embedding_service = SmartEmbeddingService()
        model_info = await embedding_service.get_model_info()

        return {
            "milvus_connected": True,
            "collection_entities": collection.num_entities,
            "embedding_service": model_info,
            "status": "healthy"
        }

    except Exception as e:
        logger.error(f"搜索状态检查失败: {e}")
        return {
            "milvus_connected": False,
            "collection_entities": 0,
            "embedding_service": None,
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/add-test-data")
async def add_test_data():
    """添加测试数据到搜索系统"""
    try:
        logger.info("添加测试数据...")

        # 初始化服务
        embedding_service = SmartEmbeddingService()
        connections.connect(host='milvus', port='19530')
        collection = Collection("document_embeddings")

        # 测试文档
        test_documents = [
            {
                "document_id": 2001,
                "chunk_id": 1,
                "content": "ChatGPT和AI技术在A股市场的投资机会分析。人工智能技术正在改变各个行业的格局。",
                "metadata": {"source": "test", "category": "AI投资"}
            },
            {
                "document_id": 2002,
                "chunk_id": 1,
                "content": "计算机行业的发展趋势：云计算、大数据、人工智能等技术的融合应用。",
                "metadata": {"source": "test", "category": "计算机行业"}
            },
            {
                "document_id": 2003,
                "chunk_id": 1,
                "content": "芯片制造产业链的投资价值分析。半导体行业是科技发展的核心驱动力。",
                "metadata": {"source": "test", "category": "芯片行业"}
            },
            {
                "document_id": 2004,
                "chunk_id": 1,
                "content": "新能源产业链的投资机会：电动汽车、储能技术、清洁能源等。",
                "metadata": {"source": "test", "category": "新能源"}
            },
            {
                "document_id": 2005,
                "chunk_id": 1,
                "content": "5G技术的商用化进程和相关的投资机会。通信技术的发展带来新的增长点。",
                "metadata": {"source": "test", "category": "5G通信"}
            }
        ]

        # 生成嵌入向量
        texts = [doc["content"] for doc in test_documents]
        embeddings = await embedding_service.encode(texts)

        # 准备数据
        import time
        current_time = int(time.time())

        test_data = [
            [doc["document_id"] for doc in test_documents],  # document_ids
            [doc["chunk_id"] for doc in test_documents],  # chunk_ids
            texts,  # content
            embeddings,  # embeddings
            [doc["metadata"] for doc in test_documents],  # metadata
            [current_time] * len(test_documents)  # created_at
        ]

        # 插入数据
        collection.insert(test_data)
        collection.flush()

        logger.info(f"成功插入 {len(test_documents)} 个测试文档")
        logger.info(f"集合总实体数: {collection.num_entities}")

        return {
            "success": True,
            "message": f"成功添加 {len(test_documents)} 个测试文档",
            "total_entities": collection.num_entities
        }

    except Exception as e:
        logger.error(f"添加测试数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加失败: {str(e)}")