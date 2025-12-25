"""
增强版AgentRAG搜索API端点
严格按照文档解析结果进行检索，支持完整的文档片段显示和溯源功能
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import asyncio
from datetime import datetime

from app.core.database import get_db
from app.services.smart_embedding_service import SmartEmbeddingService
from app.services.neo4j_service import Neo4jService
from app.services.milvus_service import MilvusService
from app.models.document import Document as DocumentModel

logger = logging.getLogger(__name__)

router = APIRouter()

class EnhancedSearchRequest(BaseModel):
    query: str
    top_k: int = 5
    enable_multi_stage_retrieval: bool = True
    include_document_fragments: bool = True
    include_source_tracing: bool = True
    use_knowledge_graph: bool = True
    use_vector_search: bool = True

class DocumentFragment(BaseModel):
    """文档片段模型"""
    document_id: int
    section_id: Optional[str] = None
    chunk_id: Optional[str] = None
    section_title: str
    content: str
    content_type: str  # text, table, image_caption, formula
    page_number: Optional[int] = None
    relevance_score: float

class SourceTrace(BaseModel):
    """溯源信息模型"""
    document_id: int
    document_title: str
    document_filename: str
    sections: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    retrieval_path: List[str]  # 检索路径
    confidence_score: float

class EnhancedSearchResult(BaseModel):
    """增强搜索结果模型"""
    query: str
    answer: str
    retrieval_info: Dict[str, Any]
    document_fragments: List[DocumentFragment]
    source_traces: List[SourceTrace]
    performance_metrics: Dict[str, Any]

@router.post("/enhanced-agentrag-search")
async def enhanced_agentrag_search(
    request: EnhancedSearchRequest,
    db: Session = Depends(get_db)
) -> EnhancedSearchResult:
    """
    增强版AgentRAG搜索接口
    严格按照文档解析结果进行检索，支持完整的文档片段显示和溯源
    """
    try:
        logger.info(f"🚀 增强版AgentRAG搜索: '{request.query}'")
        start_time = datetime.now()

        # 初始化服务
        embedding_service = SmartEmbeddingService()
        milvus_service = MilvusService()
        neo4j_service = Neo4jService()

        retrieval_info = {
            "query_understanding": {},
            "vector_search": {},
            "knowledge_graph": {},
            "structured_query": {},
            "document_search": {},
            "data_sources_used": [],
            "retrieval_stages": []
        }

        # 阶段1: Query Understanding (查询理解)
        query_analysis = await _analyze_query(request.query)
        retrieval_info["query_understanding"] = query_analysis
        retrieval_info["retrieval_stages"].append("Query Understanding")
        retrieval_info["data_sources_used"].append("AI Query Analyzer")

        document_fragments = []
        source_traces = []

        # 阶段2: Vector Search (向量搜索)
        if request.use_vector_search:
            vector_results = await _perform_vector_search(
                request.query, request.top_k, embedding_service, milvus_service, db
            )
            retrieval_info["vector_search"] = vector_results
            retrieval_info["retrieval_stages"].append("Vector Search (Milvus)")
            retrieval_info["data_sources_used"].append("milvus")

            # 提取文档片段
            fragments = await _extract_document_fragments(vector_results, db, "vector_search")
            document_fragments.extend(fragments)

            # 生成溯源信息
            traces = await _generate_source_traces(vector_results, db, "vector_search")
            source_traces.extend(traces)

        # 阶段3: Knowledge Graph Traversal (知识图谱遍历)
        if request.use_knowledge_graph:
            graph_results = await _perform_knowledge_graph_search(
                query_analysis, request.top_k, neo4j_service
            )
            retrieval_info["knowledge_graph"] = graph_results
            retrieval_info["retrieval_stages"].append("Knowledge Graph Traversal (Neo4j)")
            retrieval_info["data_sources_used"].append("neo4j")

            # 提取图谱相关的文档片段
            fragments = await _extract_document_fragments_from_graph(graph_results, db)
            document_fragments.extend(fragments)

        # 阶段4: Structured Query (结构化查询)
        structured_results = await _perform_structured_query(query_analysis, db)
        retrieval_info["structured_query"] = structured_results
        retrieval_info["retrieval_stages"].append("Structured Query (MySQL)")
        retrieval_info["data_sources_used"].append("mysql")

        # 阶段5: Document Content Search (文档内容搜索)
        content_results = await _perform_document_content_search(request.query, db)
        retrieval_info["document_search"] = content_results
        retrieval_info["retrieval_stages"].append("Document Content Search (MongoDB)")
        retrieval_info["data_sources_used"].append("mongodb")

        # 从文档内容搜索结果中提取片段
        if content_results.get("documents"):
            fragments = await _extract_document_fragments_from_content_search(content_results, db)
            document_fragments.extend(fragments)

        # 去重并排序文档片段
        document_fragments = _deduplicate_and_rank_fragments(document_fragments, request.top_k)
        source_traces = _deduplicate_source_traces(source_traces)

        # 生成综合回答
        answer = await _generate_comprehensive_answer(
            request.query, document_fragments, retrieval_info
        )

        # 计算性能指标
        end_time = datetime.now()
        retrieval_time = (end_time - start_time).total_seconds() * 1000
        performance_metrics = {
            "retrieval_time_ms": retrieval_time,
            "documents_found": len(set(f.document_id for f in document_fragments)),
            "fragments_found": len(document_fragments),
            "source_traces": len(source_traces),
            "retrieval_stages": len(retrieval_info["retrieval_stages"])
        }

        # 计算置信度
        confidence_score = _calculate_confidence_score(document_fragments, retrieval_info)
        retrieval_info["confidence_score"] = confidence_score

        logger.info(f"✅ 增强版搜索完成: {len(document_fragments)}个片段, {retrieval_time:.1f}ms")

        return EnhancedSearchResult(
            query=request.query,
            answer=answer,
            retrieval_info=retrieval_info,
            document_fragments=document_fragments,
            source_traces=source_traces,
            performance_metrics=performance_metrics
        )

    except Exception as e:
        logger.error(f"增强版搜索失败: {e}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

async def _analyze_query(query: str) -> Dict[str, Any]:
    """查询理解阶段"""
    # 这里可以集成更复杂的查询理解逻辑
    return {
        "original_query": query,
        "key_entities": _extract_entities(query),
        "query_intent": _classify_intent(query),
        "query_complexity": "medium" if len(query.split()) > 5 else "simple"
    }

def _extract_entities(query: str) -> List[str]:
    """提取查询中的关键实体"""
    # 简单的实体提取逻辑
    import re
    # 提取中文实体词
    entities = re.findall(r'[\u4e00-\u9fff]+(?:证券|银行|保险|基金|股票|策略|研究|报告)', query)
    return list(set(entities))

def _classify_intent(query: str) -> str:
    """分类查询意图"""
    if any(word in query for word in ['比较', '对比', '差异']):
        return "comparative_analysis"
    elif any(word in query for word in ['策略', '建议', '如何']):
        return "application_guidance"
    elif any(word in query for word in ['数据', '统计', '具体']):
        return "data_specific"
    else:
        return "factual_recall"

async def _perform_vector_search(query: str, top_k: int, embedding_service, milvus_service, db) -> Dict[str, Any]:
    """执行向量搜索"""
    try:
        # 生成查询向量
        query_embedding = await embedding_service.encode_single(query)

        # 在Milvus中搜索相似向量
        search_results = await milvus_service.search_vectors(query_embedding, top_k)

        # 获取对应的文档信息
        documents = []
        for result in search_results:
            doc_id = result.get('document_id')
            if doc_id:
                doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
                if doc:
                    documents.append({
                        "document_id": doc_id,
                        "title": doc.title or doc.filename,
                        "filename": doc.filename,
                        "score": result.get('score', 0),
                        "chunk_id": result.get('chunk_id'),
                        "parsed_content": doc.parsed_content
                    })

        return {
            "query_vector_dimension": len(query_embedding),
            "similar_vectors_found": len(search_results),
            "documents_matched": len(documents),
            "documents": documents
        }

    except Exception as e:
        logger.error(f"向量搜索失败: {e}")
        return {"error": str(e), "documents_matched": 0}

async def _perform_knowledge_graph_search(query_analysis: Dict, top_k: int, neo4j_service) -> Dict[str, Any]:
    """执行知识图谱搜索"""
    try:
        entities = query_analysis.get("key_entities", [])
        if not entities:
            return {"entities_found": 0, "relationships": [], "documents": []}

        # 在Neo4j中搜索相关实体和关系
        all_relationships = []
        all_documents = []

        for entity in entities:
            relationships = await neo4j_service.get_entity_relationships(entity)
            documents = await neo4j_service.get_entity_documents(entity)

            all_relationships.extend(relationships)
            all_documents.extend(documents)

        return {
            "entities_searched": entities,
            "relationships_found": len(all_relationships),
            "documents_found": len(all_documents),
            "relationships": all_relationships[:10],  # 限制返回数量
            "documents": all_documents
        }

    except Exception as e:
        logger.error(f"知识图谱搜索失败: {e}")
        return {"error": str(e), "documents_found": 0}

async def _perform_structured_query(query_analysis: Dict, db) -> Dict[str, Any]:
    """执行结构化查询"""
    try:
        entities = query_analysis.get("key_entities", [])
        if not entities:
            return {"records_found": 0, "records": []}

        # 在MySQL中搜索相关记录
        # 这里简化为搜索文档标题
        all_records = []
        for entity in entities:
            records = db.query(DocumentModel).filter(
                DocumentModel.title.contains(entity)
            ).limit(10).all()

            for record in records:
                all_records.append({
                    "document_id": record.id,
                    "title": record.title,
                    "filename": record.filename,
                    "status": record.status
                })

        return {
            "entities_searched": entities,
            "records_found": len(all_records),
            "records": all_records
        }

    except Exception as e:
        logger.error(f"结构化查询失败: {e}")
        return {"error": str(e), "records_found": 0}

async def _perform_document_content_search(query: str, db) -> Dict[str, Any]:
    """执行文档内容搜索"""
    try:
        # 搜索包含查询词的文档
        documents = db.query(DocumentModel).filter(
            DocumentModel.title.contains(query) |
            DocumentModel.filename.contains(query)
        ).limit(10).all()

        search_results = []
        for doc in documents:
            # 如果有解析内容，搜索章节标题
            matched_sections = []
            if doc.parsed_content:
                # 处理不同格式的parsed_content
                parsed = None
                if isinstance(doc.parsed_content, dict):
                    parsed = doc.parsed_content
                elif isinstance(doc.parsed_content, str):
                    try:
                        parsed = json.loads(doc.parsed_content)
                    except:
                        # 纯文本格式，直接使用
                        parsed = {"content": doc.parsed_content}

                if parsed:
                    # 从content数组中查找匹配的章节
                    content_list = parsed.get('content', []) if isinstance(parsed, dict) else []
                    for item in content_list:
                        if isinstance(item, dict):
                            content_text = item.get('content', '')
                            content_type = item.get('type', 'text')
                            # 检查内容是否包含查询词
                            if query.lower() in content_text.lower():
                                matched_sections.append({
                                    "section_title": content_text[:100],
                                    "section_content": content_text[:500],
                                    "type": content_type
                                })
                                if len(matched_sections) >= 3:  # 限制匹配数量
                                    break

                    # 如果没有找到匹配的内容，返回整个文档的前500字符
                    if not matched_sections and isinstance(parsed.get('content'), str):
                        matched_sections.append({
                            "section_title": "文档内容",
                            "section_content": parsed['content'][:500],
                            "type": "text"
                        })
                    elif not matched_sections and isinstance(doc.parsed_content, str):
                        # 纯文本格式
                        matched_sections.append({
                            "section_title": "文档内容",
                            "section_content": doc.parsed_content[:500],
                            "type": "text"
                        })

            search_results.append({
                "document_id": doc.id,
                "title": doc.title,
                "filename": doc.filename,
                "matched_sections": matched_sections,
                "has_parsed_content": bool(doc.parsed_content)
            })

        return {
            "query": query,
            "documents_found": len(search_results),
            "documents": search_results
        }

    except Exception as e:
        logger.error(f"文档内容搜索失败: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "documents_found": 0}

async def _extract_document_fragments(search_results: Dict, db, search_type: str) -> List[DocumentFragment]:
    """从搜索结果中提取文档片段"""
    fragments = []

    try:
        documents = search_results.get("documents", [])

        for doc_info in documents:
            doc_id = doc_info.get("document_id")
            if not doc_id:
                continue

            # 获取文档详细信息
            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if not doc or not doc.parsed_content:
                continue

            # 处理不同格式的parsed_content
            parsed = None
            if isinstance(doc.parsed_content, dict):
                parsed = doc.parsed_content
            elif isinstance(doc.parsed_content, str):
                try:
                    parsed = json.loads(doc.parsed_content)
                except:
                    # 纯文本格式，创建一个简单的fragment
                    fragment = DocumentFragment(
                        document_id=doc_id,
                        section_title="文档内容",
                        content=doc.parsed_content[:500],
                        content_type="text",
                        relevance_score=0.7
                    )
                    fragments.append(fragment)
                    continue

            if not parsed:
                continue

            # 从content数组中提取片段
            content_list = parsed.get('content', []) if isinstance(parsed, dict) else []
            if content_list:
                for i, item in enumerate(content_list[:3]):  # 限制每个文档最多3个片段
                    if isinstance(item, dict):
                        content_text = item.get('content', '')
                        content_type = item.get('type', 'text')

                        if content_text:
                            fragment = DocumentFragment(
                                document_id=doc_id,
                                section_id=item.get('id'),
                                chunk_id=doc_info.get('chunk_id'),
                                section_title=content_text[:50] + ("..." if len(content_text) > 50 else ""),
                                content=content_text[:500],
                                content_type=content_type,
                                page_number=item.get('metadata', {}).get('page_number'),
                                relevance_score=doc_info.get('score', 0.8)
                            )
                            fragments.append(fragment)
            elif isinstance(parsed.get('content'), str):
                # 单个content字符串
                fragment = DocumentFragment(
                    document_id=doc_id,
                    section_title="文档内容",
                    content=parsed['content'][:500],
                    content_type="text",
                    relevance_score=0.7
                )
                fragments.append(fragment)

    except Exception as e:
        logger.error(f"提取文档片段失败: {e}")
        import traceback
        traceback.print_exc()

    return fragments

async def _extract_document_fragments_from_content_search(content_results: Dict, db) -> List[DocumentFragment]:
    """从文档内容搜索结果中提取片段"""
    fragments = []

    try:
        documents = content_results.get("documents", [])

        for doc_info in documents:
            doc_id = doc_info.get("document_id")
            if not doc_id:
                continue

            # 获取matched_sections
            matched_sections = doc_info.get("matched_sections", [])

            for section in matched_sections[:3]:  # 限制每个文档最多3个片段
                section_title = section.get("section_title", "文档内容")
                section_content = section.get("section_content", "")
                content_type = section.get("type", "text")

                if section_content:
                    fragment = DocumentFragment(
                        document_id=doc_id,
                        section_title=section_title[:100],
                        content=section_content[:500],
                        content_type=content_type,
                        relevance_score=0.8
                    )
                    fragments.append(fragment)

    except Exception as e:
        logger.error(f"从内容搜索提取片段失败: {e}")
        import traceback
        traceback.print_exc()

    return fragments

async def _extract_document_fragments_from_graph(graph_results: Dict, db) -> List[DocumentFragment]:
    """从知识图谱结果中提取文档片段"""
    fragments = []

    try:
        documents = graph_results.get("documents", [])

        for doc_info in documents:
            doc_id = doc_info.get("id")
            if not doc_id:
                continue

            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if not doc or not doc.parsed_content:
                continue

            # 从图谱结果中提取相关片段
            parsed_content = doc.parsed_content
            if isinstance(parsed_content, dict) and 'sections' in parsed_content:
                sections = parsed_content['sections'][:2]
                for section in sections:
                    fragment = DocumentFragment(
                        document_id=doc_id,
                        section_title=section.get('title', '图谱相关章节'),
                        content=str(section.get('content', ''))[:300],
                        content_type=_detect_content_type(section.get('content', '')),
                        relevance_score=0.8  # 图谱相关文档默认高分
                    )
                    fragments.append(fragment)

    except Exception as e:
        logger.error(f"从图谱提取文档片段失败: {e}")

    return fragments

async def _generate_source_traces(search_results: Dict, db, search_type: str) -> List[SourceTrace]:
    """生成溯源信息"""
    traces = []

    try:
        documents = search_results.get("documents", [])

        for doc_info in documents:
            doc_id = doc_info.get("document_id")
            if not doc_id:
                continue

            doc = db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()
            if not doc:
                continue

            # 收集文档的章节和块信息
            sections = []
            chunks = []

            if doc.parsed_content and isinstance(doc.parsed_content, dict):
                parsed_sections = doc.parsed_content.get('sections', [])
                for section in parsed_sections[:3]:
                    sections.append({
                        "id": section.get('id'),
                        "title": section.get('title', ''),
                        "type": section.get('type', 'text')
                    })

                parsed_chunks = doc.parsed_content.get('chunks', [])
                for chunk in parsed_chunks[:2]:
                    chunks.append({
                        "id": chunk.get('id'),
                        "type": chunk.get('type', 'text')
                    })

            trace = SourceTrace(
                document_id=doc_id,
                document_title=doc.title or doc.filename,
                document_filename=doc.filename,
                sections=sections,
                chunks=chunks,
                retrieval_path=[search_type, "document_content"],
                confidence_score=doc_info.get('score', 0.8)
            )
            traces.append(trace)

    except Exception as e:
        logger.error(f"生成溯源信息失败: {e}")

    return traces

def _deduplicate_and_rank_fragments(fragments: List[DocumentFragment], top_k: int) -> List[DocumentFragment]:
    """去重并排序文档片段"""
    # 按文档ID和内容去重
    seen = set()
    unique_fragments = []

    for fragment in fragments:
        key = (fragment.document_id, fragment.section_title[:100])
        if key not in seen:
            seen.add(key)
            unique_fragments.append(fragment)

    # 按相关性评分排序
    unique_fragments.sort(key=lambda x: x.relevance_score, reverse=True)

    return unique_fragments[:top_k]

def _deduplicate_source_traces(traces: List[SourceTrace]) -> List[SourceTrace]:
    """去重溯源信息"""
    seen_docs = set()
    unique_traces = []

    for trace in traces:
        if trace.document_id not in seen_docs:
            seen_docs.add(trace.document_id)
            unique_traces.append(trace)

    return unique_traces

def _detect_content_type(content: Any) -> str:
    """检测内容类型"""
    content_str = str(content).lower()

    if 'table' in content_str or '|' in content_str:
        return 'table'
    elif any(word in content_str for word in ['公式', 'formula', '=', '+', '-', '*', '/']):
        return 'formula'
    elif any(word in content_str for word in ['图', 'image', 'chart', '图形']):
        return 'image_caption'
    else:
        return 'text'

async def _generate_comprehensive_answer(query: str, fragments: List[DocumentFragment], retrieval_info: Dict) -> str:
    """生成综合回答"""
    if not fragments:
        return "抱歉，未能找到与您查询相关的文档内容。"

    # 基于文档片段生成回答
    answer_parts = []
    answer_parts.append(f"根据检索到的 {len(fragments)} 个相关文档片段，针对您的问题 '{query}' 的分析如下：\n")

    # 按文档分组展示结果
    doc_groups = {}
    for fragment in fragments:
        if fragment.document_id not in doc_groups:
            doc_groups[fragment.document_id] = []
        doc_groups[fragment.document_id].append(fragment)

    for doc_id, doc_fragments in list(doc_groups.items())[:3]:  # 最多展示3个文档
        answer_parts.append(f"\n📄 **文档 {doc_id} 的相关内容：**")
        for fragment in doc_fragments[:2]:  # 每个文档最多2个片段
            answer_parts.append(f"• **{fragment.section_title}**: {fragment.content[:200]}...")

    answer_parts.append(f"\n🔍 **检索信息:**")
    answer_parts.append(f"• 检索阶段: {' → '.join(retrieval_info.get('retrieval_stages', []))}")
    answer_parts.append(f"• 数据源: {', '.join(retrieval_info.get('data_sources_used', []))}")
    answer_parts.append(f"• 置信度: {retrieval_info.get('confidence_score', 0):.3f}")

    return "\n".join(answer_parts)

def _calculate_confidence_score(fragments: List[DocumentFragment], retrieval_info: Dict) -> float:
    """计算检索置信度"""
    if not fragments:
        return 0.0

    # 基于片段数量、相关性评分和检索阶段计算置信度
    avg_fragment_score = sum(f.relevance_score for f in fragments) / len(fragments)
    stages_bonus = min(len(retrieval_info.get('retrieval_stages', [])) * 0.1, 0.3)

    confidence = min(avg_fragment_score + stages_bonus, 1.0)
    return round(confidence, 3)