"""
RAG问答服务 - 基于In-Context Learning

功能：
1. 检索相关文档块（置信度>0.7）
2. 构建In-Context Learning提示词
3. 调用DeepSeek生成答案
4. 返回来源引用、信任度解释、检索路径
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.services.embedding_service import embedding_service
from pymilvus import connections, Collection

logger = logging.getLogger(__name__)


class RAGQuestionAnsweringService:
    """RAG问答服务"""

    def __init__(self):
        self.embedding_service = embedding_service
        self.milvus_collection = None

    async def _connect_milvus(self):
        """连接Milvus"""
        try:
            connections.connect(
                alias="default",
                host=settings.milvus_host,
                port=settings.milvus_port
            )
            self.milvus_collection = Collection(
                settings.milvus_collection_name
            )
            self.milvus_collection.load()
            logger.info("Milvus连接成功")
        except Exception as e:
            logger.error(f"Milvus连接失败: {e}")
            raise

    async def _search_relevant_chunks(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        检索相关文档块

        Args:
            query_embedding: 查询向量的embedding
            top_k: 返回的top-k结果
            min_confidence: 最小置信度阈值

        Returns:
            相关文档块列表
        """
        try:
            if not self.milvus_collection:
                await self._connect_milvus()

            # 向量搜索
            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 16}
            }

            results = self.milvus_collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=top_k,
                expr=None,
                output_fields=["chunk_id", "document_id", "content", "metadata"]
            )

            # 处理结果，过滤低置信度的
            relevant_chunks = []
            for hit in results[0]:
                confidence = float(hit.score)  # IP距离，转换为置信度
                confidence = max(0, min(1, confidence))  # 归一化到[0,1]

                if confidence >= min_confidence:
                    relevant_chunks.append({
                        "chunk_id": hit.entity.get("chunk_id"),
                        "document_id": hit.entity.get("document_id"),
                        "content": hit.entity.get("content"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "confidence": confidence,
                        "distance": hit.distance
                    })

            logger.info(f"检索到 {len(relevant_chunks)} 个相关块（置信度>={min_confidence}）")
            return relevant_chunks

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def _get_document_details(
        self,
        db: Session,
        document_ids: List[int]
    ) -> Dict[int, Dict[str, Any]]:
        """获取文档详细信息"""
        try:
            placeholders = ",".join([str(id) for id in document_ids])
            query = text(f"""
                SELECT id, title, filename, file_path, created_at
                FROM documents
                WHERE id IN ({placeholders})
            """)

            result = db.execute(query)
            documents = {}
            for row in result:
                documents[row.id] = {
                    "id": row.id,
                    "title": row.title or row.filename,
                    "filename": row.filename,
                    "file_path": row.file_path,
                    "created_at": row.created_at.isoformat() if row.created_at else None
                }

            return documents

        except Exception as e:
            logger.error(f"获取文档详情失败: {e}")
            return {}

    def _build_icl_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        document_details: Dict[int, Dict[str, Any]]
    ) -> str:
        """
        构建In-Context Learning提示词

        Args:
            question: 用户问题
            context_chunks: 检索到的上下文块
            document_details: 文档详情

        Returns:
            完整的提示词
        """
        # 构建上下文片段
        context_blocks = []
        for idx, chunk in enumerate(context_chunks, 1):
            doc_id = chunk["document_id"]
            doc_info = document_details.get(doc_id, {})
            doc_title = doc_info.get("title", "未知文档")
            confidence = chunk["confidence"]

            context_block = f"""
【上下文片段 {idx}】
来源文档：{doc_title}
置信度：{confidence:.2%}

{chunk["content"]}
"""
            context_blocks.append(context_block)

        contexts = "\n".join(context_blocks)

        # 构建完整的提示词
        prompt = f"""你是一位专业的金融领域分析师和顾问。请基于提供的上下文片段回答用户的问题。

# 用户问题
{question}

# 相关上下文片段
{contexts}

# 回答要求

1. **答案质量**：
   - 直接回答用户的问题
   - 答案必须基于提供的上下文片段
   - 如果上下文片段中没有足够信息，请明确说明"根据现有文档无法确定"

2. **引用要求**：
   - 在答案中明确标注引用的上下文片段编号，如 [上下文片段 1]
   - 如果多个片段支持同一观点，全部引用，如 [上下文片段 1,2,3]

3. **推理过程**：
   - 提供清晰的推理逻辑
   - 说明你是如何从上下文片段中得出结论的
   - 如果需要推理，请标注推理步骤

4. **回答格式**：

【答案】
[你的答案内容]

【推理过程】
[你的推理逻辑和步骤]

【来源引用】
- 上下文片段 1: [文档名称] - 置信度 XX%
- 上下文片段 2: [文档名称] - 置信度 XX%
...

【可信度评估】
- 证据强度：[强/中/弱]
- 证据覆盖度：[完整/部分/不足]
- 答案确定性：[高/中/低]

请严格按照上述格式回答。
"""
        return prompt

    async def _call_deepseek(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000
    ) -> str:
        """
        调用DeepSeek API生成答案

        Args:
            prompt: 提示词
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            生成的答案
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url
            )

            response = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": "你是一位专业的金融分析师，擅长基于文档回答问题。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )

            answer = response.choices[0].message.content
            logger.info(f"DeepSeek生成答案成功，长度: {len(answer)}")
            return answer

        except Exception as e:
            logger.error(f"DeepSeek调用失败: {e}")
            raise

    async def answer_question(
        self,
        question: str,
        top_k: int = 10,
        min_confidence: float = 0.7,
        db: Session = None
    ) -> Dict[str, Any]:
        """
        回答问题的完整流程

        Args:
            question: 用户问题
            top_k: 检索的文档块数量
            min_confidence: 最小置信度阈值
            db: 数据库会话

        Returns:
            包含答案、来源、信任度、检索路径的完整响应
        """
        start_time = datetime.now()

        try:
            # 1. 获取数据库会话
            if not db:
                db_gen = get_db()
                db = next(db_gen)

            # 2. 对问题进行embedding
            logger.info(f"开始处理问题: {question}")
            query_embedding = await self.embedding_service.get_embedding(question)

            # 3. 检索相关文档块
            relevant_chunks = await self._search_relevant_chunks(
                query_embedding,
                top_k=top_k,
                min_confidence=min_confidence
            )

            if not relevant_chunks:
                return {
                    "question": question,
                    "answer": "抱歉，在现有文档中未找到相关答案。",
                    "sources": [],
                    "trust_explanation": {
                        "evidence_strength": "无",
                        "evidence_coverage": "不足",
                        "answer_certainty": "低",
                        "reason": "未检索到相关文档块"
                    },
                    "retrieval_path": {
                        "query_embedding_generated": True,
                        "chunks_retrieved": 0,
                        "chunks_after_filtering": 0,
                        "documents_matched": 0
                    },
                    "execution_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat()
                }

            # 4. 获取文档详情
            document_ids = list(set([c["document_id"] for c in relevant_chunks]))
            document_details = await self._get_document_details(db, document_ids)

            # 5. 构建ICL提示词
            prompt = self._build_icl_prompt(
                question,
                relevant_chunks,
                document_details
            )

            # 6. 调用DeepSeek生成答案
            answer = await self._call_deepseek(prompt)

            # 7. 构建检索路径可视化
            retrieval_path = {
                "step1_query_embedding": {
                    "action": "问题向量化",
                    "input": question,
                    "model": "text-embedding-v4",
                    "status": "success"
                },
                "step2_vector_search": {
                    "action": "向量检索",
                    "database": "Milvus",
                    "top_k": top_k,
                    "retrieved": len(relevant_chunks),
                    "status": "success"
                },
                "step3_confidence_filtering": {
                    "action": "置信度过滤",
                    "threshold": min_confidence,
                    "passed": len(relevant_chunks),
                    "status": "success"
                },
                "step4_document_lookup": {
                    "action": "文档详情查询",
                    "database": "MySQL",
                    "documents_found": len(document_details),
                    "status": "success"
                },
                "step5_prompt_construction": {
                    "action": "提示词构建",
                    "method": "In-Context Learning",
                    "context_blocks": len(relevant_chunks),
                    "status": "success"
                },
                "step6_answer_generation": {
                    "action": "答案生成",
                    "model": "deepseek-chat",
                    "tokens": len(answer),
                    "status": "success"
                }
            }

            # 8. 构建来源引用
            sources = []
            for chunk in relevant_chunks:
                doc_id = chunk["document_id"]
                doc_info = document_details.get(doc_id, {})

                sources.append({
                    "chunk_id": chunk["chunk_id"],
                    "document_id": doc_id,
                    "document_title": doc_info.get("title", "未知文档"),
                    "document_filename": doc_info.get("filename", ""),
                    "content": chunk["content"][:200] + "..." if len(chunk["content"]) > 200 else chunk["content"],
                    "confidence": chunk["confidence"],
                    "metadata": chunk["metadata"]
                })

            # 9. 计算信任度指标
            avg_confidence = sum([c["confidence"] for c in relevant_chunks]) / len(relevant_chunks)

            trust_explanation = {
                "evidence_strength": "强" if avg_confidence > 0.85 else "中" if avg_confidence > 0.75 else "弱",
                "evidence_coverage": "完整" if len(relevant_chunks) >= 5 else "部分" if len(relevant_chunks) >= 3 else "不足",
                "answer_certainty": "高" if avg_confidence > 0.85 else "中" if avg_confidence > 0.75 else "低",
                "average_confidence": round(avg_confidence, 3),
                "source_count": len(relevant_chunks),
                "document_count": len(document_details),
                "reasoning": f"基于{len(relevant_chunks)}个高置信度（平均{avg_confidence:.2%})的上下文片段生成，来自{len(document_details)}份文档"
            }

            execution_time = (datetime.now() - start_time).total_seconds()

            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "trust_explanation": trust_explanation,
                "retrieval_path": retrieval_path,
                "execution_time": round(execution_time, 2),
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"问答处理失败: {e}", exc_info=True)
            raise


# 全局服务实例
_rag_qa_service = None


def get_rag_qa_service() -> RAGQuestionAnsweringService:
    """获取RAG问答服务实例"""
    global _rag_qa_service
    if _rag_qa_service is None:
        _rag_qa_service = RAGQuestionAnsweringService()
    return _rag_qa_service
