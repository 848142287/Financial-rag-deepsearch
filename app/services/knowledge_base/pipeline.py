"""
知识库构建流水线
实现多模态文档处理、实体抽取、向量化存储的完整流水线
"""

import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from ..document_parser import document_parser
from ..embedding_service import embedding_service
from ..milvus_service import milvus_service
from ..neo4j_service import neo4j_service
from ..tasks.background import celery_app
from ...core.database import get_db
from ...models.document import Document, DocumentChunk, Entity

logger = logging.getLogger(__name__)


class KnowledgeBasePipeline:
    """知识库构建流水线管理器"""

    def __init__(self):
        self.supported_formats = {
            'pdf', 'docx', 'xlsx', 'txt', 'md',
            'jpg', 'jpeg', 'png', 'tiff'
        }

    async def process_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        处理文档的完整流水线

        Args:
            file_path: 文件路径
            document_id: 文档ID，如果不提供则生成新的
            config: 处理配置

        Returns:
            处理结果
        """
        if not document_id:
            document_id = str(uuid.uuid4())

        # 默认配置
        default_config = {
            'use_ocr': True,
            'extract_images': True,
            'extract_tables': True,
            'chunk_size': 512,
            'chunk_overlap': 50,
            'extract_entities': True,
            'generate_embeddings': True,
            'enable_parallel': True
        }

        if config:
            default_config.update(config)

        logger.info(f"开始处理文档: {file_path}, ID: {document_id}")

        # 记录开始时间
        start_time = time.time()

        try:
            # 创建数据库记录
            async with get_db() as db:
                document = await self._create_document_record(
                    db, document_id, file_path, default_config
                )

                # 第一阶段：文档解析
                parse_result = await self._parse_document(file_path, default_config)

                # 第二阶段：文本分块
                chunks = await self._chunk_text(parse_result, default_config)

                # 第三阶段：实体抽取
                entities = []
                if default_config['extract_entities']:
                    entities = await self._extract_entities(chunks, default_config)

                # 第四阶段：向量化存储
                if default_config['generate_embeddings']:
                    await self._generate_and_store_embeddings(
                        document_id, chunks, default_config
                    )

                # 第五阶段：知识图谱构建
                if entities:
                    await self._build_knowledge_graph(document_id, entities, default_config)

                # 更新文档状态
                await self._update_document_status(
                    db, document_id, 'completed',
                    chunks_count=len(chunks),
                    entities_count=len(entities)
                )

                logger.info(f"文档处理完成: {document_id}")

                # 计算处理时间
                end_time = time.time()
                processing_time = end_time - start_time

                return {
                    'document_id': document_id,
                    'status': 'success',
                    'chunks_count': len(chunks),
                    'entities_count': len(entities),
                    'processing_time': processing_time
                }

        except Exception as e:
            logger.error(f"文档处理失败: {document_id}, 错误: {e}")

            # 更新失败状态
            try:
                async with get_db() as db:
                    await self._update_document_status(
                        db, document_id, 'failed', error_message=str(e)
                    )
            except Exception as db_error:
                logger.error(f"更新文档状态失败: {db_error}")

            raise

    async def batch_process_documents(
        self,
        file_paths: List[str],
        config: Optional[Dict[str, Any]] = None,
        max_workers: int = 3
    ) -> List[Dict[str, Any]]:
        """
        批量处理文档

        Args:
            file_paths: 文件路径列表
            config: 处理配置
            max_workers: 最大并行处理数

        Returns:
            处理结果列表
        """
        logger.info(f"开始批量处理 {len(file_paths)} 个文档")

        if config and config.get('enable_parallel', True):
            # 并行处理
            semaphore = asyncio.Semaphore(max_workers)

            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self.process_document(file_path, config=config)

            tasks = [process_with_semaphore(fp) for fp in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"文档 {file_paths[i]} 处理失败: {result}")
                    processed_results.append({
                        'document_id': None,
                        'file_path': file_paths[i],
                        'status': 'failed',
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)

            return processed_results
        else:
            # 串行处理
            results = []
            for file_path in file_paths:
                try:
                    result = await self.process_document(file_path, config=config)
                    results.append(result)
                except Exception as e:
                    logger.error(f"文档 {file_path} 处理失败: {e}")
                    results.append({
                        'document_id': None,
                        'file_path': file_path,
                        'status': 'failed',
                        'error': str(e)
                    })
            return results

    async def _create_document_record(
        self,
        db,
        document_id: str,
        file_path: str,
        config: Dict[str, Any]
    ) -> Document:
        """创建文档记录"""
        document = Document(
            id=document_id,
            filename=file_path.split('/')[-1],
            file_path=file_path,
            status='processing',
            config=config,
            created_at=datetime.utcnow()
        )

        db.add(document)
        await db.commit()
        await db.refresh(document)

        return document

    async def _parse_document(
        self,
        file_path: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析文档"""
        return await document_parser.parse_document(
            file_path=file_path,
            use_ocr=config.get('use_ocr', True),
            extract_images=config.get('extract_images', True),
            extract_tables=config.get('extract_tables', True)
        )

    async def _chunk_text(
        self,
        parse_result: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """文本分块"""
        text_content = []

        # 合并所有文本内容
        for item in parse_result['text_content']:
            if isinstance(item, str):
                text_content.append(item)
            elif isinstance(item, dict):
                text_content.append(item.get('content', str(item)))

        # 添加表格文本
        for table in parse_result['tables']:
            table_text = table.get('text', str(table))
            text_content.append(f"\n表格内容:\n{table_text}")

        # 使用文档解析器的分块功能
        chunks = document_parser.chunk_text(
            text_content,
            chunk_size=config.get('chunk_size', 512),
            chunk_overlap=config.get('chunk_overlap', 50)
        )

        # 为每个chunk添加额外的元数据
        for i, chunk in enumerate(chunks):
            chunk['metadata'].update({
                'document_id': parse_result.get('document_id'),
                'parser': parse_result['metadata'].get('parser'),
                'has_images': len(parse_result['images']) > 0,
                'has_tables': len(parse_result['tables']) > 0
            })

        return chunks

    async def _extract_entities(
        self,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """抽取实体"""
        from app.services.knowledge_base.entity_extractor import EntityExtractor
        from app.services.llm_service import llm_service
        import re

        try:
            # 初始化实体抽取器
            entity_extractor = EntityExtractor(config.get("entity_config", {}))
            min_confidence = config.get("min_entity_confidence", 0.7)

            all_entities = []

            for chunk in chunks:
                content = chunk["content"]
                chunk_id = chunk.get("chunk_id", chunk.get("metadata", {}).get("chunk_index", 0))

                # 使用多种方法抽取实体
                chunk_entities = []

                # 1. 使用已有的实体抽取器
                try:
                    extracted = await entity_extractor.extract_entities(content, chunk_id)
                    chunk_entities.extend(extracted)
                except Exception as e:
                    logger.warning(f"Entity extractor failed for chunk {chunk_id}: {e}")

                # 2. 使用LLM进行高级实体抽取（如果没有得到足够的结果）
                if len(chunk_entities) < 3:
                    try:
                        llm_entities = await self._extract_entities_with_llm(
                            content, chunk_id, llm_service
                        )
                        chunk_entities.extend(llm_entities)
                    except Exception as e:
                        logger.warning(f"LLM entity extraction failed for chunk {chunk_id}: {e}")

                # 3. 使用正则表达式抽取特定实体（数字、公司等）
                regex_entities = self._extract_entities_with_regex(content, chunk_id)
                chunk_entities.extend(regex_entities)

                # 去重并过滤低置信度实体
                unique_entities = self._deduplicate_entities(chunk_entities)
                filtered_entities = [
                    e for e in unique_entities
                    if e.get("confidence", 0) >= min_confidence
                ]

                # 添加chunk_id引用
                for entity in filtered_entities:
                    entity["chunk_id"] = chunk_id
                    entity["document_id"] = chunk.get("document_id")

                all_entities.extend(filtered_entities)

            logger.info(f"Extracted {len(all_entities)} entities from {len(chunks)} chunks")
            return all_entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            # Fallback to simple keyword extraction
            return self._fallback_entity_extraction(chunks)

    async def _extract_entities_with_llm(
        self, content: str, chunk_id: int, llm_service
    ) -> List[Dict[str, Any]]:
        """使用LLM抽取实体"""
        prompt = f"""
        从以下金融文本中抽取实体，包括：
        1. 公司名称
        2. 股票代码
        3. 财务指标
        4. 人名
        5. 地点
        6. 时间
        7. 金额/数字

        文本：{content}

        请以JSON格式返回实体列表，每个实体包含：
        - text: 实体文本
        - type: 实体类型
        - confidence: 置信度(0-1)

        返回格式：
        {{
            "entities": [
                {{
                    "text": "实体文本",
                    "type": "实体类型",
                    "confidence": 0.9
                }}
            ]
        }}
        """

        try:
            response = await llm_service.generate(prompt)
            # 解析LLM响应
            import json
            result = json.loads(response)

            entities = []
            for entity in result.get("entities", []):
                entities.append({
                    "text": entity["text"],
                    "type": entity["type"],
                    "confidence": entity["confidence"],
                    "source": "llm"
                })

            return entities

        except Exception as e:
            logger.error(f"LLM entity extraction error: {e}")
            return []

    def _extract_entities_with_regex(self, content: str, chunk_id: int) -> List[Dict[str, Any]]:
        """使用正则表达式抽取实体"""
        entities = []

        # 抽取金额
        money_pattern = r'(\d+(?:\.\d+)?[万亿千百]?(?:元|美元|欧元|港币|人民币))'
        money_matches = re.finditer(money_pattern, content)
        for match in money_matches:
            entities.append({
                "text": match.group(1),
                "type": "MONEY",
                "confidence": 0.95,
                "source": "regex"
            })

        # 抽取百分比
        percent_pattern = r'(\d+(?:\.\d+)?%)'
        percent_matches = re.finditer(percent_pattern, content)
        for match in percent_matches:
            entities.append({
                "text": match.group(1),
                "type": "PERCENTAGE",
                "confidence": 0.9,
                "source": "regex"
            })

        # 抽取日期
        date_pattern = r'(\d{4}年\d{1,2}月\d{1,2}日|\d{4}-\d{1,2}-\d{1,2})'
        date_matches = re.finditer(date_pattern, content)
        for match in date_matches:
            entities.append({
                "text": match.group(1),
                "type": "DATE",
                "confidence": 0.9,
                "source": "regex"
            })

        return entities

    def _deduplicate_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重实体"""
        seen = set()
        unique_entities = []

        for entity in entities:
            key = (entity["text"].lower(), entity["type"])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # 保留置信度更高的
                for i, existing in enumerate(unique_entities):
                    if (existing["text"].lower() == entity["text"].lower() and
                        existing["type"] == entity["type"]):
                        if entity.get("confidence", 0) > existing.get("confidence", 0):
                            unique_entities[i] = entity
                        break

        return unique_entities

    def _fallback_entity_extraction(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """备用实体抽取方法"""
        financial_keywords = {
            '公司': 'COMPANY',
            '股票': 'STOCK',
            '债券': 'BOND',
            '基金': 'FUND',
            '期货': 'FUTURES',
            '期权': 'OPTION',
            '收入': 'REVENUE',
            '利润': 'PROFIT',
            '亏损': 'LOSS',
            '资产': 'ASSET',
            '负债': 'LIABILITY',
            '现金流': 'CASH_FLOW',
            '市盈率': 'PE_RATIO',
            '市净率': 'PB_RATIO',
            'ROE': 'ROE',
            'ROA': 'ROA',
            '净利润率': 'NET_PROFIT_MARGIN'
        }

        entities = []
        for chunk in chunks:
            content = chunk["content"]
            chunk_id = chunk.get("chunk_id", 0)

            for keyword, entity_type in financial_keywords.items():
                if keyword in content:
                    entities.append({
                        "text": keyword,
                        "type": entity_type,
                        "confidence": 0.6,
                        "chunk_id": chunk_id,
                        "source": "fallback"
                    })

        return entities

    async def _generate_and_store_embeddings(
        self,
        document_id: str,
        chunks: List[Dict[str, Any]],
        config: Dict[str, Any]
    ):
        """生成并存储向量嵌入"""
        try:
            # 准备文本内容
            texts = [chunk['content'] for chunk in chunks]

            # 批量生成嵌入
            embeddings = await embedding_service.generate_embeddings(texts)

            # 准备Milvus数据
            milvus_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                milvus_data.append({
                    'id': f"{document_id}_{i}",
                    'document_id': document_id,
                    'chunk_index': i,
                    'content': chunk['content'],
                    'embedding': embedding,
                    'metadata': chunk['metadata']
                })

            # 批量插入Milvus
            await milvus_service.insert_documents(milvus_data)

            logger.info(f"成功存储 {len(milvus_data)} 个向量")

        except Exception as e:
            logger.error(f"向量存储失败: {e}")
            raise

    async def _build_knowledge_graph(
        self,
        document_id: str,
        entities: List[Dict[str, Any]],
        config: Dict[str, Any]
    ):
        """构建知识图谱"""
        try:
            # 准备节点和关系
            nodes = []
            relationships = []

            # 创建实体节点
            for entity in entities:
                nodes.append({
                    'name': entity['text'],
                    'type': entity['type'],
                    'confidence': entity['confidence'],
                    'document_id': document_id,
                    'source_chunk': entity.get('chunk_id')
                })

            # 实体关系抽取
            relationships = await self._extract_entity_relationships(entities, chunks)

            # 批量插入Neo4j
            if nodes:
                await neo4j_service.create_entities(nodes)

            if relationships:
                await neo4j_service.create_relationships(relationships)

            logger.info(f"成功创建 {len(nodes)} 个节点, {len(relationships)} 个关系")

        except Exception as e:
            logger.error(f"知识图谱构建失败: {e}")
            # 不抛出异常，因为这不是核心功能

    async def _update_document_status(
        self,
        db,
        document_id: str,
        status: str,
        chunks_count: Optional[int] = None,
        entities_count: Optional[int] = None,
        error_message: Optional[str] = None
    ):
        """更新文档状态"""
        try:
            document = await db.get(Document, document_id)
            if document:
                document.status = status
                document.updated_at = datetime.utcnow()

                if chunks_count is not None:
                    document.chunks_count = chunks_count
                if entities_count is not None:
                    document.entities_count = entities_count
                if error_message:
                    document.error_message = error_message

                await db.commit()

        except Exception as e:
            logger.error(f"更新文档状态失败: {e}")


# 异步任务处理
@celery_app.task(bind=True)
def process_document_async(self, file_path: str, config: Optional[Dict] = None):
    """异步文档处理任务"""
    pipeline = KnowledgeBasePipeline()

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(
            pipeline.process_document(file_path, config=config)
        )

        loop.close()

        return {
            'status': 'success',
            'result': result,
            'task_id': self.request.id
        }

    except Exception as e:
        logger.error(f"异步文档处理失败: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'task_id': self.request.id
        }

    async def _extract_entity_relationships(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """抽取实体关系"""
        relationships = []

        try:
            # 方法1: 基于规则的关系抽取
            rule_relationships = self._extract_relationships_by_rules(entities, chunks)
            relationships.extend(rule_relationships)

            # 方法2: 基于共现的关系抽取
            cooccurrence_relationships = self._extract_cooccurrence_relationships(entities, chunks)
            relationships.extend(cooccurrence_relationships)

            # 方法3: 使用LLM进行关系抽取（可选）
            if len(entities) <= 50:  # 只对较少的实体使用LLM，避免成本过高
                try:
                    llm_relationships = await self._extract_relationships_by_llm(entities, chunks)
                    relationships.extend(llm_relationships)
                except Exception as e:
                    logger.warning(f"LLM关系抽取失败: {e}")

            # 去重
            unique_relationships = []
            seen = set()
            for rel in relationships:
                key = (rel['source'], rel['target'], rel['type'])
                if key not in seen:
                    seen.add(key)
                    unique_relationships.append(rel)

            return unique_relationships

        except Exception as e:
            logger.error(f"实体关系抽取失败: {e}")
            return relationships

    def _extract_relationships_by_rules(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """基于规则抽取关系"""
        relationships = []

        # 定义金融领域的关系规则
        relationship_patterns = [
            {
                'source_type': 'COMPANY',
                'target_type': 'PERSON',
                'relation_type': 'HAS_EXECUTIVE',
                'patterns': [
                    r'(\S+(?:银行|保险|证券|集团|公司))\s*(?:的)?(?:董事长|总裁|CEO|总经理|行长)\s*([^\s，。；！？]+)',
                    r'([^\s，。；！？]+)\s*(?:任|担任)\s*(\S+(?:银行|保险|证券|集团|公司))\s*(?:的)?(?:董事长|总裁|CEO|总经理|行长)'
                ]
            },
            {
                'source_type': 'COMPANY',
                'target_type': 'STOCK',
                'relation_type': 'HAS_STOCK',
                'patterns': [
                    r'(\S+(?:银行|保险|证券|集团|公司))\s*(?:股票代码|代码)\s*(\d{6})',
                    r'(\d{6})\s*(?:是)?\s*(\S+(?:银行|保险|证券|集团|公司))\s*(?:的)?(?:股票|代码)'
                ]
            }
        ]

        import re

        # 合并所有文本用于关系抽取
        full_text = ' '.join([chunk.get('content', '') for chunk in chunks])

        for pattern_def in relationship_patterns:
            for pattern in pattern_def['patterns']:
                matches = re.finditer(pattern, full_text)
                for match in matches:
                    source_text = match.group(1)
                    target_text = match.group(2)

                    # 查找对应的实体
                    source_entity = self._find_entity_by_text(entities, source_text, pattern_def['source_type'])
                    target_entity = self._find_entity_by_text(entities, target_text, pattern_def['target_type'])

                    if source_entity and target_entity:
                        relationship = {
                            'source': source_entity['text'],
                            'target': target_entity['text'],
                            'type': pattern_def['relation_type'],
                            'confidence': 0.8,
                            'source': 'rule_extraction',
                            'evidence': match.group(0)
                        }
                        relationships.append(relationship)

        return relationships

    def _extract_cooccurrence_relationships(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """基于共现抽取关系"""
        relationships = []

        # 统计实体在同一chunk中的共现次数
        cooccurrence_counts = {}
        entity_chunk_map = {}

        # 构建实体到chunk的映射
        for entity in entities:
            chunk_id = entity.get('chunk_id')
            if chunk_id:
                if chunk_id not in entity_chunk_map:
                    entity_chunk_map[chunk_id] = []
                entity_chunk_map[chunk_id].append(entity)

        # 统计共现
        for chunk_id, chunk_entities in entity_chunk_map.items():
            for i, entity1 in enumerate(chunk_entities):
                for entity2 in chunk_entities[i+1:]:
                    # 只考虑特定类型的组合
                    type_pair = (entity1['type'], entity2['type'])
                    if self._should_create_relationship(type_pair):
                        key = (entity1['text'], entity2['text'])
                        if key not in cooccurrence_counts:
                            cooccurrence_counts[key] = 0
                        cooccurrence_counts[key] += 1

        # 创建关系
        for (entity1_text, entity2_text), count in cooccurrence_counts.items():
            if count >= 1:  # 至少共现一次
                entity1 = self._find_entity_by_text(entities, entity1_text)
                entity2 = self._find_entity_by_text(entities, entity2_text)

                if entity1 and entity2:
                    relationship = {
                        'source': entity1_text,
                        'target': entity2_text,
                        'type': self._infer_relationship_type(entity1['type'], entity2['type']),
                        'confidence': min(0.3 + count * 0.2, 0.8),  # 基于共现次数计算置信度
                        'source': 'cooccurrence',
                        'cooccurrence_count': count
                    }
                    relationships.append(relationship)

        return relationships

    async def _extract_relationships_by_llm(
        self,
        entities: List[Dict[str, Any]],
        chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """使用LLM抽取关系"""
        relationships = []

        try:
            from app.services.llm_service import llm_service

            # 构建实体列表
            entity_list = '\n'.join([
                f"- {i+1}. {entity['text']} ({entity['type']})"
                for i, entity in enumerate(entities[:20])  # 限制实体数量
            ])

            # 构建上下文文本（选取包含实体的chunk）
            context_chunks = []
            for chunk in chunks:
                chunk_content = chunk.get('content', '')
                if any(entity['text'] in chunk_content for entity in entities[:20]):
                    context_chunks.append(chunk_content[:500])  # 限制长度

            context_text = '\n\n'.join(context_chunks[:5])  # 限制chunk数量

            prompt = f"""
请分析以下金融文档中的实体关系。

实体列表：
{entity_list}

文档内容：
{context_text}

请识别实体之间的关系，并以JSON格式返回：
[
    {{
        "source": "实体1名称",
        "target": "实体2名称",
        "type": "关系类型",
        "confidence": 0.8
    }}
]

关系类型包括：HAS_EXECUTIVE（拥有高管）、HAS_STOCK（拥有股票）、SUBSIDIARY_OF（是...的子公司）、PARTNER_WITH（与...合作）、INVESTS_IN（投资于）等。
只返回有高置信度的关系。
"""

            response = await llm_service.generate_response(prompt)

            # 解析LLM响应
            try:
                import json
                llm_relationships = json.loads(response)
                for rel in llm_relationships:
                    rel['source'] = 'llm_extraction'
                    relationships.append(rel)
            except json.JSONDecodeError:
                logger.warning("LLM返回的关系格式不正确")

        except Exception as e:
            logger.error(f"LLM关系抽取失败: {e}")

        return relationships

    def _find_entity_by_text(
        self,
        entities: List[Dict[str, Any]],
        text: str,
        expected_type: str = None
    ) -> Optional[Dict[str, Any]]:
        """根据文本查找实体"""
        for entity in entities:
            if entity['text'] == text or text in entity['text'] or entity['text'] in text:
                if expected_type is None or entity['type'] == expected_type:
                    return entity
        return None

    def _should_create_relationship(self, type_pair: Tuple[str, str]) -> bool:
        """判断是否应该创建关系"""
        # 定义允许创建关系的实体类型组合
        allowed_pairs = [
            ('COMPANY', 'PERSON'),
            ('PERSON', 'COMPANY'),
            ('COMPANY', 'STOCK'),
            ('STOCK', 'COMPANY'),
            ('COMPANY', 'COMPANY'),
            ('PERSON', 'PERSON')
        ]
        return type_pair in allowed_pairs or (type_pair[1], type_pair[0]) in allowed_pairs

    def _infer_relationship_type(
        self,
        type1: str,
        type2: str
    ) -> str:
        """推断关系类型"""
        type_mapping = {
            ('COMPANY', 'PERSON'): 'HAS_EXECUTIVE',
            ('PERSON', 'COMPANY'): 'WORKS_AT',
            ('COMPANY', 'STOCK'): 'HAS_STOCK',
            ('STOCK', 'COMPANY'): 'STOCK_OF',
            ('COMPANY', 'COMPANY'): 'RELATED_TO',
            ('PERSON', 'PERSON'): 'COLLEAGUE'
        }
        return type_mapping.get((type1, type2), 'RELATED_TO')


# 全局流水线实例
knowledge_base_pipeline = KnowledgeBasePipeline()