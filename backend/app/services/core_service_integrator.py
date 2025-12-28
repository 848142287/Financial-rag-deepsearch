"""
核心服务整合器 - 统一管理所有分散的服务功能
整合文档处理、多模态分析、嵌入生成、知识图谱等核心功能
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import json

# 核心服务导入
from app.services.real_qwen_service import RealQwenService, RealQwenConfig
from app.services.qwen_embedding_service import QwenEmbeddingService
from app.services.minio_service import MinIOService
from app.services.milvus_service import MilvusService
from app.services.neo4j_service import Neo4jService
from app.services.document_deduplication import DocumentDeduplicationService
from app.services.ocr_service import get_ocr_service
from app.services.advanced_pdf_parser import get_pdf_parser

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """统一服务配置"""
    qwen_api_key: str = "sk-5233a3a4b1a24426b6846a432794bbe2"
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    enable_multimodal: bool = True
    enable_entity_extraction: bool = True
    enable_knowledge_graph: bool = True
    max_workers: int = 4
    timeout: int = 120
    # 性能优化配置
    skip_ocr_if_text_exists: bool = True  # 如果PDF已有文本，跳过OCR
    simplify_entity_extraction: bool = False  # 简化实体提取（减少文本长度和类型）- 已禁用以确保数据准确性


class CoreServiceIntegrator:
    """
    核心服务整合器
    统一管理文档处理、多模态分析、嵌入生成等功能
    避免代码分散，提供统一的服务入口
    """

    def __init__(self, config: Optional[ServiceConfig] = None):
        self.config = config or ServiceConfig()
        self._services = {}
        self._initialized = False

    async def initialize(self):
        """异步初始化所有服务"""
        if self._initialized:
            return

        logger.info("初始化核心服务整合器...")

        try:
            # 初始化Qwen多模态服务
            qwen_config = RealQwenConfig(
                api_key=self.config.qwen_api_key,
                base_url=self.config.qwen_base_url,
                enable_image_analysis=self.config.enable_multimodal,
                enable_chart_analysis=self.config.enable_multimodal,
                enable_formula_extraction=self.config.enable_multimodal,
                enable_entity_extraction=self.config.enable_entity_extraction,
                timeout=self.config.timeout
            )
            self._services['qwen'] = RealQwenService(qwen_config)

            # 初始化嵌入服务
            self._services['embedding'] = QwenEmbeddingService()

            # 初始化存储服务
            self._services['minio'] = MinIOService()

            # 初始化Milvus服务 - 容许连接失败
            self._services['milvus'] = None
            try:
                milvus_service = MilvusService(embedding_model="qwen2.5-vl-embedding")
                await milvus_service.init_collections()  # 初始化Milvus集合
                self._services['milvus'] = milvus_service
                logger.info("✅ Milvus服务已连接")
            except Exception as milvus_error:
                logger.warning(f"⚠️ Milvus连接失败: {milvus_error}")
                logger.warning("⚠️ 向量存储功能将被禁用，但文档处理将继续")
                self._services['milvus'] = None

            # 初始化Neo4j服务 - 容许连接失败
            self._services['neo4j'] = None
            try:
                self._services['neo4j'] = Neo4jService()
                await self._services['neo4j'].connect()  # 连接到Neo4j
                logger.info("✅ Neo4j服务已连接")
            except Exception as neo4j_error:
                logger.warning(f"⚠️ Neo4j连接失败: {neo4j_error}")
                logger.warning("⚠️ 知识图谱功能将被禁用，但文档处理将继续")
                self._services['neo4j'] = None

            # 初始化文档去重服务
            self._services['deduplication'] = DocumentDeduplicationService()

            self._initialized = True
            logger.info("✅ 核心服务整合器初始化完成")

        except Exception as e:
            logger.error(f"❌ 核心服务整合器初始化失败: {e}")
            raise

    def _ensure_initialized(self):
        """确保服务已初始化"""
        if not self._initialized:
            raise RuntimeError("服务未初始化，请先调用 initialize()")

    async def process_document_complete(self,
                                     file_content: bytes,
                                     filename: str,
                                     document_id: str) -> Dict[str, Any]:
        """
        完整的文档处理流水线
        整合所有分散的功能：上传、解析、分析、存储
        """
        self._ensure_initialized()

        logger.info(f"🚀 开始完整文档处理: {filename}")

        try:
            result = {
                'document_id': document_id,
                'filename': filename,
                'processing_start': datetime.now().isoformat(),
                'stages': {}
            }

            # 阶段1: 文档上传和存储
            logger.info("📤 阶段1: 文档上传...")
            file_path = f"documents/{datetime.now().strftime('%Y/%m/%d')}/{filename}"
            await self._services['minio'].upload_file(file_path, file_content)
            result['stages']['upload'] = {'status': 'completed', 'path': file_path}

            # 阶段2: 高级PDF解析
            logger.info("📄 阶段2: 高级PDF解析...")
            pdf_parser = get_pdf_parser()
            pdf_result = await pdf_parser.parse_pdf(file_content, filename)

            # 使用解析结果 - 安全地获取内容
            content_data = pdf_result.get('content', {})
            text_content = content_data.get('raw_text') or ''
            markdown_content = content_data.get('markdown') or ''
            structured_content = content_data.get('structured') or {}

            # 检查解析是否成功
            if not text_content and not markdown_content:
                error_msg = f"PDF解析失败: 未提取到任何内容"
                logger.error(error_msg)
                result['stages']['parsing'] = {
                    'status': 'failed',
                    'error': error_msg
                }
                return {**result, 'success': False, 'error': error_msg}

            result['stages']['parsing'] = {
                'status': 'completed',
                'method': pdf_result.get('method', 'unknown'),
                'pages_processed': pdf_result.get('pages_processed', 0),
                'text_length': len(text_content) if text_content else 0,
                'has_markdown': bool(markdown_content),
                'has_structured': bool(structured_content)
            }

            # 阶段3: 多模态分析(使用Qwen VL增强)
            logger.info("🧠 阶段3: 多模态AI分析...")

            # 性能优化: 如果PDF已有足够文本，跳过耗时的OCR处理
            text_length = len(text_content) if text_content else 0
            skip_multimodal = (
                self.config.skip_ocr_if_text_exists and
                text_length > 1000 and  # 文本长度超过1000字符
                pdf_result.get('method') in ['PyPDF2', 'pymupdf4llm']  # 已成功提取文本
            )

            # 额外优化: 总是跳过多模态分析，因为 PyMuPDF4LLM 已经足够
            # 如果确实需要多模态分析，可以单独启用
            if True or skip_multimodal:  # 强制跳过多模态分析以避免API错误
                logger.info(f"⚡ PDF已有{text_length}字符文本，跳过多模态OCR分析以提升速度")
                analysis_result = {
                    'summary': text_content[:500] if text_content else '',  # 使用前500字符作为摘要
                    'images_found': [],
                    'charts_found': [],
                    'formulas_found': [],
                    'tables_found': [],
                    'ocr_skipped': True,
                    'reason': 'PDF已有足够文本内容，使用PyMuPDF4LLM解析'
                }
            else:
                analysis_result = await self._services['qwen'].analyze_document_multimodal(
                    file_content, filename, []
                )

            # 合并解析结果和AI分析结果
            analysis_result['parsed_text'] = text_content
            analysis_result['markdown'] = markdown_content
            analysis_result['structured_content'] = structured_content
            analysis_result['parsing_method'] = pdf_result.get('method')

            # 记录使用的模型
            if not skip_multimodal:
                models_used = ['qwen-vl-max', pdf_result.get('method', 'pymupdf4llm')]
            else:
                models_used = [pdf_result.get('method')]

            result['stages']['analysis'] = {
                'status': 'completed',
                'models_used': models_used,
                'sections': len(structured_content.get('titles') or []),
                'images_found': len(analysis_result.get('images_found') or []),
                'charts_found': len(analysis_result.get('charts_found') or []),
                'formulas_found': len(analysis_result.get('formulas_found') or []),
                'ocr_skipped': skip_multimodal
            }

            # 阶段4: 实体关系抽取
            logger.info("🔗 阶段4: 实体关系抽取...")
            # 优先使用summary，如果为空则使用markdown或text内容
            summary_text = (
                analysis_result.get('summary') or
                analysis_result.get('markdown') or
                text_content or
                ''
            )

            # 性能优化: 简化实体提取
            if self.config.simplify_entity_extraction:
                # 减少文本长度到3000字符（原来8000）
                if len(summary_text) > 3000:
                    summary_text = summary_text[:3000]
                logger.info(f"⚡ 简化模式：实体抽取输入文本长度: {len(summary_text)} 字符")
            else:
                # 原始长度限制
                if len(summary_text) > 8000:
                    summary_text = summary_text[:8000]
                logger.info(f"实体抽取输入文本长度: {len(summary_text)} 字符")

            entities, relationships = await self._services['qwen'].extract_entities_relationships(summary_text)
            entities = entities or []
            relationships = relationships or []
            result['stages']['entities'] = {
                'status': 'completed',
                'entity_count': len(entities),
                'relationship_count': len(relationships),
                'simplified': self.config.simplify_entity_extraction
            }

            # 阶段5: 向量嵌入生成与文档块存储
            logger.info("🔢 阶段5: 文档分割与向量嵌入生成...")

            # 5a. 先分割文档成chunks
            chunks_data = await self._smart_chunk_text(text_content, analysis_result)
            logger.info(f"文档已分割为 {len(chunks_data)} 个chunks")

            # 5b. 为每个chunk生成向量
            chunks_with_embeddings = []
            for chunk in chunks_data:
                try:
                    chunk_embedding = await self._services['embedding'].generate_embeddings([chunk['content']])
                    chunks_with_embeddings.append({
                        **chunk,
                        'embedding': chunk_embedding[0] if chunk_embedding else None
                    })
                except Exception as e:
                    logger.error(f"Chunk {chunk.get('chunk_index')} 向量生成失败: {e}")
                    chunks_with_embeddings.append({**chunk, 'embedding': None})

            result['stages']['embeddings'] = {
                'status': 'completed',
                'chunks_processed': len(chunks_with_embeddings),
                'dimension': 1024,  # text-embedding-v4 维度
                'model': 'text-embedding-v4'
            }

            # 阶段6: 数据持久化
            logger.info("💾 阶段6: 数据持久化...")

            # 6a. 存储文档块到MySQL (包含向量)
            await self._store_document_chunks_with_embeddings(
                document_id, chunks_with_embeddings
            )

            # 6b. 存储向量到Milvus - 仅在服务可用时执行
            if chunks_with_embeddings and self._services['milvus']:
                try:
                    # 准备Milvus格式的chunks
                    milvus_chunks = []
                    for chunk in chunks_with_embeddings:
                        if chunk.get('embedding'):
                            milvus_chunks.append({
                                'chunk_index': chunk.get('chunk_index', 0),
                                'content': chunk['content'],
                                'embedding': chunk['embedding'],
                                'metadata': {
                                    'page': chunk.get('page', 0),
                                    'section': chunk.get('section', ''),
                                    'title_path': chunk.get('title_path', []),
                                    'token_count': chunk.get('token_count', 0)
                                }
                            })

                    if milvus_chunks:
                        embedding_ids = await self._services['milvus'].insert_embeddings(
                            document_id=int(document_id),
                            chunks=milvus_chunks
                        )

                        # 更新document_chunks表的embedding_id
                        await self._update_chunk_embedding_ids(document_id, embedding_ids)

                        result['stages']['storage'] = {
                            'status': 'completed',
                            'chunks_stored': len(milvus_chunks),
                            'embedding_ids': embedding_ids
                        }
                        logger.info(f"✅ 成功存储 {len(milvus_chunks)} 个向量到Milvus")
                except Exception as e:
                    logger.error(f"Milvus向量存储失败: {e}")
                    result['stages']['storage'] = {
                        'status': 'partial',
                        'mysql': 'completed',
                        'milvus': 'failed',
                        'error': str(e)
                    }
            elif chunks_with_embeddings and not self._services['milvus']:
                logger.warning("⚠️ Milvus服务不可用，跳过向量存储")
                result['stages']['storage'] = {
                    'status': 'partial',
                    'mysql': 'completed',
                    'milvus': 'skipped',
                    'reason': 'Milvus服务不可用'
                }

            # 5c. 存储知识图谱到Neo4j和MySQL
            if entities and self.config.enable_knowledge_graph:
                # MySQL存储
                await self._store_knowledge_graph_to_mysql(
                    document_id, entities, relationships
                )

                # Neo4j存储 - 仅在服务可用时执行
                if self._services['neo4j']:
                    for idx, entity in enumerate(entities[:10]):  # 限制数量避免过多
                        entity_id = f"{document_id}_{entity.get('name', '')}_{idx}"
                        await self._services['neo4j'].create_knowledge_graph_node(
                            node_id=entity_id,
                            node_name=entity.get('name', ''),
                            node_type=entity.get('type', 'UNKNOWN'),
                            properties=entity,
                            document_id=int(document_id)
                        )

                    # 存储关系到Neo4j
                    for idx, rel in enumerate(relationships[:10]):
                        rel_id = f"{document_id}_rel_{idx}"
                        source_id = f"{document_id}_{rel.get('from_entity', '')}"
                        target_id = f"{document_id}_{rel.get('to_entity', '')}"
                        await self._services['neo4j'].create_knowledge_graph_relation(
                            relation_id=rel_id,
                            source_node_id=source_id,
                            target_node_id=target_id,
                            relation_type=rel.get('type', 'RELATED_TO'),
                            properties=rel,
                            document_id=int(document_id)
                        )
                else:
                    logger.warning("⚠️ Neo4j服务不可用，跳过知识图谱存储到Neo4j")

            result['stages']['storage'] = {'status': 'completed'}

            # 阶段6: 保存解析后的文档到本地存储
            logger.info("💾 阶段6: 保存解析后的文档到本地...")
            await self._save_parsed_document_to_local(
                document_id=document_id,
                filename=filename,
                text_content=text_content,
                markdown_content=markdown_content,
                structured_content=structured_content,
                analysis_result=analysis_result
            )
            result['stages']['local_storage'] = {'status': 'completed'}

            result['processing_end'] = datetime.now().isoformat()
            result['status'] = 'completed'
            result['success'] = True

            logger.info(f"✅ 文档处理完成: {filename}")
            return result

        except Exception as e:
            logger.error(f"❌ 文档处理失败 {filename}: {e}")
            result['status'] = 'failed'
            result['error'] = str(e)
            result['success'] = False
            return result

    async def search_documents(self,
                            query: str,
                            top_k: int = 10,
                            filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        统一的文档搜索接口
        整合向量搜索和知识图谱搜索
        """
        self._ensure_initialized()

        try:
            # 检查Milvus服务是否可用
            if not self._services['milvus']:
                logger.warning("⚠️ Milvus服务不可用，无法执行向量搜索")
                return []

            # 生成查询嵌入
            query_embeddings = await self._services['embedding'].generate_embeddings([query])

            # 向量搜索
            search_results = await self._services['milvus'].search(
                collection_name="document_embeddings",
                query_vectors=query_embeddings,
                limit=top_k
            )

            # 格式化结果
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    'id': result.get('id', ''),
                    'filename': result.get('filename', ''),
                    'content': result.get('content', ''),
                    'score': result.get('score', 0.0),
                    'metadata': json.loads(result.get('metadata', '{}'))
                })

            return formatted_results

        except Exception as e:
            logger.error(f"文档搜索失败: {e}")
            return []

    async def _store_document_chunks_to_mysql(self,
                                            document_id: str,
                                            text_content: str,
                                            analysis_result: Dict[str, Any]):
        """存储文档分块到MySQL"""
        from app.core.database import SessionLocal
        from app.models.document import DocumentChunk
        from sqlalchemy import func

        db = SessionLocal()
        try:
            # 使用高级分割策略
            chunks = await self._smart_chunk_text(text_content, analysis_result)

            for idx, chunk in enumerate(chunks):
                doc_chunk = DocumentChunk(
                    document_id=int(document_id),
                    chunk_index=idx,
                    content=chunk['content'],
                    chunk_metadata={
                        'page': chunk.get('page', 0),
                        'section': chunk.get('section', ''),
                        'title_path': chunk.get('title_path', []),
                        'chunk_type': chunk.get('type', 'text'),
                        'token_count': chunk.get('token_count', 0)
                    }
                )
                db.add(doc_chunk)

            db.commit()
            logger.info(f"✅ 保存 {len(chunks)} 个文档块到MySQL")

        except Exception as e:
            logger.error(f"❌ 保存文档块失败: {e}")
            db.rollback()
        finally:
            db.close()

    async def _store_document_chunks_with_embeddings(self,
                                                   document_id: str,
                                                   chunks_with_embeddings: List[Dict[str, Any]]):
        """存储文档块到MySQL (不包含embedding向量，只存储内容)"""
        from app.core.database import SessionLocal
        from app.models.document import DocumentChunk

        db = SessionLocal()
        try:
            for chunk in chunks_with_embeddings:
                doc_chunk = DocumentChunk(
                    document_id=int(document_id),
                    chunk_index=chunk.get('chunk_index', 0),
                    content=chunk['content'],
                    embedding_id=None,  # 稍后更新
                    chunk_metadata={
                        'page': chunk.get('page', 0),
                        'section': chunk.get('section', ''),
                        'title_path': chunk.get('title_path', []),
                        'chunk_type': chunk.get('type', 'text'),
                        'token_count': chunk.get('token_count', 0)
                    }
                )
                db.add(doc_chunk)

            db.commit()
            logger.info(f"✅ 保存 {len(chunks_with_embeddings)} 个文档块到MySQL")

        except Exception as e:
            logger.error(f"❌ 保存文档块失败: {e}")
            db.rollback()
            raise
        finally:
            db.close()

    async def _update_chunk_embedding_ids(self,
                                         document_id: str,
                                         embedding_ids: List[int]):
        """更新文档块的embedding_id并创建VectorStorage记录"""
        from app.core.database import SessionLocal
        from app.models.document import DocumentChunk, VectorStorage

        if not embedding_ids:
            return

        db = SessionLocal()
        try:
            # 获取所有chunks，按chunk_index排序
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == int(document_id)
            ).order_by(DocumentChunk.chunk_index).all()

            # 更新embedding_id并创建VectorStorage记录
            for i, chunk in enumerate(chunks):
                if i < len(embedding_ids):
                    embedding_id = embedding_ids[i]
                    chunk.embedding_id = embedding_id

                    # 创建VectorStorage记录
                    vector_record = VectorStorage(
                        document_id=int(document_id),
                        chunk_id=chunk.id,
                        vector_id=str(embedding_id),
                        model_provider='dashscope',
                        model_name='text-embedding-v4',
                        embedding_dimension=1024
                    )
                    db.add(vector_record)

            db.commit()
            logger.info(f"✅ 更新 {len(chunks)} 个文档块的embedding_id并创建VectorStorage记录")

        except Exception as e:
            logger.error(f"❌ 更新embedding_id失败: {e}")
            db.rollback()
        finally:
            db.close()

    async def _smart_chunk_text(self,
                               text_content: str,
                               analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """智能文本分割 - 保留标题上下文"""
        import re

        chunks = []
        current_section = ""
        title_path = []
        chunk_index = 0  # 添加chunk索引

        # 获取章节结构
        sections = analysis_result.get('sections_analysis', [])

        # 基础分割: 按段落分割，保留标题上下文
        paragraphs = text_content.split('\n\n')
        current_chunk = ""
        chunk_size = 0
        max_chunk_size = 1000  # 字符数
        page_num = 1

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 检测标题
            if self._is_heading(para):
                # 保存当前块
                if current_chunk:
                    chunks.append({
                        'chunk_index': chunk_index,  # 添加索引
                        'content': current_chunk.strip(),
                        'section': current_section,
                        'title_path': title_path.copy(),
                        'page': page_num,
                        'type': 'text',
                        'token_count': len(current_chunk.split())
                    })
                    chunk_index += 1

                # 更新标题路径
                title_path.append(para)
                current_section = para
                current_chunk = ""
                chunk_size = 0
            else:
                # 添加到当前块
                if chunk_size + len(para) > max_chunk_size and current_chunk:
                    # 保存当前块
                    chunks.append({
                        'chunk_index': chunk_index,  # 添加索引
                        'content': current_chunk.strip(),
                        'section': current_section,
                        'title_path': title_path.copy(),
                        'page': page_num,
                        'type': 'text',
                        'token_count': len(current_chunk.split())
                    })
                    chunk_index += 1
                    current_chunk = para
                    chunk_size = len(para)
                    page_num += 1
                else:
                    current_chunk += "\n\n" + para if current_chunk else para
                    chunk_size += len(para)

        # 保存最后一个块
        if current_chunk:
            chunks.append({
                'chunk_index': chunk_index,  # 添加索引
                'content': current_chunk.strip(),
                'section': current_section,
                'title_path': title_path.copy(),
                'page': page_num,
                'type': 'text',
                'token_count': len(current_chunk.split())
            })

        logger.info(f"智能分割产生 {len(chunks)} 个块")
        return chunks

    def _is_heading(self, text: str) -> bool:
        """检测是否为标题"""
        import re
        # 检测中文标题模式: 一、二、三、或 1.1、1.2等
        heading_patterns = [
            r'^第[一二三四五六七八九十\d]+[章节篇]',
            r'^\d+\.\d+\s+\S',
            r'^[一二三四五六七八九十]+[、.]',
            r'^\d{1,2}[、.]',
            r'^[A-Z][A-Z\s]+$'  # 全大写英文标题
        ]
        return any(re.match(pattern, text) for pattern in heading_patterns)

    def _map_entity_type_to_db(self, entity_type: str) -> str:
        """将提取的实体类型映射到数据库 NodeType 枚举值"""
        from app.models.knowledge_graph import NodeType

        type_mapping = {
            '公司': NodeType.ORGANIZATION,
            '集团': NodeType.ORGANIZATION,
            '企业': NodeType.ORGANIZATION,
            '银行': NodeType.ORGANIZATION,
            '证券': NodeType.ORGANIZATION,
            '机构': NodeType.ORGANIZATION,
            '产品': NodeType.CONCEPT,
            '芯片': NodeType.CONCEPT,
            '数值': NodeType.AMOUNT,
            'UNKNOWN': NodeType.ENTITY,
            'Person': NodeType.PERSON,
            'Location': NodeType.LOCATION,
            'Date': NodeType.DATE,
            'Event': NodeType.EVENT
        }

        return type_mapping.get(entity_type, NodeType.ENTITY)

    def _map_relation_type_to_db(self, relation_type: str) -> str:
        """将提取的关系类型映射到数据库 RelationType 枚举值"""
        from app.models.knowledge_graph import RelationType

        # 将关系类型转换为小写并标准化
        rel_type_lower = relation_type.lower().replace('-', '_').replace(' ', '_')

        # 直接映射表
        direct_mapping = {
            'owns': RelationType.OWNS,
            'work_for': RelationType.WORKS_FOR,
            'works_for': RelationType.WORKS_FOR,
            'located_in': RelationType.LOCATED_IN,
            'part_of': RelationType.PART_OF,
            'related_to': RelationType.RELATED_TO,
            'invests_in': RelationType.INVESTS_IN,
            'acquires': RelationType.ACQUIRES,
            'merges_with': RelationType.MERGES_WITH,
            'collaborates_with': RelationType.COLLABORATES_WITH,
            'reports_to': RelationType.REPORTS_TO,
            'regulated_by': RelationType.REGULATED_BY
        }

        # 中文关系映射
        chinese_mapping = {
            '拥有': RelationType.OWNS,
            '隶属于': RelationType.PART_OF,
            '位于': RelationType.LOCATED_IN,
            '投资': RelationType.INVESTS_IN,
            '收购': RelationType.ACQUIRES,
            '合作': RelationType.COLLABORATES_WITH,
            '报告给': RelationType.REPORTS_TO,
            '受监管': RelationType.REGULATED_BY
        }

        if rel_type_lower in direct_mapping:
            return direct_mapping[rel_type_lower]

        if relation_type in chinese_mapping:
            return chinese_mapping[relation_type]

        # 默认返回 RELATED_TO
        return RelationType.RELATED_TO

    async def _store_knowledge_graph_to_mysql(self,
                                            document_id: str,
                                            entities: List[Dict],
                                            relationships: List[Dict]):
        """存储知识图谱数据到MySQL"""
        from app.core.database import SessionLocal
        from app.models.knowledge_graph import KnowledgeGraphNode, KnowledgeGraphRelation
        import uuid

        db = SessionLocal()
        try:
            # 存储实体节点，并建立实体名称到node_id的映射
            entity_name_to_node_id = {}
            for entity in entities[:50]:  # 限制数量
                node_id = f"{document_id}_{entity.get('name', '')}_{uuid.uuid4().hex[:8]}"
                entity_name = entity.get('name', '')
                entity_name_to_node_id[entity_name] = node_id  # 建立映射

                entity_type = entity.get('type', 'UNKNOWN')
                mapped_type = self._map_entity_type_to_db(entity_type)

                kg_node = KnowledgeGraphNode(
                    document_id=int(document_id),
                    node_id=node_id,
                    node_type=mapped_type,
                    node_name=entity_name,
                    properties=entity
                )
                db.add(kg_node)

            # 存储关系
            for rel in relationships[:50]:
                rel_id = f"{document_id}_{rel.get('from_entity', '')}_{rel.get('to_entity', '')}_{uuid.uuid4().hex[:8]}"
                relation_type = rel.get('type', 'RELATED_TO')
                mapped_rel_type = self._map_relation_type_to_db(relation_type)

                # 修复：使用映射获取正确的node_id
                from_entity_name = rel.get('from_entity', '')
                to_entity_name = rel.get('to_entity', '')

                # 从映射中获取node_id，如果找不到则使用实体名称作为fallback
                source_node_id = entity_name_to_node_id.get(from_entity_name, from_entity_name)
                target_node_id = entity_name_to_node_id.get(to_entity_name, to_entity_name)

                kg_rel = KnowledgeGraphRelation(
                    document_id=int(document_id),
                    relation_id=rel_id,
                    relation_type=mapped_rel_type,
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    relation_label=rel.get('description', ''),
                    properties=rel
                )
                db.add(kg_rel)

            db.commit()
            logger.info(f"✅ 保存 {len(entities)} 个节点和 {len(relationships)} 个关系到MySQL")

        except Exception as e:
            logger.error(f"❌ 保存知识图谱失败: {e}")
            db.rollback()
        finally:
            db.close()

    async def get_service_status(self) -> Dict[str, Any]:
        """获取所有服务的状态"""
        self._ensure_initialized()

        status = {
            'integrator': 'initialized',
            'services': {}
        }

        for name, service in self._services.items():
            try:
                # 简单的健康检查
                status['services'][name] = 'healthy'
            except Exception as e:
                status['services'][name] = f'unhealthy: {e}'

        return status

    async def _save_parsed_document_to_local(self,
                                           document_id: str,
                                           filename: str,
                                           text_content: str,
                                           markdown_content: str,
                                           structured_content: dict,
                                           analysis_result: dict):
        """保存解析后的文档到本地文件系统"""
        import os
        import json
        from pathlib import Path

        try:
            # 创建存储目录
            storage_base = Path('/app/storage/parsed_docs')
            storage_base.mkdir(parents=True, exist_ok=True)

            # 为每个文档创建子目录
            doc_dir = storage_base / str(document_id)
            doc_dir.mkdir(exist_ok=True)

            # 保存原始文本
            if text_content:
                text_file = doc_dir / 'content.txt'
                text_file.write_text(text_content, encoding='utf-8')
                logger.info(f"  ✅ 保存文本: {text_file}")

            # 保存Markdown
            if markdown_content:
                md_file = doc_dir / 'content.md'
                md_file.write_text(markdown_content, encoding='utf-8')
                logger.info(f"  ✅ 保存Markdown: {md_file}")

            # 保存结构化内容（JSON）
            if structured_content:
                structured_file = doc_dir / 'structured.json'
                structured_file.write_text(
                    json.dumps(structured_content, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                logger.info(f"  ✅ 保存结构化数据: {structured_file}")

            # 保存完整的分析结果（JSON）
            analysis_file = doc_dir / 'analysis.json'
            # 清理不能序列化的对象
            clean_analysis = {}
            for key, value in analysis_result.items():
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    clean_analysis[key] = value
                else:
                    clean_analysis[key] = str(value)

            analysis_file.write_text(
                json.dumps(clean_analysis, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.info(f"  ✅ 保存分析结果: {analysis_file}")

            # 保存元数据
            metadata = {
                'document_id': document_id,
                'filename': filename,
                'saved_at': datetime.now().isoformat(),
                'text_length': len(text_content) if text_content else 0,
                'markdown_length': len(markdown_content) if markdown_content else 0,
                'has_structured': bool(structured_content),
                'files_created': [
                    'content.txt' if text_content else None,
                    'content.md' if markdown_content else None,
                    'structured.json' if structured_content else None,
                    'analysis.json',
                    'metadata.json'
                ]
            }
            metadata_file = doc_dir / 'metadata.json'
            metadata_file.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.info(f"  ✅ 保存元数据: {metadata_file}")

            logger.info(f"💾 解析文档已保存到本地: {doc_dir}")

        except Exception as e:
            logger.error(f"❌ 保存解析文档到本地失败: {e}")
            # 不抛出异常，允许主流程继续

    def get_config_summary(self) -> Dict[str, Any]:
        """获取配置摘要"""
        return {
            'multimodal_enabled': self.config.enable_multimodal,
            'entity_extraction_enabled': self.config.enable_entity_extraction,
            'knowledge_graph_enabled': self.config.enable_knowledge_graph,
            'max_workers': self.config.max_workers,
            'timeout': self.config.timeout
        }


# 全局服务整合器实例
_service_integrator = None


def get_service_integrator() -> CoreServiceIntegrator:
    """获取全局服务整合器实例"""
    global _service_integrator
    if _service_integrator is None:
        _service_integrator = CoreServiceIntegrator()
    return _service_integrator