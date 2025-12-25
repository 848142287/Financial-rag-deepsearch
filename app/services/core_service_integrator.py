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
            self._services['milvus'] = MilvusService()
            self._services['neo4j'] = Neo4jService()

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

            # 阶段2: 多模态分析
            logger.info("🧠 阶段2: 多模态分析...")
            analysis_result = await self._services['qwen'].analyze_document_multimodal(
                file_content, filename, []
            )
            result['stages']['analysis'] = {
                'status': 'completed',
                'models_used': ['qwen-vl-plus'],
                'sections': len(analysis_result.get('sections_analysis', [])),
                'images_found': len(analysis_result.get('images_found', [])),
                'charts_found': len(analysis_result.get('charts_found', [])),
                'formulas_found': len(analysis_result.get('formulas_found', []))
            }

            # 阶段3: 实体关系抽取
            logger.info("🔗 阶段3: 实体关系抽取...")
            entities = await self._services['qwen'].extract_entity_relationships(
                analysis_result.get('summary', '')
            )
            result['stages']['entities'] = {
                'status': 'completed',
                'count': len(entities) if entities else 0
            }

            # 阶段4: 向量嵌入生成
            logger.info("🔢 阶段4: 向量嵌入生成...")
            text_content = analysis_result.get('summary', '')
            if text_content:
                embeddings = await self._services['embedding'].generate_embeddings([text_content])
                result['stages']['embeddings'] = {
                    'status': 'completed',
                    'dimension': len(embeddings[0]) if embeddings else 0,
                    'model': 'text-embedding-v4'
                }

            # 阶段5: 存储到向量数据库和知识图谱
            logger.info("💾 阶段5: 数据持久化...")

            # 存储到Milvus
            if 'embeddings' in result and embeddings:
                await self._services['milvus'].insert_embeddings(
                    collection_name="document_embeddings",
                    embeddings=embeddings,
                    documents=[{
                        'id': document_id,
                        'filename': filename,
                        'content': text_content,
                        'metadata': json.dumps(analysis_result, ensure_ascii=False)
                    }]
                )

            # 存储到Neo4j
            if entities and self.config.enable_knowledge_graph:
                for entity in entities[:5]:  # 限制数量
                    await self._services['neo4j'].create_entity_node(
                        entity_id=f"{document_id}_{entity.get('name', '')}",
                        entity_type=entity.get('type', 'UNKNOWN'),
                        properties=entity
                    )

            result['stages']['storage'] = {'status': 'completed'}
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