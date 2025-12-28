"""
增强的本地文件存储服务
实现完整的文档解析结果保存，包含内容去重、元数据补充、增强功能集成
"""
import os
import json
import hashlib
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentQuality(str, Enum):
    """文档质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class StorageConfig:
    """存储配置"""
    base_path: str = "/app/storage/parsed_docs"
    enable_deduplication: bool = True
    enable_compression: bool = False
    enable_versioning: bool = False
    max_versions: int = 5
    generate_index: bool = True


@dataclass
class DocumentMetadata:
    """完整文档元数据"""
    # 基础信息
    document_id: str
    filename: str
    file_path: str

    # 文件信息
    file_size: int = 0
    file_extension: str = ""
    mime_type: str = ""

    # 文档统计
    page_count: int = 0
    chunk_count: int = 0
    total_characters: int = 0
    total_tokens: int = 0

    # 提取信息
    extraction_method: str = ""
    extraction_time: float = 0.0
    processing_time: float = 0.0

    # 内容统计
    has_tables: bool = False
    table_count: int = 0
    has_images: bool = False
    image_count: int = 0
    has_formulas: bool = False
    formula_count: int = 0

    # 质量指标
    status: str = ""
    quality_score: float = 0.0
    quality_level: str = ""

    # 语言和编码
    language: str = "zh-CN"
    encoding: str = "utf-8"

    # 时间戳
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # 增强功能标记
    has_semantic_enhancement: bool = False
    has_knowledge_graph: bool = False
    has_embeddings: bool = False

    # 来源信息
    source: str = "document_parser"
    parser_version: str = "1.0.0"


class EnhancedLocalStorage:
    """
    增强的本地文件存储服务

    功能：
    1. 内容去重
    2. 完整元数据生成
    3. 独立Markdown文件保存
    4. 增强功能结果保存
    5. 文档索引生成
    6. 质量评估
    """

    def __init__(self, config: Optional[StorageConfig] = None):
        """初始化存储服务"""
        self.config = config or StorageConfig()
        self.base_path = Path(self.config.base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"增强本地存储服务初始化完成，路径: {self.base_path}")

    async def save_document(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        content: str,
        markdown_content: Optional[str] = None,
        chunks: Optional[List[Dict]] = None,
        embeddings: Optional[List[List[float]]] = None,
        semantic_enhancement: Optional[Dict] = None,
        knowledge_graph: Optional[Dict] = None,
        analysis_result: Optional[Dict] = None,
        extraction_method: str = "unknown",
        extraction_time: float = 0.0
    ) -> Dict[str, Any]:
        """
        保存文档的完整信息

        Args:
            document_id: 文档ID
            filename: 文件名
            file_path: 文件路径
            content: 原始文本内容
            markdown_content: Markdown格式内容（可选）
            chunks: 文档分块（可选）
            embeddings: 向量嵌入（可选）
            semantic_enhancement: 语义增强结果（可选）
            knowledge_graph: 知识图谱（可选）
            analysis_result: 完整分析结果（可选）
            extraction_method: 提取方法
            extraction_time: 提取耗时

        Returns:
            保存结果
        """
        try:
            start_time = datetime.now()

            # 创建文档目录
            doc_dir = self.base_path / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # 1. 生成完整元数据
            metadata = self._generate_metadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                content=content,
                chunks=chunks,
                extraction_method=extraction_method,
                extraction_time=extraction_time,
                has_semantic=bool(semantic_enhancement),
                has_kg=bool(knowledge_graph),
                has_embeddings=bool(embeddings)
            )

            # 2. 内容去重
            deduplicated_content = self._deduplicate_content(content) if self.config.enable_deduplication else content
            deduplication_stats = self._get_deduplication_stats(content, deduplicated_content)

            # 3. 保存原始文本
            text_file = doc_dir / 'content.txt'
            text_file.write_text(deduplicated_content, encoding='utf-8')
            logger.info(f"✅ 保存文本: {text_file}")

            # 4. 保存Markdown
            md_content = markdown_content or self._convert_to_markdown(content, metadata)
            md_file = doc_dir / 'content.md'
            md_file.write_text(md_content, encoding='utf-8')
            logger.info(f"✅ 保存Markdown: {md_file}")

            # 5. 保存元数据
            metadata_file = doc_dir / 'metadata.json'
            metadata_file.write_text(
                json.dumps(asdict(metadata), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.info(f"✅ 保存元数据: {metadata_file}")

            # 6. 保存Chunks
            if chunks:
                chunks_file = doc_dir / 'chunks.json'
                chunks_file.write_text(
                    json.dumps(chunks, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                logger.info(f"✅ 保存Chunks: {chunks_file}")

            # 7. 保存Embeddings
            if embeddings:
                embeddings_file = doc_dir / 'embeddings.json'
                embeddings_data = {
                    'count': len(embeddings),
                    'dimension': len(embeddings[0]) if embeddings else 0,
                    'embeddings': embeddings
                }
                embeddings_file.write_text(
                    json.dumps(embeddings_data, ensure_ascii=False),
                    encoding='utf-8'
                )
                logger.info(f"✅ 保存Embeddings: {embeddings_file}")

            # 8. 保存语义增强结果
            if semantic_enhancement:
                semantic_file = doc_dir / 'semantic_enhancement.json'
                semantic_file.write_text(
                    json.dumps(semantic_enhancement, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                logger.info(f"✅ 保存语义增强: {semantic_file}")

            # 9. 保存知识图谱
            if knowledge_graph:
                kg_file = doc_dir / 'knowledge_graph.json'
                kg_file.write_text(
                    json.dumps(knowledge_graph, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                logger.info(f"✅ 保存知识图谱: {kg_file}")

            # 10. 保存完整分析结果
            if analysis_result:
                analysis_file = doc_dir / 'analysis.json'
                # 清理不能序列化的对象
                clean_analysis = self._clean_analysis_result(analysis_result)
                analysis_file.write_text(
                    json.dumps(clean_analysis, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )
                logger.info(f"✅ 保存分析结果: {analysis_file}")

            # 11. 生成质量报告
            quality_report = self._assess_quality(metadata, deduplication_stats)
            quality_file = doc_dir / 'quality_report.json'
            quality_file.write_text(
                json.dumps(quality_report, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            logger.info(f"✅ 保存质量报告: {quality_file}")

            # 12. 生成处理日志
            processing_log = {
                'document_id': document_id,
                'filename': filename,
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration': (datetime.now() - start_time).total_seconds(),
                'stages': {
                    'metadata': 'completed',
                    'text': 'completed',
                    'markdown': 'completed',
                    'chunks': 'completed' if chunks else 'skipped',
                    'embeddings': 'completed' if embeddings else 'skipped',
                    'semantic': 'completed' if semantic_enhancement else 'skipped',
                    'knowledge_graph': 'completed' if knowledge_graph else 'skipped',
                    'analysis': 'completed' if analysis_result else 'skipped',
                    'quality': 'completed'
                },
                'deduplication': deduplication_stats
            }
            log_file = doc_dir / 'processing_log.json'
            log_file.write_text(
                json.dumps(processing_log, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            # 13. 更新全局索引
            if self.config.generate_index:
                await self._update_index(document_id, metadata)

            logger.info(f"✅ 文档保存完成: {document_id}")

            return {
                'success': True,
                'document_id': document_id,
                'path': str(doc_dir),
                'files': list(doc_dir.glob('*')),
                'metadata': asdict(metadata),
                'quality': quality_report,
                'deduplication': deduplication_stats
            }

        except Exception as e:
            logger.error(f"❌ 文档保存失败 {document_id}: {e}")
            return {
                'success': False,
                'document_id': document_id,
                'error': str(e)
            }

    def _generate_metadata(
        self,
        document_id: str,
        filename: str,
        file_path: str,
        content: str,
        chunks: Optional[List[Dict]],
        extraction_method: str,
        extraction_time: float,
        has_semantic: bool,
        has_kg: bool,
        has_embeddings: bool
    ) -> DocumentMetadata:
        """生成完整文档元数据"""
        try:
            file_path_obj = Path(file_path)
            file_size = file_path_obj.stat().st_size if file_path_obj.exists() else 0

            # 统计内容
            total_chars = len(content)
            total_tokens = len(content.split())  # 简单估算

            # 检测表格
            has_tables = '| 表格' in content or 'Table' in content
            table_count = content.count('|') // 10 if has_tables else 0  # 粗略估计

            # 检测图片
            has_images = '![图片' in content or '![' in content
            image_count = content.count('![') if has_images else 0

            # 检测公式
            has_formulas = '$$' in content or '$' in content
            formula_count = content.count('$$') if has_formulas else 0

            # 计算质量分数
            quality_score = self._calculate_quality_score(
                total_chars=total_chars,
                chunk_count=len(chunks) if chunks else 0,
                has_tables=has_tables,
                has_images=has_images,
                extraction_time=extraction_time
            )

            # 确定质量等级
            if quality_score >= 0.8:
                quality_level = DocumentQuality.HIGH.value
            elif quality_score >= 0.5:
                quality_level = DocumentQuality.MEDIUM.value
            else:
                quality_level = DocumentQuality.LOW.value

            return DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                file_size=file_size,
                file_extension=file_path_obj.suffix,
                mime_type=self._get_mime_type(file_path_obj.suffix),
                page_count=content.count('\f') + 1,  # PDF分页符
                chunk_count=len(chunks) if chunks else 0,
                total_characters=total_chars,
                total_tokens=total_tokens,
                extraction_method=extraction_method,
                extraction_time=extraction_time,
                has_tables=has_tables,
                table_count=table_count,
                has_images=has_images,
                image_count=image_count,
                has_formulas=has_formulas,
                formula_count=formula_count,
                status="success",
                quality_score=quality_score,
                quality_level=quality_level,
                has_semantic_enhancement=has_semantic,
                has_knowledge_graph=has_kg,
                has_embeddings=has_embeddings
            )

        except Exception as e:
            logger.error(f"生成元数据失败: {e}")
            return DocumentMetadata(
                document_id=document_id,
                filename=filename,
                file_path=file_path,
                status="error"
            )

    def _deduplicate_content(self, content: str) -> str:
        """去除重复内容"""
        try:
            lines = content.split('\n')
            seen = set()
            unique_lines = []

            for line in lines:
                # 计算行哈希
                line_hash = hashlib.md5(line.strip().encode('utf-8')).hexdigest()

                # 跳过已见过的行（但保留空行和短标题）
                if line_hash not in seen or len(line.strip()) < 10:
                    seen.add(line_hash)
                    unique_lines.append(line)

            deduplicated = '\n'.join(unique_lines)

            logger.info(f"内容去重: {len(lines)} → {len(unique_lines)} 行")

            return deduplicated

        except Exception as e:
            logger.error(f"内容去重失败: {e}")
            return content

    def _get_deduplication_stats(self, original: str, deduplicated: str) -> Dict:
        """获取去重统计信息"""
        original_lines = len(original.split('\n'))
        deduplicated_lines = len(deduplicated.split('\n'))
        removed_lines = original_lines - deduplicated_lines
        removal_rate = removed_lines / original_lines if original_lines > 0 else 0

        original_size = len(original.encode('utf-8'))
        deduplicated_size = len(deduplicated.encode('utf-8'))
        saved_bytes = original_size - deduplicated_size

        return {
            'original_lines': original_lines,
            'deduplicated_lines': deduplicated_lines,
            'removed_lines': removed_lines,
            'removal_rate': round(removal_rate * 100, 2),
            'original_size_bytes': original_size,
            'deduplicated_size_bytes': deduplicated_size,
            'saved_bytes': saved_bytes,
            'compression_ratio': round(saved_bytes / original_size * 100, 2) if original_size > 0 else 0
        }

    def _convert_to_markdown(self, content: str, metadata: DocumentMetadata) -> str:
        """将内容转换为Markdown格式"""
        # 如果内容已经包含Markdown标记，直接返回
        if any(marker in content for marker in ['# ', '**', '*', '```', '| ', '![']):
            return content

        # 否则添加基本Markdown格式
        lines = content.split('\n\n')
        markdown_lines = []

        for i, section in enumerate(lines):
            section = section.strip()
            if not section:
                continue

            # 如果是标题（短且在开头）
            if i == 0 and len(section) < 100:
                markdown_lines.append(f"# {section}\n")
            else:
                markdown_lines.append(f"{section}\n")

        return '\n'.join(markdown_lines)

    def _calculate_quality_score(
        self,
        total_chars: int,
        chunk_count: int,
        has_tables: bool,
        has_images: bool,
        extraction_time: float
    ) -> float:
        """计算文档质量分数"""
        score = 0.0

        # 内容长度评分 (0-0.3)
        if total_chars > 10000:
            score += 0.3
        elif total_chars > 5000:
            score += 0.2
        elif total_chars > 1000:
            score += 0.1

        # 分块评分 (0-0.2)
        if chunk_count > 20:
            score += 0.2
        elif chunk_count > 10:
            score += 0.15
        elif chunk_count > 0:
            score += 0.1

        # 多媒体内容 (0-0.2)
        if has_tables:
            score += 0.1
        if has_images:
            score += 0.1

        # 处理速度 (0-0.2)
        if extraction_time < 30:
            score += 0.2
        elif extraction_time < 60:
            score += 0.15
        elif extraction_time < 120:
            score += 0.1

        # 基础分 (0-0.1)
        score += 0.1

        return min(1.0, score)

    def _assess_quality(
        self,
        metadata: DocumentMetadata,
        deduplication_stats: Dict
    ) -> Dict:
        """评估文档质量"""
        scores = {
            'content_length': 0.2 if metadata.total_characters > 1000 else 0,
            'has_chunks': 0.2 if metadata.chunk_count > 0 else 0,
            'processing_speed': 0.2 if metadata.extraction_time < 60 else 0,
            'success_status': 0.2 if metadata.status == 'success' else 0,
            'has_metadata': 0.2 if len(asdict(metadata)) > 10 else 0
        }

        total_score = sum(scores.values())

        return {
            'overall_quality': metadata.quality_level,
            'quality_score': round(metadata.quality_score, 2),
            'calculated_score': round(total_score, 2),
            'dimension_scores': scores,
            'deduplication_benefit': deduplication_stats.get('compression_ratio', 0),
            'recommendations': self._generate_recommendations(metadata, total_score)
        }

    def _generate_recommendations(self, metadata: DocumentMetadata, score: float) -> List[str]:
        """生成改进建议"""
        recommendations = []

        if metadata.chunk_count == 0:
            recommendations.append("建议进行文档分块以提高检索效果")

        if not metadata.has_tables and metadata.total_characters > 5000:
            recommendations.append("建议提取表格信息")

        if metadata.extraction_time > 60:
            recommendations.append("考虑优化提取流程以提高性能")

        if score < 0.5:
            recommendations.append("文档质量较低，建议重新处理")

        if not recommendations:
            recommendations.append("文档质量良好，无需改进")

        return recommendations

    def _get_mime_type(self, extension: str) -> str:
        """获取MIME类型"""
        mime_types = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.html': 'text/html'
        }
        return mime_types.get(extension.lower(), 'application/octet-stream')

    def _clean_analysis_result(self, analysis_result: Dict) -> Dict:
        """清理分析结果，移除不可序列化的对象"""
        clean = {}

        for key, value in analysis_result.items():
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                clean[key] = value
            elif hasattr(value, '__dict__'):
                # 处理dataclass对象
                clean[key] = self._clean_analysis_result(value.__dict__)
            else:
                clean[key] = str(value)

        return clean

    async def _update_index(self, document_id: str, metadata: DocumentMetadata):
        """更新全局文档索引"""
        try:
            index_file = self.base_path / 'index.json'

            # 读取现有索引
            if index_file.exists():
                index = json.loads(index_file.read_text(encoding='utf-8'))
            else:
                index = {
                    'version': '1.0.0',
                    'total_documents': 0,
                    'documents': [],
                    'last_updated': None
                }

            # 添加或更新文档
            metadata_dict = asdict(metadata)
            existing_doc = next((doc for doc in index['documents'] if doc['document_id'] == document_id), None)

            if existing_doc:
                # 更新现有文档
                existing_doc.update(metadata_dict)
            else:
                # 添加新文档
                index['documents'].append(metadata_dict)
                index['total_documents'] += 1

            # 更新时间戳
            index['last_updated'] = datetime.now().isoformat()

            # 保存索引
            index_file.write_text(
                json.dumps(index, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )

            logger.info(f"✅ 索引已更新: {document_id}")

        except Exception as e:
            logger.error(f"更新索引失败: {e}")

    async def get_document_index(self) -> Dict:
        """获取文档索引"""
        try:
            index_file = self.base_path / 'index.json'

            if not index_file.exists():
                return {
                    'version': '1.0.0',
                    'total_documents': 0,
                    'documents': [],
                    'last_updated': None
                }

            return json.loads(index_file.read_text(encoding='utf-8'))

        except Exception as e:
            logger.error(f"读取索引失败: {e}")
            return {}

    async def get_document_stats(self) -> Dict:
        """获取文档统计信息"""
        try:
            index = await self.get_document_index()

            if not index or not index.get('documents'):
                return {
                    'total_documents': 0,
                    'total_size': 0,
                    'avg_quality': 0,
                    'by_quality': {},
                    'by_status': {}
                }

            docs = index['documents']

            # 统计
            total_size = sum(doc.get('file_size', 0) for doc in docs)
            avg_quality = sum(doc.get('quality_score', 0) for doc in docs) / len(docs) if docs else 0

            # 按质量分组
            by_quality = {}
            for doc in docs:
                quality = doc.get('quality_level', 'unknown')
                by_quality[quality] = by_quality.get(quality, 0) + 1

            # 按状态分组
            by_status = {}
            for doc in docs:
                status = doc.get('status', 'unknown')
                by_status[status] = by_status.get(status, 0) + 1

            return {
                'total_documents': len(docs),
                'total_size': total_size,
                'total_size_mb': round(total_size / 1024 / 1024, 2),
                'avg_quality': round(avg_quality, 2),
                'by_quality': by_quality,
                'by_status': by_status,
                'with_semantic': sum(1 for doc in docs if doc.get('has_semantic_enhancement')),
                'with_kg': sum(1 for doc in docs if doc.get('has_knowledge_graph')),
                'with_embeddings': sum(1 for doc in docs if doc.get('has_embeddings'))
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    async def search_documents(
        self,
        query: str,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """搜索文档"""
        try:
            index = await self.get_document_index()

            if not index or not index.get('documents'):
                return []

            results = []

            for doc in index['documents']:
                # 应用过滤器
                if filters:
                    if 'quality_level' in filters and doc.get('quality_level') != filters['quality_level']:
                        continue
                    if 'min_quality_score' in filters and doc.get('quality_score', 0) < filters['min_quality_score']:
                        continue
                    if 'has_semantic_enhancement' in filters and doc.get('has_semantic_enhancement') != filters['has_semantic_enhancement']:
                        continue

                # 搜索匹配
                if query.lower() in doc.get('filename', '').lower() or query.lower() in doc.get('document_id', '').lower():
                    results.append(doc)

            return results

        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    async def delete_document(self, document_id: str) -> bool:
        """删除文档"""
        try:
            doc_dir = self.base_path / document_id

            if not doc_dir.exists():
                logger.warning(f"文档不存在: {document_id}")
                return False

            # 删除文档目录
            import shutil
            shutil.rmtree(doc_dir)

            # 更新索引
            index_file = self.base_path / 'index.json'
            if index_file.exists():
                index = json.loads(index_file.read_text(encoding='utf-8'))
                index['documents'] = [doc for doc in index['documents'] if doc['document_id'] != document_id]
                index['total_documents'] = len(index['documents'])
                index['last_updated'] = datetime.now().isoformat()
                index_file.write_text(
                    json.dumps(index, ensure_ascii=False, indent=2),
                    encoding='utf-8'
                )

            logger.info(f"✅ 文档已删除: {document_id}")
            return True

        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            return False


# 全局实例
enhanced_local_storage = EnhancedLocalStorage()
