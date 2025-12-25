"""
多模态文档解析器主类
协调各个引擎进行联合解析
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import tempfile
import base64

from ..engines.mineru_engine import MineruEngine
from ..engines.qwen_vl_engine import QwenVLEngine
from ..processors.structure_analyzer import StructureAnalyzer
from ..processors.content_aggregator import ContentAggregator
from ..evaluators.integrity_evaluator import IntegrityEvaluator
from ..repairers.auto_repairer import AutoRepairer

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """内容类型枚举"""
    CHAPTER = "chapter"
    TEXT = "text"
    IMAGE = "image"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"
    FOOTNOTE = "footnote"
    HEADER = "header"
    FOOTER = "footer"


@dataclass
class ContentBlock:
    """内容块"""
    id: str
    content_type: ContentType
    content: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    page_number: int = 0
    chapter_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = None


@dataclass
class Chapter:
    """章节结构"""
    id: str
    title: str
    level: int  # 1: 一级标题, 2: 二级标题, ...
    start_page: int
    end_page: int
    sub_chapters: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """解析后的文档"""
    document_id: str
    title: str
    total_pages: int
    chapters: List[Chapter]
    content_blocks: List[ContentBlock]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parsing_stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsingConfig:
    """解析配置"""
    use_mineru: bool = True
    use_qwen_vl_ocr: bool = True
    use_qwen_vl_max: bool = True
    prefer_high_quality: bool = True
    enable_auto_repair: bool = True
    integrity_threshold: float = 0.8
    max_repair_attempts: int = 3
    parallel_processing: bool = True
    temp_dir: Optional[str] = None


class MultimodalDocumentParser:
    """多模态文档解析器"""

    def __init__(self, config: Optional[ParsingConfig] = None):
        """
        初始化多模态文档解析器

        Args:
            config: 解析配置
        """
        self.config = config or ParsingConfig()

        # 初始化各引擎
        self.mineru_engine = MineruEngine() if self.config.use_mineru else None
        self.qwen_vl_engine = QwenVLEngine() if (self.config.use_qwen_vl_ocr or self.config.use_qwen_vl_max) else None

        # 初始化处理器
        self.structure_analyzer = StructureAnalyzer()
        self.content_aggregator = ContentAggregator()
        self.integrity_evaluator = IntegrityEvaluator()
        self.auto_repairer = AutoRepairer() if self.config.enable_auto_repair else None

        # 创建临时目录
        self.temp_dir = self.config.temp_dir or tempfile.mkdtemp(prefix="multimodal_parser_")
        os.makedirs(self.temp_dir, exist_ok=True)

        logger.info(f"多模态文档解析器初始化完成，临时目录: {self.temp_dir}")

    async def parse_document(self, file_path: str) -> ParsedDocument:
        """
        解析文档

        Args:
            file_path: 文件路径

        Returns:
            解析后的文档对象
        """
        start_time = asyncio.get_event_loop().time()
        document_id = Path(file_path).stem

        logger.info(f"开始解析文档: {file_path}")

        try:
            # 第一步：多引擎并行解析
            raw_results = await self._parallel_parse(file_path, document_id)

            # 第二步：结构化分析
            chapters = await self.structure_analyzer.analyze_structure(raw_results)

            # 第三步：内容聚合
            content_blocks = await self.content_aggregator.aggregate_content(
                raw_results, chapters
            )

            # 第四步：创建文档对象
            document = ParsedDocument(
                document_id=document_id,
                title=self._extract_title(content_blocks, chapters),
                total_pages=self._get_total_pages(raw_results),
                chapters=chapters,
                content_blocks=content_blocks,
                metadata=self._extract_metadata(raw_results),
                parsing_stats=self._calculate_parsing_stats(raw_results, chapters, content_blocks)
            )

            # 第五步：完整性评估
            integrity_score = await self.integrity_evaluator.evaluate_integrity(document)

            logger.info(f"文档完整性评分: {integrity_score:.3f}")

            # 第六步：自动修复（如果需要）
            if integrity_score < self.config.integrity_threshold and self.auto_repairer:
                document = await self.auto_repairer.repair_document(
                    document, file_path, integrity_score
                )
                logger.info("文档自动修复完成")

            # 记录处理时间
            processing_time = asyncio.get_event_loop().time() - start_time
            document.parsing_stats['processing_time'] = processing_time

            logger.info(f"文档解析完成，耗时: {processing_time:.2f}秒")

            return document

        except Exception as e:
            logger.error(f"文档解析失败: {str(e)}")
            raise

    async def _parallel_parse(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """并行解析文档"""
        tasks = []

        # Mineru解析任务
        if self.mineru_engine:
            tasks.append(self._run_mineru_parse(file_path, document_id))

        # Qwen-VL-OCR解析任务
        if self.qwen_vl_engine and self.config.use_qwen_vl_ocr:
            tasks.append(self._run_qwen_vl_ocr(file_path, document_id))

        # Qwen-VL-Max解析任务
        if self.qwen_vl_engine and self.config.use_qwen_vl_max:
            tasks.append(self._run_qwen_vl_max(file_path, document_id))

        # 并行执行所有解析任务
        if self.config.parallel_processing and tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            results = []
            for task in tasks:
                try:
                    result = await task
                    results.append(result)
                except Exception as e:
                    logger.error(f"解析任务失败: {str(e)}")
                    results.append(e)

        # 整理结果
        raw_results = {
            'mineru_results': None,
            'qwen_ocr_results': None,
            'qwen_max_results': None
        }

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"解析引擎 {i} 出错: {str(result)}")
                continue

            if self.mineru_engine and i == 0:
                raw_results['mineru_results'] = result
            elif self.config.use_qwen_vl_ocr and self.mineru_engine and i == 1:
                raw_results['qwen_ocr_results'] = result
            elif self.config.use_qwen_vl_ocr and not self.mineru_engine and i == 0:
                raw_results['qwen_ocr_results'] = result
            elif self.config.use_qwen_vl_max:
                raw_results['qwen_max_results'] = result

        return raw_results

    async def _run_mineru_parse(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """运行Mineru解析"""
        try:
            output_dir = os.path.join(self.temp_dir, f"{document_id}_mineru")
            os.makedirs(output_dir, exist_ok=True)

            result = await self.mineru_engine.parse_document(file_path, output_dir)
            logger.info(f"Mineru解析完成，提取内容块: {len(result.get('content_blocks', []))}")
            return result

        except Exception as e:
            logger.error(f"Mineru解析失败: {str(e)}")
            return {}

    async def _run_qwen_vl_ocr(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """运行Qwen-VL-OCR解析"""
        try:
            result = await self.qwen_vl_engine.parse_with_ocr(file_path, document_id)
            logger.info(f"Qwen-VL-OCR解析完成，识别文本块: {len(result.get('text_blocks', []))}")
            return result

        except Exception as e:
            logger.error(f"Qwen-VL-OCR解析失败: {str(e)}")
            return {}

    async def _run_qwen_vl_max(self, file_path: str, document_id: str) -> Dict[str, Any]:
        """运行Qwen-VL-Max解析"""
        try:
            result = await self.qwen_vl_engine.parse_with_vl_max(file_path, document_id)
            logger.info(f"Qwen-VL-Max解析完成，分析结果: {len(result.get('analysis_results', []))}")
            return result

        except Exception as e:
            logger.error(f"Qwen-VL-Max解析失败: {str(e)}")
            return {}

    def _extract_title(self, content_blocks: List[ContentBlock], chapters: List[Chapter]) -> str:
        """提取文档标题"""
        # 优先从章节中获取一级标题
        for chapter in chapters:
            if chapter.level == 1 and chapter.title:
                return chapter.title

        # 从内容块中查找标题
        for block in content_blocks:
            if block.content_type == ContentType.HEADER and block.content:
                return block.content.strip()

        # 返回默认标题
        return "未命名文档"

    def _get_total_pages(self, raw_results: Dict[str, Any]) -> int:
        """获取总页数"""
        max_pages = 0

        for engine_name, results in raw_results.items():
            if results and isinstance(results, dict):
                pages = results.get('total_pages', 0)
                if pages > max_pages:
                    max_pages = pages

        return max_pages

    def _extract_metadata(self, raw_results: Dict[str, Any]) -> Dict[str, Any]:
        """提取元数据"""
        metadata = {
            'engines_used': [],
            'parsing_timestamp': asyncio.get_event_loop().time(),
            'config': {
                'use_mineru': self.config.use_mineru,
                'use_qwen_vl_ocr': self.config.use_qwen_vl_ocr,
                'use_qwen_vl_max': self.config.use_qwen_vl_max
            }
        }

        # 收集各引擎的元数据
        if raw_results.get('mineru_results'):
            metadata['engines_used'].append('mineru')
            metadata.update(raw_results['mineru_results'].get('metadata', {}))

        if raw_results.get('qwen_ocr_results'):
            metadata['engines_used'].append('qwen_vl_ocr')
            metadata.update(raw_results['qwen_ocr_results'].get('metadata', {}))

        if raw_results.get('qwen_max_results'):
            metadata['engines_used'].append('qwen_vl_max')
            metadata.update(raw_results['qwen_max_results'].get('metadata', {}))

        return metadata

    def _calculate_parsing_stats(
        self,
        raw_results: Dict[str, Any],
        chapters: List[Chapter],
        content_blocks: List[ContentBlock]
    ) -> Dict[str, Any]:
        """计算解析统计信息"""
        stats = {
            'total_chapters': len(chapters),
            'total_content_blocks': len(content_blocks),
            'content_type_distribution': {},
            'engine_contributions': {}
        }

        # 内容类型分布
        content_types = [block.content_type.value for block in content_blocks]
        from collections import Counter
        type_counter = Counter(content_types)
        stats['content_type_distribution'] = dict(type_counter)

        # 引擎贡献统计
        for block in content_blocks:
            engine = block.metadata.get('source_engine', 'unknown')
            stats['engine_contributions'][engine] = stats['engine_contributions'].get(engine, 0) + 1

        return stats

    async def export_to_json(self, document: ParsedDocument, output_path: str):
        """导出为JSON格式"""
        try:
            # 转换为可序列化的格式
            export_data = {
                'document_id': document.document_id,
                'title': document.title,
                'total_pages': document.total_pages,
                'chapters': [
                    {
                        'id': chapter.id,
                        'title': chapter.title,
                        'level': chapter.level,
                        'start_page': chapter.start_page,
                        'end_page': chapter.end_page,
                        'sub_chapters': chapter.sub_chapters,
                        'blocks': chapter.blocks,
                        'metadata': chapter.metadata
                    }
                    for chapter in document.chapters
                ],
                'content_blocks': [
                    {
                        'id': block.id,
                        'content_type': block.content_type.value,
                        'content': block.content,
                        'bbox': block.bbox,
                        'page_number': block.page_number,
                        'chapter_id': block.chapter_id,
                        'confidence': block.confidence,
                        'metadata': block.metadata
                    }
                    for block in document.content_blocks
                ],
                'metadata': document.metadata,
                'parsing_stats': document.parsing_stats
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)

            logger.info(f"文档已导出到: {output_path}")

        except Exception as e:
            logger.error(f"导出失败: {str(e)}")
            raise

    def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"临时目录已清理: {self.temp_dir}")
        except Exception as e:
            logger.error(f"清理临时目录失败: {str(e)}")

    def __del__(self):
        """析构函数"""
        self.cleanup()