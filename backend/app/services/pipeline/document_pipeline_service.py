"""
文档处理流水线服务 - 完整版
整合所有步骤：解析 -> 多模态分析 -> 深度汇总 -> 增强Markdown -> 知识图谱 -> 向量存储 -> 本地存储
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import uuid

from app.core.structured_logging import get_structured_logger
from app.core.config import settings
from app.services.llm_service import LLMService
from app.services.multimodal.engines.qwen_vl_engine import QwenVLEngine
from app.services.enhanced_milvus_service import EnhancedMilvusService
from app.services.minio_service import MinioService

logger = get_structured_logger(__name__)

@dataclass
class PipelineResult:
    """流水线处理结果"""
    document_id: str
    filename: str
    file_type: str
    success: bool
    parsing_result: Dict[str, Any] = field(default_factory=dict)
    multimodal_analysis: Dict[str, Any] = field(default_factory=dict)
    deepseek_summary: Dict[str, Any] = field(default_factory=dict)
    enhanced_markdown: str = ""
    knowledge_graph: Dict[str, Any] = field(default_factory=dict)
    vector_storage: Dict[str, Any] = field(default_factory=dict)
    file_storage: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentPipelineService:
    """
    文档处理流水线服务

    完整流程：
    1. 文档上传和基础解析 (openpyxl/python-pptx)
    2. 增强解析 (EnhancedExcelParser/PPTParserWrapper)
    3. 多模态分析 (qwen-vl-plus)
    4. 深度汇总 (deepseek)
    5. 增强Markdown生成
    6. 知识图谱抽取 (neo4j)
    7. 向量存储 (milvus)
    8. 本地文件存储
    """

    def __init__(
        self,
        llm_service: LLMService = None,
        milvus_service: EnhancedMilvusService = None,
        minio_service: MinioService = None
    ):
        self.llm_service = llm_service or LLMService()
        self.milvus_service = milvus_service
        self.minio_service = minio_service

        # 初始化多模态引擎
        self.qwen_vl_engine = QwenVLEngine()

        # 处理配置
        self.enable_multimodal = True
        self.enable_deepseek_summary = True
        self.enable_knowledge_graph = True
        self.enable_vector_storage = True
        self.enable_file_storage = True

        # 本地存储路径
        self.local_storage_path = Path(settings.file_storage_path) / "processed"
        self.local_storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("🚀 文档处理流水线服务初始化完成")
        logger.info(f"  - 多模态分析: {'启用' if self.enable_multimodal else '禁用'}")
        logger.info(f"  - Deepseek汇总: {'启用' if self.enable_deepseek_summary else '禁用'}")
        logger.info(f"  - 知识图谱: {'启用' if self.enable_knowledge_graph else '禁用'}")
        logger.info(f"  - 向量存储: {'启用' if self.enable_vector_storage else '禁用'}")
        logger.info(f"  - 本地存储: {self.local_storage_path}")
        logger.info("=" * 80)

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str = None,
        options: Dict[str, Any] = None
    ) -> PipelineResult:
        """
        处理文档（完整流水线）

        Args:
            file_content: 文件内容（字节）
            filename: 文件名
            document_id: 文档ID（可选，自动生成）
            options: 处理选项

        Returns:
            PipelineResult: 处理结果
        """
        start_time = asyncio.get_event_loop().time()

        # 生成文档ID
        if not document_id:
            document_id = str(uuid.uuid4())

        # 文件类型
        file_ext = Path(filename).suffix.lower()
        file_type = file_ext[1:] if file_ext else "unknown"

        logger.info("=" * 80)
        logger.info(f"📄 开始处理文档: {filename} (ID: {document_id})")
        logger.info("=" * 80)

        result = PipelineResult(
            document_id=document_id,
            filename=filename,
            file_type=file_type,
            success=False
        )

        try:
            # 步骤1: 文档解析
            logger.info("📖 [步骤 1/8] 文档基础解析...")
            parsing_result = await self._parse_document(file_content, filename, document_id)
            result.parsing_result = parsing_result

            # 步骤2: 多模态分析（如果有图片）
            if self.enable_multimodal and parsing_result.get("images"):
                logger.info("🎨 [步骤 2/8] 多模态分析...")
                multimodal_result = await self._analyze_multimodal(
                    parsing_result["images"],
                    document_id
                )
                result.multimodal_analysis = multimodal_result
            else:
                logger.info("⏭️  [步骤 2/8] 跳过多模态分析（无图片）")
                result.multimodal_analysis = {"status": "skipped", "reason": "no_images"}

            # 步骤3: Deepseek深度汇总
            if self.enable_deepseek_summary:
                logger.info("🧠 [步骤 3/8] Deepseek深度汇总...")
                summary_result = await self._deepseek_summary(
                    parsing_result,
                    result.multimodal_analysis,
                    filename
                )
                result.deepseek_summary = summary_result
            else:
                logger.info("⏭️  [步骤 3/8] 跳过深度汇总")
                result.deepseek_summary = {"status": "skipped"}

            # 步骤4: 生成增强Markdown
            logger.info("📝 [步骤 4/8] 生成增强Markdown...")
            enhanced_markdown = await self._generate_enhanced_markdown(
                parsing_result,
                result.multimodal_analysis,
                result.deepseek_summary,
                filename
            )
            result.enhanced_markdown = enhanced_markdown

            # 步骤5: 知识图谱抽取
            if self.enable_knowledge_graph:
                logger.info("🕸️  [步骤 5/8] 知识图谱抽取...")
                kg_result = await self._extract_knowledge_graph(
                    enhanced_markdown,
                    document_id,
                    filename
                )
                result.knowledge_graph = kg_result
            else:
                logger.info("⏭️  [步骤 5/8] 跳过知识图谱抽取")
                result.knowledge_graph = {"status": "skipped"}

            # 步骤6: 向量存储（原有）
            if self.enable_vector_storage and self.milvus_service:
                logger.info("🔍 [步骤 6/8] 向量存储...")
                vector_result = await self._store_vectors(
                    enhanced_markdown,
                    document_id,
                    filename,
                    parsing_result
                )
                result.vector_storage = vector_result
            else:
                logger.info("⏭️  [步骤 6/8] 跳过向量存储")
                result.vector_storage = {"status": "skipped"}

            # 步骤7: 分层索引构建（新增）
            logger.info("📚 [步骤 7/8] 分层索引构建...")
            try:
                from app.services.hierarchical_index import get_hierarchical_index_pipeline_integration
                pipeline_integration = get_hierarchical_index_pipeline_integration()

                hierarchical_index = await pipeline_integration.build_index_from_pipeline(
                    document_id=document_id,
                    markdown_content=parsing_result.get("markdown_content", ""),
                    deepseek_summary=result.deepseek_summary
                )

                result.hierarchical_index = {
                    "status": "success",
                    "total_chapters": len(hierarchical_index.chapters),
                    "total_chunks": len(hierarchical_index.chunks),
                    "processing_time": hierarchical_index.processing_time
                }

                logger.info(f"  ✅ 分层索引构建成功: 章节={len(hierarchical_index.chapters)}, 片段={len(hierarchical_index.chunks)}")

            except Exception as e:
                logger.warning(f"⚠️ 分层索引构建失败（不影响主流程）: {str(e)}")
                result.hierarchical_index = {
                    "status": "failed",
                    "error": str(e)
                }

            # 步骤8: 本地文件存储
            if self.enable_file_storage:
                logger.info("💾 [步骤 8/8] 本地文件存储...")
                file_result = await self._store_locally(
                    result,
                    document_id,
                    filename
                )
                result.file_storage = file_result
            else:
                logger.info("⏭️  [步骤 8/8] 跳过本地文件存储")
                result.file_storage = {"status": "skipped"}

            # 计算处理时间
            result.processing_time = asyncio.get_event_loop().time() - start_time
            result.success = True

            logger.info("=" * 80)
            logger.info(f"✅ 文档处理完成！总耗时: {result.processing_time:.2f}秒")
            logger.info("=" * 80)

            return result

        except Exception as e:
            result.processing_time = asyncio.get_event_loop().time() - start_time
            result.error_message = str(e)
            logger.error(f"❌ 文档处理失败: {str(e)}")
            return result

    async def _parse_document(
        self,
        file_content: bytes,
        filename: str,
        document_id: str
    ) -> Dict[str, Any]:
        """步骤1: 文档解析"""
        from app.services.parsing.document_parsing_service import DocumentParsingService

        # 创建解析服务实例
        parsing_service = DocumentParsingService(services={})

        # 解析文档
        text_content, markdown_content, parse_result = await parsing_service.parse_document(
            file_content=file_content,
            filename=filename,
            document_id=document_id
        )

        # 提取图片（如果有）
        images = parse_result.get("images", [])

        return {
            "text_content": text_content,
            "markdown_content": markdown_content,
            "parse_result": parse_result,
            "images": images,
            "metadata": {
                "filename": filename,
                "document_id": document_id,
                "parsed_at": datetime.now().isoformat()
            }
        }

    async def _analyze_multimodal(
        self,
        images: List[str],
        document_id: str
    ) -> Dict[str, Any]:
        """步骤2: 多模态分析"""
        try:
            # 批量分析图片
            results = await self.qwen_vl_engine.analyze_images_batch(
                image_paths=images,
                document_id=document_id,
                analysis_type="general"
            )

            # 整合结果
            all_analyses = []
            for i, result in enumerate(results):
                if "error" not in result:
                    all_analyses.append({
                        "image_path": images[i],
                        "analysis": result.get("full_analysis", ""),
                        "metadata": result.get("metadata", {})
                    })

            return {
                "status": "success",
                "analyzed_count": len(all_analyses),
                "total_images": len(images),
                "analyses": all_analyses
            }

        except Exception as e:
            logger.error(f"多模态分析失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "analyzed_count": 0,
                "total_images": len(images)
            }

    async def _deepseek_summary(
        self,
        parsing_result: Dict[str, Any],
        multimodal_analysis: Dict[str, Any],
        filename: str
    ) -> Dict[str, Any]:
        """步骤3: Deepseek深度汇总（基于规则汇总 + Deepseek检查）"""
        try:
            # 步骤3.1: 基于规则的章节汇总
            logger.info("  📋 基于规则的章节汇总...")
            rule_based_summary = self._rule_based_summary(
                parsing_result,
                multimodal_analysis
            )

            # 步骤3.2: Deepseek检查和增强
            logger.info("  🤖 Deepseek检查和增强...")
            enhanced_summary = await self._deepseek_enhance(
                rule_based_summary,
                parsing_result
            )

            return {
                "status": "success",
                "rule_based_summary": rule_based_summary,
                "enhanced_summary": enhanced_summary,
                "model": "deepseek-chat (enhancement)",
                "created_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Deepseek汇总失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    def _rule_based_summary(
        self,
        parsing_result: Dict[str, Any],
        multimodal_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        基于规则的章节汇总
        快速提取结构化信息，减少对LLM的依赖
        """
        markdown_content = parsing_result.get("markdown_content", "")

        summary = {
            "sections": [],
            "key_points": [],
            "statistics": {},
            "entities": [],
            "images_analysis": []
        }

        # 1. 按章节拆分
        sections = self._extract_sections(markdown_content)
        summary["sections"] = sections

        # 2. 提取关键信息
        for section in sections:
            # 提取数字和统计数据
            numbers = self._extract_numbers(section["content"])
            if numbers:
                summary["key_points"].extend(numbers)

            # 提取日期
            dates = self._extract_dates(section["content"])
            if dates:
                summary["key_points"].extend(dates)

            # 提取关键词
            keywords = self._extract_keywords(section["content"])
            if keywords:
                summary["entities"].extend(keywords)

        # 3. 统计信息
        summary["statistics"] = {
            "total_sections": len(sections),
            "total_words": len(markdown_content),
            "has_images": multimodal_analysis.get("status") == "success",
            "image_count": multimodal_analysis.get("analyzed_count", 0)
        }

        # 4. 图片分析汇总
        if multimodal_analysis.get("status") == "success":
            for analysis in multimodal_analysis.get("analyses", []):
                summary["images_analysis"].append({
                    "image": analysis.get("image_path", ""),
                    "summary": analysis.get("analysis", "")[:200]
                })

        return summary

    def _extract_sections(self, markdown_content: str) -> List[Dict[str, Any]]:
        """提取章节结构"""
        import re

        sections = []
        lines = markdown_content.split('\n')

        current_section = {"title": "概述", "level": 0, "content": []}

        for line in lines:
            # 检测标题
            if line.startswith('#'):
                # 保存上一个章节
                if current_section["content"]:
                    current_section["content"] = '\n'.join(current_section["content"])
                    sections.append(current_section)

                # 创建新章节
                level = len(line) - len(line.lstrip('#'))
                title = line.strip('#').strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "content": []
                }
            else:
                current_section["content"].append(line)

        # 保存最后一个章节
        if current_section["content"]:
            current_section["content"] = '\n'.join(current_section["content"])
            sections.append(current_section)

        return sections

    def _extract_numbers(self, text: str) -> List[Dict[str, Any]]:
        """提取数字和统计信息"""
        import re

        # 匹配数字（包括百分比、金额等）
        patterns = [
            r'(\d+\.?\d*)%',  # 百分比
            r'(\d+\.?\d*)\s*(万|亿|千)万?',  # 中文单位
            r'\$?(\d{1,3}(,\d{3})*(\.\d+)?)',  # 金额
            r'(\d{4}年?\d{1,2}月?\d{1,2}日?)'  # 日期
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                numbers.append({
                    "type": "number",
                    "value": str(match[0] if isinstance(match, tuple) else match),
                    "context": text[max(0, text.find(str(match))-20):text.find(str(match))+50]
                })

        return numbers[:10]  # 限制返回数量

    def _extract_dates(self, text: str) -> List[Dict[str, Any]]:
        """提取日期"""
        import re

        date_patterns = [
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'(\d{4})-(\d{1,2})-(\d{1,2})',
            r'(\d{1,2})/(\d{1,2})/(\d{4})'
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                dates.append({
                    "type": "date",
                    "value": str(match),
                    "context": text[max(0, text.find(str(match))-20):text.find(str(match))+50]
                })

        return dates[:5]

    def _extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """提取关键词（简单的词频统计）"""
        import re
        from collections import Counter

        # 简单的中文分词（按字符）
        chinese_words = re.findall(r'[\u4e00-\u9fa5]{2,4}', text)

        # 统计词频
        word_freq = Counter(chinese_words)

        # 返回高频词
        keywords = []
        for word, freq in word_freq.most_common(10):
            if freq >= 2:  # 至少出现2次
                keywords.append({
                    "type": "keyword",
                    "word": word,
                    "frequency": freq
                })

        return keywords

    async def _deepseek_enhance(
        self,
        rule_based_summary: Dict[str, Any],
        parsing_result: Dict[str, Any]
    ) -> str:
        """
        Deepseek检查和增强
        主要任务：
        1. 检查规则汇总的准确性
        2. 识别复杂的关系和模式
        3. 提供洞察和建议
        """
        prompt = f"""请检查和增强以下基于规则提取的文档摘要：

## 规则提取结果
章节数量: {rule_based_summary['statistics'].get('total_sections', 0)}
图片数量: {rule_based_summary['statistics'].get('image_count', 0)}

### 主要章节
{chr(10).join([f"- {s['title']}" for s in rule_based_summary.get('sections', [])[:5]])}

### 关键信息样本
{json.dumps(rule_based_summary.get('key_points', [])[:5], ensure_ascii=False)}

## 要求
请提供：
1. **准确性检查**：上述提取信息是否有明显错误？
2. **关键洞察**：从这些信息中能发现什么重要趋势或问题？
3. **改进建议**：还需要补充哪些关键信息？
4. **风险提示**：有什么需要特别注意的风险点？

请用简洁的Markdown格式输出，重点突出问题和建议。"""

        enhanced = await self.llm_service.simple_chat(
            prompt=prompt,
            system_prompt="你是文档分析专家，负责检查和优化基于规则提取的信息。",
            temperature=0.3
        )

        return enhanced

    def _get_summary_system_prompt(self) -> str:
        """获取汇总系统提示词"""
        return """你是一个专业的金融文档分析专家。你的任务是对文档内容进行深度分析和汇总。

请提供：
1. 文档核心内容概述
2. 关键数据和信息提取
3. 重要结论和观点
4. 风险提示和注意事项

输出格式：
- 使用清晰的标题和分段
- 突出关键信息
- 保持客观和专业"""

    def _build_summary_prompt(
        self,
        parsing_result: Dict[str, Any],
        multimodal_analysis: Dict[str, Any]
    ) -> str:
        """构造汇总提示词"""
        prompt = f"""请对以下文档内容进行深度分析汇总：

## 文档内容
{parsing_result.get('markdown_content', '')[:3000]}

## 图片分析
"""

        if multimodal_analysis.get("status") == "success":
            for analysis in multimodal_analysis.get("analyses", [])[:5]:
                prompt += f"\n{analysis.get('analysis', '')}\n"

        prompt += """

## 要求
请提供：
1. 文档概述（主要内容和目的）
2. 关键信息提取（重要数据、时间、人物、事件）
3. 核心观点和结论
4. 需要关注的重点

请用清晰的Markdown格式输出。"""

        return prompt

    async def _generate_enhanced_markdown(
        self,
        parsing_result: Dict[str, Any],
        multimodal_analysis: Dict[str, Any],
        deepseek_summary: Dict[str, Any],
        filename: str
    ) -> str:
        """步骤4: 生成增强Markdown"""
        markdown_parts = []

        # 标题
        markdown_parts.append(f"# {filename}\n")
        markdown_parts.append(f"**处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        markdown_parts.append("---\n\n")

        # 原始内容
        markdown_parts.append("## 原始内容\n\n")
        markdown_parts.append(parsing_result.get("markdown_content", "")[:5000])
        markdown_parts.append("\n\n---\n\n")

        # 多模态分析
        if multimodal_analysis.get("status") == "success":
            markdown_parts.append("## 图片分析\n\n")
            for i, analysis in enumerate(multimodal_analysis.get("analyses", []), 1):
                markdown_parts.append(f"### 图片 {i}\n\n")
                markdown_parts.append(f"{analysis.get('analysis', '')}\n\n")

        # Deepseek汇总
        if deepseek_summary.get("status") == "success":
            markdown_parts.append("## AI深度汇总\n\n")
            markdown_parts.append(deepseek_summary.get("summary", ""))
            markdown_parts.append("\n\n")

        return "\n".join(markdown_parts)

    async def _extract_knowledge_graph(
        self,
        enhanced_markdown: str,
        document_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """步骤5: 知识图谱抽取"""
        try:
            # 使用Deepseek抽取实体和关系
            prompt = f"""请从以下文档中抽取实体和关系，构建知识图谱。

## 文档内容
{enhanced_markdown[:4000]}

## 要求
请以JSON格式返回：
{{
  "entities": [
    {{"name": "实体名称", "type": "类型（Company/Person/Stock等）", "confidence": 0.9}}
  ],
  "relations": [
    {{"source": "实体1", "target": "实体2", "type": "关系类型", "confidence": 0.8}}
  ]
}}"""

            kg_data = await self.llm_service.structured_completion(
                prompt=prompt,
                system_prompt="你是知识图谱抽取专家，擅长识别实体和关系。",
                schema={
                    "entities": "实体列表",
                    "relations": "关系列表"
                }
            )

            # TODO: 存储到Neo4j
            # 目前Neo4j服务被禁用，先返回数据

            return {
                "status": "success",
                "entities": kg_data.get("entities", []),
                "relations": kg_data.get("relations", []),
                "entity_count": len(kg_data.get("entities", [])),
                "relation_count": len(kg_data.get("relations", [])),
                "note": "Neo4j存储未启用，仅返回抽取结果"
            }

        except Exception as e:
            logger.error(f"知识图谱抽取失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _store_vectors(
        self,
        enhanced_markdown: str,
        document_id: str,
        filename: str,
        parsing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """步骤6: 向量存储"""
        try:
            # TODO: 实现向量化并存储到Milvus
            # 当前Milvus服务存在，但需要实现embedding

            return {
                "status": "success",
                "note": "向量存储功能待完善",
                "document_id": document_id
            }

        except Exception as e:
            logger.error(f"向量存储失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

    async def _store_locally(
        self,
        result: PipelineResult,
        document_id: str,
        filename: str
    ) -> Dict[str, Any]:
        """步骤7: 本地文件存储"""
        try:
            # 创建文档目录
            doc_dir = self.local_storage_path / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # 保存增强Markdown
            markdown_path = doc_dir / f"{filename}.md"
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(result.enhanced_markdown)

            # 保存完整结果JSON
            json_path = doc_dir / f"{filename}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "document_id": document_id,
                    "filename": filename,
                    "processing_time": result.processing_time,
                    "parsing_result": result.parsing_result,
                    "multimodal_analysis": result.multimodal_analysis,
                    "deepseek_summary": result.deepseek_summary,
                    "knowledge_graph": result.knowledge_graph,
                    "vector_storage": result.vector_storage,
                    "processed_at": datetime.now().isoformat()
                }, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "markdown_path": str(markdown_path),
                "json_path": str(json_path),
                "document_dir": str(doc_dir)
            }

        except Exception as e:
            logger.error(f"本地存储失败: {str(e)}")
            return {
                "status": "failed",
                "error": str(e)
            }

# 全局单例
_document_pipeline_service = None

def get_document_pipeline_service() -> DocumentPipelineService:
    """获取文档流水线服务单例"""
    global _document_pipeline_service
    if _document_pipeline_service is None:
        _document_pipeline_service = DocumentPipelineService()
    return _document_pipeline_service
