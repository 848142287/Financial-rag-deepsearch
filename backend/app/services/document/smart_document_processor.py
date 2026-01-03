"""
智能文档处理器 - Smart Document Processor

增强文档切块和预处理功能：
- 结构化切块（针对规范类、技术手册类文档）
- 自动清理目录、附录、噪声内容
- 多级标题识别和元数据提取
- 标题路径注入（增强检索相关性）

作为现有two_stage_markdown_fusion的可选增强模块，不替换现有流程
"""

import re
from dataclasses import dataclass

from app.core.structured_logging import get_structured_logger

logger = get_structured_logger(__name__)

@dataclass
class ChunkMetadata:
    """切块元数据"""
    source: str  # 来源文件
    h1: str = ""  # 一级标题
    h2: str = ""  # 二级标题
    h3: str = ""  # 三级标题
    title_path: str = ""  # 标题路径（如 "规则 > 命名规范 > 变量命名"）
    chunk_type: str = "content"  # content, heading, rule, example
    position: int = 0  # 在文档中的位置

@dataclass
class DocumentChunk:
    """文档切块"""
    content: str  # 增强后的内容（包含标题路径）
    raw_content: str  # 原始内容
    metadata: ChunkMetadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（兼容现有向量存储格式）"""
        return {
            "content": self.content,
            "metadata": {
                "source": self.metadata.source,
                "h1": self.metadata.h1,
                "h2": self.metadata.h2,
                "h3": self.metadata.h3,
                "title_path": self.metadata.title_path,
                "chunk_type": self.metadata.chunk_type,
                "position": self.metadata.position,
                "raw_content": self.raw_content
            }
        }

class SmartDocumentProcessor:
    """
    智能文档处理器

    提供增强的文档切块和预处理功能
    特别适合规范类、技术手册类文档
    """

    def __init__(
        self,
        enable_title_path_injection: bool = True,
        enable_noise_cleanup: bool = True,
        enable_toc_removal: bool = True
    ):
        """
        初始化智能文档处理器

        Args:
            enable_title_path_injection: 是否注入标题路径（提升检索相关性）
            enable_noise_cleanup: 是否清理噪声内容
            enable_toc_removal: 是否移除目录
        """
        self.enable_title_path_injection = enable_title_path_injection
        self.enable_noise_cleanup = enable_noise_cleanup
        self.enable_toc_removal = enable_toc_removal

        logger.info(
            f"智能文档处理器初始化: "
            f"title_path={enable_title_path_injection}, "
            f"noise_cleanup={enable_noise_cleanup}, "
            f"toc_removal={enable_toc_removal}"
        )

    def process(
        self,
        document: str,
        source: str = "",
        processing_mode: str = "auto"
    ) -> List[DocumentChunk]:
        """
        处理文档（智能切块）

        Args:
            document: 文档内容（Markdown格式）
            source: 文档来源标识
            processing_mode: 处理模式
                - "auto": 自动检测文档类型
                - "structured": 结构化切块（适合规范类文档）
                - "simple": 简单切块（适合普通文档）

        Returns:
            DocumentChunk列表
        """
        # 预处理
        if self.enable_toc_removal:
            document = self._remove_toc_and_appendix(document)

        # 选择处理模式
        if processing_mode == "auto":
            # 检测是否为结构化文档
            if self._is_structured_document(document):
                processing_mode = "structured"
            else:
                processing_mode = "simple"

        # 执行切块
        if processing_mode == "structured":
            chunks = self._process_structured(document, source)
        else:
            chunks = self._process_simple(document, source)

        logger.info(f"✅ 文档处理完成: {len(chunks)}个切块 (模式: {processing_mode})")

        return chunks

    def process_file(
        self,
        file_path: str,
        processing_mode: str = "auto"
    ) -> List[DocumentChunk]:
        """
        处理文件（自动加载 + 智能切块）

        Args:
            file_path: 文件路径
            processing_mode: 处理模式

        Returns:
            DocumentChunk列表
        """
        # 加载文档
        load_result = load_document(file_path)

        # 处理文档
        source = load_result.metadata.get("filename", file_path)
        return self.process(load_result.content, source, processing_mode)

    def _is_structured_document(self, content: str) -> bool:
        """
        检测是否为结构化文档

        判断依据：
        - 包含多级标题（#, ##, ###）
        - 包含规约标记（【强制】, 【推荐】等）
        - 包含编号列表
        """
        # 检测多级标题
        heading_pattern = r'^(#{1,3})\s+.+$'
        headings = re.findall(heading_pattern, content, re.MULTILINE)

        # 检测规约标记
        rule_pattern = r'【(强制|推荐|参考)】'
        rules = re.findall(rule_pattern, content)

        # 判断
        return len(headings) >= 5 or len(rules) >= 3

    def _process_structured(
        self,
        content: str,
        source: str
    ) -> List[DocumentChunk]:
        """
        结构化处理（适合规范类、技术手册类文档）

        保留文档结构，注入标题路径
        """
        lines = content.split('\n')
        chunks = []

        current_h1 = ""
        current_h2 = ""
        current_h3 = ""
        buffer = []
        position = 0

        def flush_buffer(chunk_type: str = "content"):
            """将当前缓冲区内容转换为切块"""
            nonlocal position

            if not buffer:
                return

            text = "\n".join(buffer).strip()
            if not text:
                buffer.clear()
                return

            # 构建标题路径
            title_path = self._build_title_path(current_h1, current_h2, current_h3)

            # 增强内容（注入标题路径）
            if self.enable_title_path_injection and title_path:
                enhanced_content = f"{title_path}\n\n{text}"
            else:
                enhanced_content = text

            # 创建切块
            chunk = DocumentChunk(
                content=enhanced_content,
                raw_content=text,
                metadata=ChunkMetadata(
                    source=source,
                    h1=current_h1,
                    h2=current_h2,
                    h3=current_h3,
                    title_path=title_path,
                    chunk_type=chunk_type,
                    position=position
                )
            )

            chunks.append(chunk)
            buffer.clear()
            position += 1

        for line in lines:
            line = line.rstrip()

            # 跳过噪声行
            if self.enable_noise_cleanup and self._should_skip_line(line):
                continue

            # 清理链接
            line = self._clean_links(line)

            # 增强标题标记
            line, is_header = self._enhance_headers(line)

            # 识别标题层级
            if line.startswith("# "):
                flush_buffer()
                current_h1 = line.lstrip("# ").strip()
                current_h2 = ""
                current_h3 = ""
                continue

            if line.startswith("## "):
                flush_buffer()
                current_h2 = line.lstrip("# ").strip()
                current_h3 = ""
                continue

            if line.startswith("### "):
                flush_buffer()
                current_h3 = line.lstrip("# ").strip()
                buffer.append(current_h3)
                continue

            # 识别规约项（如 "1.【强制】..."）
            if re.match(r'^\d+\.\s*【', line.strip()):
                flush_buffer(chunk_type="rule")
                current_h3 = line.strip()
                buffer.append(current_h3)
                continue

            # 普通内容
            buffer.append(line)

        # 处理最后的buffer
        flush_buffer()

        return chunks

    def _process_simple(
        self,
        content: str,
        source: str,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> List[DocumentChunk]:
        """
        简单处理（适合普通文档）

        按段落和字符数切块
        """
        chunks = []

        # 按段落分割
        paragraphs = content.split('\n\n')

        current_chunk = ""
        position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunk = DocumentChunk(
                        content=current_chunk.strip(),
                        raw_content=current_chunk.strip(),
                        metadata=ChunkMetadata(
                            source=source,
                            chunk_type="content",
                            position=position
                        )
                    )
                    chunks.append(chunk)
                    position += 1

                current_chunk = para + "\n\n"

        # 添加最后一个chunk
        if current_chunk.strip():
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                raw_content=current_chunk.strip(),
                metadata=ChunkMetadata(
                    source=source,
                    chunk_type="content",
                    position=position
                )
            )
            chunks.append(chunk)

        return chunks

    def _build_title_path(self, h1: str, h2: str, h3: str) -> str:
        """构建标题路径"""
        parts = [p for p in [h1, h2, h3] if p]
        return " > ".join(parts) if parts else ""

    def _should_skip_line(self, line: str) -> bool:
        """判断是否应跳过该行"""
        line = line.strip()

        if not line:
            return False

        # 页码
        if re.match(r'^\d+/\d+$', line):
            return True

        # 分隔线
        if re.match(r'^[-=_. ]{3,}$', line):
            return True

        # 单个字符或符号
        if len(line) <= 2 and not line.isalnum():
            return True

        return False

    def _clean_links(self, line: str) -> str:
        """清理Markdown链接，保留文字"""
        return re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', line)

    def _enhance_headers(self, line: str) -> Tuple[str, bool]:
        """增强标题结构识别"""
        line_stripped = line.strip()

        # 一级标题：一、编程规约
        if re.match(r'^#*\s*[一二三四五六七八九十]+、', line_stripped):
            clean_text = line_stripped.lstrip('#').strip()
            return f"# {clean_text}", True

        # 二级标题：(一) 命名风格
        if re.match(r'^#*\s*\([一二三四五六七八九十]+\)', line_stripped):
            clean_text = line_stripped.lstrip('#').strip()
            return f"## {clean_text}", True

        # 三级标题：1.【强制】...
        if re.match(r'^\d+\.\s*【', line_stripped):
            return f"### {line_stripped}", True

        return line, line_stripped.startswith('#')

    def _remove_toc_and_appendix(self, text: str) -> str:
        """删除目录和附录"""
        lines = text.split('\n')
        cleaned = []
        skip_toc = False

        # 目录特征：标题后跟大量点和页码
        toc_pattern = r'^.*\.{3,}\s*\d+\s*$'

        for line in lines:
            # 检测目录开始
            if line.strip() in ["# 目录", "目录", "Table of Contents"]:
                skip_toc = True
                continue

            # 检测目录项
            if skip_toc and re.match(toc_pattern, line.strip()):
                continue

            # 检测目录结束（遇到正文标题）
            if skip_toc and re.match(r'^#*\s*[一二三四五六七八九十]+、', line.strip()):
                skip_toc = False

            # 检测附录开始
            if re.match(r'^\s*#*\s*附\s*\d+\s*[:：]', line.strip()):
                break
            if re.match(r'^\s*#*\s*附录\s*\d*\s*[:：]?', line.strip()):
                break

            if not skip_toc:
                cleaned.append(line)

        return '\n'.join(cleaned)

# 便捷函数
def process_document_smart(
    document: str,
    source: str = "",
    enable_title_path_injection: bool = True
) -> List[Dict[str, Any]]:
    """
    智能处理文档的便捷函数

    Args:
        document: 文档内容（Markdown格式）
        source: 文档来源
        enable_title_path_injection: 是否注入标题路径

    Returns:
        切块字典列表（兼容现有格式）
    """
    processor = SmartDocumentProcessor(
        enable_title_path_injection=enable_title_path_injection
    )

    chunks = processor.process(document, source)

    return [chunk.to_dict() for chunk in chunks]

def process_file_smart(
    file_path: str,
    enable_title_path_injection: bool = True
) -> List[Dict[str, Any]]:
    """
    智能处理文件的便捷函数

    Args:
        file_path: 文件路径
        enable_title_path_injection: 是否注入标题路径

    Returns:
        切块字典列表（兼容现有格式）
    """
    processor = SmartDocumentProcessor(
        enable_title_path_injection=enable_title_path_injection
    )

    chunks = processor.process_file(file_path)

    return [chunk.to_dict() for chunk in chunks]
